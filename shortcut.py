from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from typing import Callable, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from typing import Optional
import torch.nn as nn
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast,CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm, LlamaModel, BaseModelOutputWithPast
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
import matplotlib.pyplot as plt
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map


import argparse
import data_utils
import utils

import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model to load")
parser.add_argument("--auth_token", type = str, help = "authorisation token used to load huggingface models")
parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset to calibrate and calculate perplexity on.",
        choices=["wikitext2", "ptb", "c4", "alpaca","glue","hotpot","pubmedqa","medqa_4options","billsum","multilegalpile","mathqa","sciq"],
        default="wikitext2",
    )
parser.add_argument(
    "--cal-nsamples",
    type=int,
    help="Number of samples of the calibration data to load.",
    default=128,
)


parser.add_argument("--cal-batch-size", type=int, default=8, help="Batch size for loading the calibration data.")
parser.add_argument("--cal-max-seqlen", type=int, default=1024, help="Maximum sequence length for the calibration data.")
parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")

parser.add_argument("--hook_level", type=int, default=0, help="0 is for decoder block and 1 is for nn.linear")
parser.add_argument("--sparsity", type=float, default=0.3, help="sparsity")

parser.add_argument("--ppl-eval-batch-size", type=int, default=4, help="Batch size for evaluating the perplexity.")

parser.add_argument("--type2_engg", type=int, default=0, help="zero means not actual pruning and one means actual pruning")
parser.add_argument("--sparsity_allocation", type=int, default=1, help="1 means optimal allocation by measuring cosine similarity and 0 is uniform")
args = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained(args.model,use_auth_token = args.auth_token)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    args.model,
    device_map="cuda:0",              
    torch_dtype=torch.float32,             
    use_auth_token = args.auth_token
)
config = LlamaConfig.from_pretrained(
    args.model,
    use_auth_token=args.auth_token
)

dataset = data_utils.get_dataset("sciq")
train_dataset, test_dataset = dataset["train"], dataset["test"]
train_loader = data_utils.prepare_dataloader(
    dataset=train_dataset,
    tokenizer=tokenizer,
    max_seqlen=args.cal_max_seqlen,
    batch_size=args.cal_batch_size,
    nsamples=args.cal_nsamples,
    varied_seqlen=args.varied_seqlen,
    seed=args.seed,
)
test_loader = data_utils.prepare_test_dataloader(
        dataset=test_dataset, tokenizer=tokenizer, batch_size=args.ppl_eval_batch_size
    )

class mymodel(LlamaModel):
    def __init__(self,model,config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers[:5]
        self.rotary_emb = model.model.rotary_emb
        self.norm = model.model.norm
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None):

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)
        
        cache_position: torch.Tensor = torch.arange(0, 0 + inputs_embeds.shape[1], device=inputs_embeds.device)
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        device = hidden_states.device
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = tuple(v.to(device) for v in position_embeddings)
        
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings
            )

        
        
        return hidden_states


class shortcut_model(LlamaModel):
    def __init__(self,model,config):
        super().__init__(config)
        device = next(model.parameters()).device
        self.config = config
        self.embed_tokens = model.model.embed_tokens
        self.layers = LlamaDecoderLayer(config,69).to(device)
        self.rotary_emb = model.model.rotary_emb
        self.norm = model.model.norm

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None):

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)
        
        cache_position: torch.Tensor = torch.arange(0, 0 + inputs_embeds.shape[1], device=inputs_embeds.device)
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        device = hidden_states.device
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        position_embeddings = tuple(v.to(device) for v in position_embeddings)
        
        
        hidden_states = self.layers(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings
        )

        
        
        return hidden_states


# teacher_model = mymodel(model,config)
# student_model = shortcut_model(model,config)


class kd_trainer():
    def __init__(self, teacher_model, student_model):
        self.teacher_model = teacher_model
        self.student_model = student_model
        
    def training(self):
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(self.student_model.parameters(), lr=1e-4)
        device = next(self.teacher_model.parameters()).device


        loss__ = []
        for _ in tqdm(range(3)):
            loss_ = []
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.no_grad():
                    teacher_logits = self.teacher_model(input_ids, attention_mask=attention_mask)
                student_logits = self.student_model(input_ids, attention_mask=attention_mask)

                loss = self.distill_hidden_loss(student_logits, teacher_logits)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_.append(loss.item())
            # print("Loss:", torch.mean(torch.tensor(loss_),dtype = torch.float32))
            loss__.append(torch.mean(torch.tensor(loss_),dtype = torch.float32))

    def distill_hidden_loss(self,student_hidden, teacher_hidden):
        return F.mse_loss(student_hidden, teacher_hidden)

# trainer = kd_trainer(teacher_model, student_model)
# trainer.training()


class Lego_Model(LlamaModel):
    def __init__(self,model,config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = model.model.embed_tokens
        self.layers = model.model.layers
        self.norm = model.model.norm
        self.rotary_emb = model.model.rotary_emb

    

class Causal_Lego_Model(LlamaForCausalLM):
    def __init__(self,model, config):
        super().__init__(config)
        self.model = Lego_Model(model,config)
        self.vocab_size = config.vocab_size
        self.lm_head = model.lm_head
        self.device_ = model.device
        import pdb;pdb.set_trace()

    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Cache] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     cache_position: Optional[torch.LongTensor] = None,
    #     logits_to_keep: Union[int, torch.Tensor] = 0,
    #     # **kwargs: Unpack[TransformersKwargs],
    # ) -> CausalLMOutputWithPast:
    #     outputs: BaseModelOutputWithPast = self.model(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         position_ids=position_ids,
    #         past_key_values=past_key_values,
    #         inputs_embeds=inputs_embeds,
    #         use_cache=use_cache,
    #         cache_position=cache_position,
    #         # **kwargs,
    #     )
    #     hidden_states = outputs.last_hidden_state.to(self.device)
    #     # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    #     slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    #     logits = self.lm_head(hidden_states[:, slice_indices, :])

    #     loss = None
    #     if labels is not None:
    #         loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    #     return CausalLMOutputWithPast(
    #         loss=loss,
    #         logits=logits,
    #         past_key_values=outputs.past_key_values,
    #         hidden_states=outputs.hidden_states,
    #         attentions=outputs.attentions,
    #     )


model1 = Causal_Lego_Model(model,config)



dataset_ppl = utils.evaluate_ppl(model1, model.config.pad_token_id, test_loader)
print(dataset_ppl)

# lm = HFLM(pretrained=model1, tokenizer=tokenizer)
# results = evaluator.simple_evaluate(
#     model=lm,
#     tasks=[args.cal_dataset],
#     batch_size=4
# )
# print(make_table(results))

# lm = HFLM(pretrained=model, tokenizer=tokenizer)
# results = evaluator.simple_evaluate(
#     model=lm,
#     tasks=[args.cal_dataset],
#     batch_size=4
# )
# print(make_table(results))














# plt.figure(figsize=(12,7))
# plt.plot(loss__)


# plt.savefig("kd.png")

