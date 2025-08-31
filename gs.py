
import argparse

import logging
import torch

import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm

import data_utils
from utils import cleanup_memory




class functor_gs():
    def __init__(self):
        self.storage = {}
    
    def __call__(self,module,input,output):

        dot = (input[0].detach().cpu()*output[0].detach().cpu()).sum(dim=-1)
        xin_norm = input[0].detach().cpu().norm(dim=-1)
        xout_norm = output[0].detach().cpu().norm(dim=-1)
        cos_sim = dot / (xin_norm * xout_norm + 1e-8)
        mean_cos_sim = cos_sim.mean()
        s = 1 - mean_cos_sim
        self.storage[module._name] = s.item()

class global_sparsity_allocation():
    def __init__(self, language_model, data, sparsity):
        self.model = language_model
        self.data = data
        self.sparsity = sparsity
        self.gs = self.optimal_sparsity_allocation()
    def optimal_sparsity_allocation(self):

        list_layers = list(self.model.model.layers.named_children())
        func_ = functor_gs()
        hooks = []
        for module in list_layers:
            module[1]._name = module[0]
            hook = module[1].register_forward_hook(func_)
            hooks.append(hook)

        with torch.no_grad():
            _ = self.model(**self.data)
        
        for hook in hooks:
            hook.remove()

        vals = list(func_.storage.values())
        v = [-1*(i/0.1) for i in vals]

        v= torch.tensor(v)
        soft = torch.softmax(v,dim=-1)

        gs = 32*self.sparsity*soft
        return gs


if __name__=="main":
    logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type = str, help = "model to load")
    parser.add_argument("--auth_token", type = str, help = "authorisation token used to load huggingface models")
    parser.add_argument(
            "--cal-dataset",
            type=str,
            help="Dataset to calibrate and calculate perplexity on.",
            choices=["wikitext2", "ptb", "c4", "alpaca","glue","hotpot","pubmed"],
            default="wikitext2",
        )
    parser.add_argument(
        "--cal-nsamples",
        type=int,
        help="Number of samples of the calibration data to load.",
        default=128,
    )

    parser.add_argument("--cal-batch-size", type=int, default=16, help="Batch size for loading the calibration data.")
    parser.add_argument("--cal-max-seqlen", type=int, default=1024, help="Maximum sequence length for the calibration data.")
    parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")

    parser.add_argument("--hook_level", type=int, default=0, help="0 is for decoder block and 1 is for nn.linear")
    parser.add_argument("--sparsity", type=float, default=0.3, help="sparsity")

    parser.add_argument("--ppl-eval-batch-size", type=int, default=4, help="Batch size for evaluating the perplexity.")

    parser.add_argument("--type2_engg", type=int, default=0, help="zero means not actual pruning and one means actual pruning")
    args = parser.parse_args()

    if args.model.split('/')[0] == "meta-llama":
        tokenizer = LlamaTokenizer.from_pretrained(args.model,use_auth_token = args.auth_token)
        tokenizer.pad_token = tokenizer.eos_token

        model = LlamaForCausalLM.from_pretrained(
            args.model,
            device_map="auto",              
            torch_dtype=torch.float32,             
            use_auth_token = args.auth_token
        )
    elif args.model.split('/')[0] == "Qwen":
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.float32).to(torch.device("cuda:0"))
        model.config.head_dim = model.config.hidden_size//model.config.num_attention_heads

    dataset = data_utils.get_dataset(args.cal_dataset)
    if args.cal_dataset == "hotpot":
        train_dataset, test_dataset = dataset["train"], dataset["validation"]
    elif args.cal_dataset == "pubmed":
        train_dataset, test_dataset = dataset["train"], dataset["train"]
    else:
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

    for i in train_loader:
        i.pop("labels")
        data = i
        break

    data = {k: v.to("cuda:0") for k, v in data.items()}

    gs = global_sparsity_allocation(model, data, args.sparsity)



