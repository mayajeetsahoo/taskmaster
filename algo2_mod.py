import math
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, repeat_kv
from transformers.models.llama.modeling_llama import LlamaForCausalLM, LlamaAttention, eager_attention_forward, apply_rotary_pos_emb
from transformers import modeling_utils
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from typing import Callable, Optional
import scipy
import data_utils
import utils

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)

sparsity = [0.9]
for i in range(31):
    sparsity.append(0.2)
print(sparsity)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model to load")
parser.add_argument("--auth_token", type = str, help = "authorisation token used to load huggingface models")
parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset to calibrate and calculate perplexity on.",
        choices=["wikitext2", "ptb", "c4", "alpaca","glue"],
        default="wikitext2",
    )
parser.add_argument(
    "--cal-nsamples",
    type=int,
    help="Number of samples of the calibration data to load.",
    default=128,
)

parser.add_argument("--cal-batch-size", type=int, default=32, help="Batch size for loading the calibration data.")
parser.add_argument("--cal-max-seqlen", type=int, default=1024, help="Maximum sequence length for the calibration data.")
parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")

parser.add_argument("--hook_level", type=int, default=0, help="0 is for decoder block and 1 is for nn.linear")
parser.add_argument("--sparsity", type=float, default=0.3, help="sparsity")

parser.add_argument("--ppl-eval-batch-size", type=int, default=4, help="Batch size for evaluating the perplexity.")

parser.add_argument("--type2_engg", type=int, default=1, help="zero means not actual pruning and one means actual pruning for only type 2 algo")
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
# import pdb;pdb.set_trace()
for i in train_loader:
    i.pop("labels")
    data = i
    break

param = sum(int(p.nelement()) for p in model.parameters())
logging.info(f"unpruned model parameter count is :{param}")
logging.info("calculating unpruned model perplexity")
# dataset_ppl = utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
# logging.info(f"unpruned model perplexity is :{dataset_ppl}")

data = {k: v.to("cuda:0") for k, v in data.items()}

storage = {}

def prehook_pos_emb(module,args,kwargs):
    storage["positional_embeddings"] = kwargs.get("position_embeddings")
    # if positional_embeddings is not None:
    #     module._saved_pos_emb = positional_embeddings
    
pre_hook = model.model.layers[0].register_forward_pre_hook(prehook_pos_emb,with_kwargs = True)

class InterruptExecution(Exception):
    pass

def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

def apply_compressed_rope_opt(q: torch.Tensor, k: torch.Tensor, cos_full, sin_full, index_map, index_id):
    """
    q, k: [batch, num_heads, seq_len, head_dim_compressed]
    index_map: List of length num_heads, each a Tensor of shape [compressed_dim]
    cos/sin_full: [seq_len, head_dim_full]
    """
    cos_full = torch.squeeze(cos_full)  # [seq_len, 128]
    sin_full = torch.squeeze(sin_full)

    batch, num_heads, seq_len, head_dim_compressed = q.shape
    head_dim_full = cos_full.shape[-1]
    device = q.device

    # Preallocate full-size q and k with zeros
    q_full = torch.zeros(batch, num_heads, seq_len, head_dim_full, dtype=q.dtype, device=device)
    k_full = torch.zeros_like(q_full)

    for h in range(num_heads):
        idx = index_map[index_id][h].to(device)  # [compressed_dim]
        q_full[:, h, :, idx] = q[:, h, :, :]
        k_full[:, h, :, idx] = k[:, h, :, :]

    # Apply RoPE (vectorized)
    q_rot = q_full * cos_full + rotate_half(q_full) * sin_full
    k_rot = k_full * cos_full + rotate_half(k_full) * sin_full

    return q_rot, k_rot  # [batch, num_heads, seq_len, head_dim_full]

def apply_compressed_rope_opt_qwen(q: torch.Tensor, k: torch.Tensor, cos_full, sin_full, index_map, index_id):
    """
    q, k: [batch, num_heads, seq_len, head_dim_compressed]
    index_map: List of length num_heads, each a Tensor of shape [compressed_dim]
    cos/sin_full: [seq_len, head_dim_full]
    """
    cos_full = torch.squeeze(cos_full)  # [seq_len, 128]
    sin_full = torch.squeeze(sin_full)

    batch, num_heads, seq_len, head_dim_compressed = q.shape
    head_dim_full = cos_full.shape[-1]
    device = q.device

    # Preallocate full-size q and k with zeros
    q_full = torch.zeros(batch, num_heads, seq_len, head_dim_full, dtype=q.dtype, device=device)
    
    batch, num_heads, seq_len, head_dim_compressed = k.shape
    k_full = torch.zeros(batch, num_heads, seq_len, head_dim_full, dtype=q.dtype, device=device)

    for h in range(num_heads):
        start = h*7
        end = start+7
        idx = index_map[index_id][h].to(device)  # [compressed_dim]
        q_full[:, start:end, :, idx] = q[:, start:end, :, :]
        k_full[:, h, :, idx] = k[:, h, :, :]

    # Apply RoPE (vectorized)
    q_rot = q_full * cos_full + rotate_half(q_full) * sin_full
    k_rot = k_full * cos_full + rotate_half(k_full) * sin_full

    return q_rot, k_rot  # [batch, num_heads, seq_len, head_dim_full]

class self_attn_comp_qwen(Qwen2Attention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__(config,layer_idx)
        self.head_dim_qk = math.ceil((1-args.sparsity)*(model.config.hidden_size/model.config.num_attention_heads))
        if args.type2_engg == 0: 
            self.head_dim_qk = self.head_dim  # compressed dimension
        elif args.type2_engg == 1:
            self.head_dim_qk = math.ceil((1-args.sparsity)*(model.config.hidden_size/model.config.num_attention_heads))
            self.head_dim_qk = math.ceil((1-sparsity[layer_idx])*(model.config.hidden_size/model.config.num_attention_heads))
        self.original_head_dim = self.head_dim  # 128
        
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim_qk, bias=True
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim_qk, bias=True
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        qk_shape = (*input_shape, -1, self.head_dim_qk)
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        if args.type2_engg == 1: 
            indexes_q = torch.randn(0)
            for p in range(module.self_attn.num_key_value_groups):
                val = p*model.config.head_dim
                indexes_q = torch.cat([indexes_q,val+indices[self.layer_idx][0]])
            for p in range(7,7+module.self_attn.num_key_value_groups,1):
                val = p*model.config.head_dim
                indexes_q = torch.cat([indexes_q,val+indices[self.layer_idx][1]])
            indexes_k = torch.cat([indices[self.layer_idx][0],64+indices[self.layer_idx][1]]).int()

            
            query_mul = torch.matmul(hidden_states,self.q_proj.weight.data.T)
            query_og_shape = (*query_mul.shape[:-1],self.q_proj.bias.data.shape[0])
            zero_q = torch.zeros(query_og_shape).to(query_mul.device)
            indexes_q = indexes_q.int().to(query_mul.device)
            # import pdb;pdb.set_trace()
            zero_q[:,:,indexes_q] = query_mul
            query_states = (zero_q + self.q_proj.bias.data).view(hidden_shape).transpose(1, 2)
            

            key_mul = torch.matmul(hidden_states,self.k_proj.weight.data.T)
            key_og_shape = (*key_mul.shape[:-1],self.k_proj.bias.data.shape[0])
            zero_k = torch.zeros(key_og_shape).to(key_mul.device)
            indexes_k = indexes_k.int().to(key_mul.device)
            zero_k[:,:,indexes_k] = key_mul
            key_states = (zero_k + self.k_proj.bias.data).view(hidden_shape).transpose(1, 2)
        elif args.type2_engg ==0:
            query_states = self.q_proj(hidden_states).view(qk_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(qk_shape).transpose(1, 2)

        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # if args.type2_engg == 0: 
        #     query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # else:
        #     query_states, key_states = apply_compressed_rope_opt_qwen(query_states, key_states, cos, sin,indices, index_id = self.layer_idx)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = modeling_utils.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            # sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
    
class self_attn_comp(LlamaAttention):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        if args.type2_engg == 0: 
            self.head_dim_qk = self.head_dim  # compressed dimension
        elif args.type2_engg == 1: 
            self.head_dim_qk = math.ceil((1-args.sparsity)*(model.config.hidden_size/model.config.num_attention_heads))
            self.head_dim_qk = math.ceil((1-sparsity[layer_idx])*(model.config.hidden_size/model.config.num_attention_heads))
        self.original_head_dim = self.head_dim  # 128
        
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim_qk, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim_qk, bias=config.attention_bias
        )
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        qk_shape = (*input_shape, -1, self.head_dim_qk)
        hidden_shape = (*input_shape, -1, self.head_dim)

        
        query_states = self.q_proj(hidden_states).view(qk_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(qk_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        if args.type2_engg == 0: 
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        # query_states, key_states = apply_compressed_rope(query_states, key_states, cos, sin,indices, index_id = self.layer_idx)
        else:
            query_states, key_states = apply_compressed_rope_opt(query_states, key_states, cos, sin,indices, index_id = self.layer_idx)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = modeling_utils.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        device = attn_output.device
        self.o_proj.weight.data = self.o_proj.weight.data.to(device)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

class algo2_functor():
    def __init__(self):
        self.storage = {}

    def __call__(self,module,input,output):
        ## storing weight matrices of query and key
    
        cos, sin = storage["positional_embeddings"]
        input_shape = input[0].shape[:-1]
        hidden_shape = (*input_shape, -1, model.config.head_dim)
        # import pdb;pdb.set_trace()
        query_states = module.self_attn.q_proj(input[0]).view(hidden_shape).transpose(1, 2).to(cos.device)
        key_states = module.self_attn.k_proj(input[0]).view(hidden_shape).transpose(1, 2).to(cos.device)
        if args.model.split("/")[0] == "Qwen":
            key_states = torch.repeat_interleave(key_states, dim=1, repeats=module.self_attn.num_key_value_groups)

        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (query_states * cos) + (rotate_half(query_states) * sin)
        k_embed = (key_states * cos) + (rotate_half(key_states) * sin)

        co_var_q = torch.matmul(q_embed.transpose(-2,-1),q_embed).sum(dim=0).detach().cpu()
        co_var_k = torch.matmul(k_embed.transpose(-2,-1),k_embed).sum(dim=0).detach().cpu()
    
        self.storage[int(module._name)]=(co_var_q,co_var_k)

        raise InterruptExecution()

list_layers = list(model.model.layers.named_children())

indices = {}

for layer in range(model.config.num_hidden_layers):
    print(f'compressing query and key matrices of {layer} th decoder layer')
    module = list_layers[layer][1]
    name = list_layers[layer][0] ## string

    layer_functor = algo2_functor()

    module._name = name
    hook = module.register_forward_hook(layer_functor)

    try:
        with torch.no_grad():
            _ = model(**data)
    except InterruptExecution:
        pass

    hook.remove()
    ## pruning
    
    torch.cuda.empty_cache()
    attention_mat_size = model.config.hidden_size

    weight_q_transposed = torch.squeeze(torch.unsqueeze(module.self_attn.q_proj.weight.data.detach().cpu().T,dim=0).view(1,attention_mat_size,-1,model.config.head_dim).transpose(1,2),dim=0)
    weight_k_transposed = torch.squeeze(torch.unsqueeze(module.self_attn.k_proj.weight.data.detach().cpu().T,dim=0).view(1,attention_mat_size,-1,model.config.head_dim).transpose(1,2),dim=0)
    
    k = math.ceil((1-args.sparsity)*(model.config.hidden_size/model.config.num_attention_heads))
    
    k = math.ceil((1-sparsity[int(module._name)])*(model.config.hidden_size/model.config.num_attention_heads))
    # import pdb;pdb.set_trace()
    
    if args.model.split("/")[0] == "meta-llama":
        if args.type2_engg ==1:    
            new_k = torch.empty(model.config.num_attention_heads,model.config.hidden_size,k)
            new_q = torch.empty(model.config.num_attention_heads,model.config.hidden_size,k)
        elif args.type2_engg==0:
            new_k = torch.empty(model.config.num_attention_heads,model.config.hidden_size,int(model.config.head_dim))
            new_q = torch.empty(model.config.num_attention_heads,model.config.hidden_size,int(model.config.head_dim))

        layer_indices= []

        for head in range(model.config.num_attention_heads):
            covar_q = torch.from_numpy(scipy.linalg.sqrtm(layer_functor.storage[layer][0][head,:])) 
            covar_k = torch.from_numpy(scipy.linalg.sqrtm(layer_functor.storage[layer][1][head,:]))
            scores = []
            for col in range(model.config.head_dim):
                score = torch.linalg.norm(covar_q[:,col],ord = 1) * torch.linalg.norm(covar_k[:,col], ord = 1)
                scores.append(score)
            top_k_scores , top_k_indices = torch.topk(torch.tensor(scores),k,largest= True)

            top_k_indices = top_k_indices.sort().values

            layer_indices.append(top_k_indices)
            
            if args.type2_engg == 1: 
                s_k = torch.zeros((torch.tensor(scores).shape[0],k),dtype = torch.float32)
                s_k[top_k_indices,range(k)]=1.0

            elif args.type2_engg == 0:
                s_k = torch.zeros((torch.tensor(scores).shape[0],torch.tensor(scores).shape[0]),dtype = torch.float32)
                s_k[top_k_indices,top_k_indices]=1.0
            
            new_k[head,:,:] = (weight_k_transposed[head,:,:]@s_k)
            new_q[head,:,:] = (weight_q_transposed[head,:,:]@s_k)
        indices[int(module._name)] = layer_indices

        
        # using reshape instead of view because of non contiguous data
        if args.type2_engg == 1: 
            module.self_attn.q_proj.weight.data = torch.squeeze(torch.unsqueeze(new_q,dim=0).transpose(1,2).reshape(1,model.config.hidden_size,k*model.config.num_attention_heads)).T.to(module.self_attn.q_proj.weight.device)
            module.self_attn.k_proj.weight.data = torch.squeeze(torch.unsqueeze(new_k,dim=0).transpose(1,2).reshape(1,model.config.hidden_size,k*model.config.num_attention_heads)).T.to(module.self_attn.k_proj.weight.device)

        elif args.type2_engg == 0:
            # head_dim = model.config.hidden_size/model.config.num_attention_heads
            module.self_attn.q_proj.weight.data = torch.squeeze(torch.unsqueeze(new_q,dim=0).transpose(1,2).reshape(1,model.config.hidden_size,model.config.head_dim*model.config.num_attention_heads)).T.to(module.self_attn.q_proj.weight.device)
            module.self_attn.k_proj.weight.data = torch.squeeze(torch.unsqueeze(new_k,dim=0).transpose(1,2).reshape(1,model.config.hidden_size,model.config.head_dim*model.config.num_attention_heads)).T.to(module.self_attn.k_proj.weight.device)

    elif args.model.split("/")[0] == "Qwen":

        if args.type2_engg ==1:    
            new_k = torch.empty(model.config.num_key_value_heads,model.config.hidden_size,k)
            new_q = torch.empty(model.config.num_attention_heads,model.config.hidden_size,k)
        elif args.type2_engg==0:
            new_k = torch.empty(model.config.num_key_value_heads,model.config.hidden_size,int(model.config.head_dim))
            new_q = torch.empty(model.config.num_attention_heads,model.config.hidden_size,int(model.config.head_dim))

        # new_k = torch.empty(model.config.num_key_value_heads,model.config.hidden_size,int(model.config.head_dim))
        # new_q = torch.empty(model.config.num_attention_heads,model.config.hidden_size,int(model.config.head_dim))
        # import pdb;pdb.set_trace()
        scores_heads = {}
        for head in range(model.config.num_attention_heads):
            covar_q = torch.from_numpy(scipy.linalg.sqrtm(layer_functor.storage[layer][0][head,:])) 
            covar_k = torch.from_numpy(scipy.linalg.sqrtm(layer_functor.storage[layer][1][head,:]))
            scores = []
            for col in range(model.config.head_dim):
                score = (torch.linalg.norm(covar_q[:,col],ord = 1)**2) * (torch.linalg.norm(covar_k[:,col], ord = 1)**2)
                scores.append(score.item())
            scores_heads[head] = scores
        
        score_grp = {}
        grp_size = module.self_attn.num_key_value_groups
        for group in range(module.self_attn.config.num_key_value_heads):
            start = group * grp_size
            end = start + grp_size

            group_list = [scores_heads[k] for k in range(start, end)]
            group_sum = [sum(x) for x in zip(*group_list)]
            square_roots = [math.sqrt(x) for x in group_sum]
            score_grp[group] = square_roots
        
        layer_indices= []
        for group in range(module.self_attn.config.num_key_value_heads): 
            start = group * grp_size
            end = start + grp_size

            top_k_scores , top_k_indices = torch.topk(torch.tensor(score_grp[group]),k,largest= True)

            top_k_indices = top_k_indices.sort().values
            layer_indices.append(top_k_indices)
            # s_k = torch.zeros((torch.tensor(scores).shape[0],torch.tensor(scores).shape[0]),dtype = torch.float32)
            # s_k[top_k_indices,top_k_indices]=1.0

            if args.type2_engg == 1: 
                s_k = torch.zeros((torch.tensor(scores).shape[0],k),dtype = torch.float32)
                s_k[top_k_indices,range(k)]=1.0

            elif args.type2_engg == 0:
                s_k = torch.zeros((torch.tensor(scores).shape[0],torch.tensor(scores).shape[0]),dtype = torch.float32)
                s_k[top_k_indices,top_k_indices]=1.0

            new_k[group,:,:] = (weight_k_transposed[group,:,:]@s_k)
            new_q[start:end,:,:] = (weight_q_transposed[start:end,:,:]@s_k)


        indices[int(module._name)] = layer_indices
        
        # module.self_attn.q_proj.weight.data = torch.squeeze(torch.unsqueeze(new_q,dim=0).transpose(1,2).reshape(1,model.config.hidden_size,model.config.head_dim*model.config.num_attention_heads)).T.to(module.self_attn.q_proj.weight.device)
        # module.self_attn.k_proj.weight.data = torch.squeeze(torch.unsqueeze(new_k,dim=0).transpose(1,2).reshape(1,model.config.hidden_size,model.config.head_dim*model.config.num_key_value_heads)).T.to(module.self_attn.k_proj.weight.device)
        
        if args.type2_engg == 1: 
            module.self_attn.q_proj.weight.data = torch.squeeze(torch.unsqueeze(new_q,dim=0).transpose(1,2).reshape(1,model.config.hidden_size,k*model.config.num_attention_heads)).T.to(module.self_attn.q_proj.weight.device)
            module.self_attn.k_proj.weight.data = torch.squeeze(torch.unsqueeze(new_k,dim=0).transpose(1,2).reshape(1,model.config.hidden_size,k*model.config.num_key_value_heads)).T.to(module.self_attn.k_proj.weight.device)
            
            # indexes = torch.randn(0)
            # for p in range(module.self_attn.num_key_value_groups):
            #     val = p*model.config.head_dim
            #     indexes = torch.cat([indexes,val+layer_indices[0]])
            # for p in range(module.self_attn.num_key_value_groups):
            #     val = p*model.config.head_dim
            #     indexes = torch.cat([indexes,val+layer_indices[1]])
            # module.self_attn.q_proj.bias.data = module.self_attn.q_proj.bias.data[indexes.int()]
            # module.self_attn.k_proj.bias.data = module.self_attn.k_proj.bias.data[torch.cat([layer_indices[0],64+layer_indices[1]]).int()]

        elif args.type2_engg == 0:
            # head_dim = model.config.hidden_size/model.config.num_attention_heads
            module.self_attn.q_proj.weight.data = torch.squeeze(torch.unsqueeze(new_q,dim=0).transpose(1,2).reshape(1,model.config.hidden_size,model.config.head_dim*model.config.num_attention_heads)).T.to(module.self_attn.q_proj.weight.device)
            module.self_attn.k_proj.weight.data = torch.squeeze(torch.unsqueeze(new_k,dim=0).transpose(1,2).reshape(1,model.config.hidden_size,model.config.head_dim*model.config.num_key_value_heads)).T.to(module.self_attn.k_proj.weight.device)


    if args.model.split("/")[0] == "Qwen":
        v_bias = module.self_attn.v_proj.bias.data
        q_bias = module.self_attn.q_proj.bias.data
        k_bias = module.self_attn.k_proj.bias.data

    device = module.self_attn.q_proj.weight.device
    q = module.self_attn.q_proj.weight.data
    k = module.self_attn.k_proj.weight.data
    v = module.self_attn.v_proj.weight.data
    o = module.self_attn.o_proj.weight.data
    if args.model.split("/")[0] == "Qwen":
        module.self_attn = self_attn_comp_qwen(model.config,int(name))
    elif args.model.split("/")[0]== "meta-llama":
        module.self_attn = self_attn_comp(model.config,int(name))
    module.self_attn.q_proj.weight.data = q.to(device)
    module.self_attn.k_proj.weight.data = k.to(device)
    module.self_attn.v_proj.weight.data = v.to(device)
    module.self_attn.o_proj.weight.data = o.to(device)

    if args.model.split("/")[0] == "Qwen":
        module.self_attn.v_proj.bias.data = v_bias.to(device)
        module.self_attn.k_proj.bias.data = k_bias.to(device)
        module.self_attn.q_proj.bias.data = q_bias.to(device)

    torch.cuda.empty_cache()
    

param = sum(int(p.nelement()) for p in model.parameters())
logging.info(f"pruned model parameter count is :{param}")
logging.info("calculating pruned model perplexity")
dataset_ppl = utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
logging.info(f"pruned model perplexity is :{dataset_ppl}")