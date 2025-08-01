import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple
import logging
import torch
import torch.nn as nn
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm

import data_utils
from utils import cleanup_memory
import utils

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)


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

parser.add_argument("--cal-batch-size", type=int, default=16, help="Batch size for loading the calibration data.")
parser.add_argument("--cal-max-seqlen", type=int, default=1024, help="Maximum sequence length for the calibration data.")
parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")

parser.add_argument("--hook_level", type=int, default=0, help="0 is for decoder block and 1 is for nn.linear")
parser.add_argument("--sparsity", type=float, default=0.3, help="sparsity")

parser.add_argument("--ppl-eval-batch-size", type=int, default=8, help="Batch size for evaluating the perplexity.")

parser.add_argument("--type2_engg", type=int, default=0, help="zero means not actual pruning and one means actual pruning")
args = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained(args.model,use_auth_token = args.auth_token)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    args.model,
    device_map="auto",              
    torch_dtype=torch.float32,             
    use_auth_token = args.auth_token
)

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

for i in train_loader:
    i.pop("labels")
    data = i
    break

data = {k: v.to("cuda:0") for k, v in data.items()}

param = sum(int(p.nelement()) for p in model.parameters())
logging.info(f"unpruned model parameter count is :{param}")
logging.info("calculating unpruned model perplexity")
dataset_ppl = utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
logging.info(f"unpruned model perplexity is :{dataset_ppl}")


class InterruptExecution(Exception):
    pass

class algo1_functor():
    def __init__(self):
        self.co_var = None
    
    def __call__(self,module,input,output):
        gate = module.gate_proj(input[0])
        up = module.up_proj(input[0])
        hidden = torch.nn.functional.silu(gate) * up
        transpose = torch.transpose(hidden,1,2)
        mul = torch.matmul(transpose,hidden)

        co_var = mul.sum(dim=0)/hidden.shape[0]
        co_var = co_var.to(torch.float32).detach().cpu()
        self.co_var = co_var

        raise InterruptExecution()
    

list_layers = list(model.model.layers.named_children())

for layer in range(model.config.num_hidden_layers):
    print(f'compressing mlp layer of {layer} th decoder layer')

    module = list_layers[layer][1].mlp
    name = list_layers[layer][0]

    module._name = name
    functor = algo1_functor()
    hook = module.register_forward_hook(functor)

    with torch.no_grad():
        try:
            _ = model(**data)
        except InterruptExecution:
            pass

    hook.remove()
    ## pruning

    torch.cuda.empty_cache()
    co_var = functor.co_var

    I = torch.eye(co_var.size(0), device=co_var.device, dtype=co_var.dtype)
    A = (co_var+I)
    score_mat = torch.matmul(co_var,torch.inverse(A))
    scores_vector = torch.diagonal(score_mat)

    topk_scores, topk_indices = torch.topk(scores_vector, int((1-args.sparsity)*co_var.size(0)), largest=True)
    topk_indices = topk_indices.sort().values
    d_int = scores_vector.shape[0]
    S_k = torch.zeros((d_int, int((1-args.sparsity)*co_var.size(0))), dtype=torch.float32)
    S_k[topk_indices, torch.arange(int((1-args.sparsity)*co_var.size(0)))] = 1.0
    W_U_k = torch.concat([module.up_proj.weight,module.gate_proj.weight],dim=1).T.to(torch.float32) @ S_k.to(module.gate_proj.weight.device) 

    Mat = (S_k.T @ co_var @ S_k).to(torch.float64)
    W_D_k = torch.linalg.pinv(Mat).to(torch.float32) @ (S_k.T @ co_var @ module.down_proj.weight.data.T.cpu())
    
    new_up_proj = W_U_k.T[:,:model.config.hidden_size]
    new_gate_proj = W_U_k.T[:,model.config.hidden_size:]
    module.up_proj.weight.data = new_up_proj.to(module.up_proj.weight.device)
    module.gate_proj.weight.data = new_gate_proj.to(module.gate_proj.weight.device)

    module.down_proj.weight.data = W_D_k.T.to(module.down_proj.weight.device)


param = sum(int(p.nelement()) for p in model.parameters())
logging.info(f"pruned model parameter count is :{param}")
logging.info("calculating pruned model perplexity")
dataset_ppl = utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
logging.info(f"pruned model perplexity is :{dataset_ppl}")