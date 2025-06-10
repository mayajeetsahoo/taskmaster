import os
import gc
import sys
import time
import json
import copy
import random
import argparse
from typing import Tuple

import torch
import numpy as np
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm

import data_utils

device = "cuda:7"

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model to load")
parser.add_argument("--auth_token", type = str, help = "authorisation token used to load huggingface models")
parser.add_argument(
        "--cal-dataset",
        type=str,
        help="Dataset to calibrate and calculate perplexity on.",
        choices=["wikitext2", "ptb", "c4", "alpaca"],
        default="wikitext2",
    )
parser.add_argument(
    "--cal-nsamples",
    type=int,
    help="Number of samples of the calibration data to load.",
    default=128,
)

parser.add_argument("--cal-batch-size", type=int, default=16, help="Batch size for loading the calibration data.")
parser.add_argument(
    "--cal-max-seqlen", type=int, default=1024, help="Maximum sequence length for the calibration data."
)
parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")

args = parser.parse_args()

tokenizer = LlamaTokenizer.from_pretrained(args.model,use_auth_token = args.auth_token)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    args.model,
    device_map="auto",              
    torch_dtype="auto",             
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
for i in train_loader:
    i.pop("labels")
    data = i
    break

# hook_outputs = {}

layer_c = [0]
def save_hidden_states_hook(module, input, output):
    layer_c[0] = layer_c[0] + 1
    layer = layer_c[0]
    transpose = torch.transpose(output[0],1,2)
    mul = torch.matmul(transpose,output[0])

    co_var = mul.sum(dim=0)/args.cal_batch_size
    co_var = co_var.detach().cpu()
    co_var = co_var.numpy()
    np.save("co_var_"+str(layer)+".npy",co_var)

    mean = output[0].sum(dim=0)/args.cal_batch_size
    mean = mean.detach().cpu()
    mean = mean.numpy()
    np.save("mean_"+str(layer)+".npy",mean)

    print(str(layer)+".npy")


# Register hook to each decoder block
for i, block in enumerate(model.model.layers):
    block.register_forward_hook(save_hidden_states_hook)

model.eval()
with torch.no_grad():
    _ = model(**data)


import pdb;pdb.set_trace()

# cal_sigma = []
# for i in list(hook_outputs.keys()):
#     layer_output = hook_outputs[i]
#     cal_data = torch.zeros(4096,4096)
#     for j in range(16):
#         mat = layer_output[j]
#         transpose_mat = torch.transpose(mat,0,1)
#         product = torch.matmul(transpose_mat,mat)
#         cal_data = cal_data + product
#     cal_data = cal_data/16
#     cal_sigma.append(cal_data)

# cal_sigma = []
# for i in list(hook_outputs.keys()):
#     layer_output = hook_outputs[i].to(device)
#     transpose = torch.transpose(layer_output,1,2)
#     mul = torch.matmul(transpose,layer_output)
#     final = mul.sum(dim=0)/args.cal_batch_size
#     cal_sigma.append(final)



