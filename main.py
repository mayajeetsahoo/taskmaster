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
import torch.nn as nn
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
parser.add_argument("--cal-max-seqlen", type=int, default=1024, help="Maximum sequence length for the calibration data.")
parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")

parser.add_argument("--hook_level", type=int, default=0, help="0 is for decoder block and 1 is for nn.linear")

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


layer_c = [0]
def save_hidden_states_hook(module, input, output):
    layer_c[0] = layer_c[0] + 1
    layer = layer_c[0]
    transpose = torch.transpose(output[0],1,2)
    mul = torch.matmul(transpose,output[0])

    co_var = mul.sum(dim=0)/args.cal_batch_size
    co_var = co_var.detach().cpu()
    co_var = co_var.numpy()
    np.save("files/co_var_"+str(layer)+".npy",co_var)

    mean = output[0].sum(dim=0)/args.cal_batch_size
    mean = mean.detach().cpu()
    mean = mean.numpy()
    np.save("files/mean_"+str(layer)+".npy",mean)

    print(str(layer)+".npy")

if args.hook_level == 0:
    os.makedirs("files",exist_ok = True)
    # Register hook to each decoder block
    for i, block in enumerate(model.model.layers):
        block.register_forward_hook(save_hidden_states_hook)


hook_outputs = {}

def hook_fn(module, input, output):
    name = module._hook_name  # Custom attribute we'll assign below
    hook_outputs[name] = output.detach().cpu()
    print(f"Hook triggered on: {name} | Output shape: {output.shape}")

    transpose = torch.transpose(output,1,2)
    mul = torch.matmul(transpose,output)

    co_var = mul.sum(dim=0)/output.shape[0]
    co_var = co_var.detach().cpu()
    co_var = co_var.numpy()
    np.save("files_atom/co_var_"+str(name)+".npy",co_var)

    mean = output.sum(dim=0)/output.shape[0]
    mean = mean.detach().cpu()
    mean = mean.numpy()
    np.save("files_atom/mean_"+str(name)+".npy",mean)

if args.hook_level ==1:
    os.makedirs("files_atom",exist_ok = True)
    # Register hooks on every nn.Linear layer
    for name, module in model.model.named_modules():
        if isinstance(module, nn.Linear):
            module._hook_name = name  # Add name so we know which layer triggered
            module.register_forward_hook(hook_fn)


model.eval()
with torch.no_grad():
    _ = model(**data)








