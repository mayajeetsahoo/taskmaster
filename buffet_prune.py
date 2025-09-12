import argparse
import logging

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
import torch
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from datasets import concatenate_datasets
from accelerate import Accelerator
from accelerate import dispatch_model
from accelerate.utils import infer_auto_device_map

import data_utils
import utils
from gs import   global_sparsity_allocation
from algo1_modular import mode_gpt_algo1
import gc

# accelerator = Accelerator()

parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "model to load")
parser.add_argument("--auth_token", type = str, help = "authorisation token used to load huggingface models")
parser.add_argument("--cal-nsamples",type=int,help="Number of samples of the calibration data to load.",default=128)
parser.add_argument("--cal-batch-size", type=int, default=8, help="Batch size for loading the calibration data.")
parser.add_argument("--cal-max-seqlen", type=int, default=1024, help="Maximum sequence length for the calibration data.")
parser.add_argument("--varied-seqlen", action="store_true", help="Varied sequence lengths in the calibration data.")
parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
parser.add_argument("--sparsity", type=float, default=0.3, help="sparsity")
parser.add_argument("--sparsity_allocation", type=int, default=1, help="1 means optimal allocation by measuring cosine similarity and 0 is uniform")
args = parser.parse_args()



## Legal
train_dataset_mlp, test_dataset_mlp = data_utils.get_dataset("multilegalpile")
train_dataset_mlp = train_dataset_mlp.shuffle(seed=42).select(range(5000))

train_dataset_bill, test_dataset_bill = data_utils.get_dataset("billsum")
train_dataset_bill = train_dataset_bill.shuffle(seed=42).select(range(5000))

train_dataset_legal = concatenate_datasets([train_dataset_mlp, train_dataset_bill])
test_dataset_legal = concatenate_datasets([test_dataset_mlp, test_dataset_bill]).shuffle(seed=42)



## Medical
train_dataset_pmed, test_dataset_pmed = data_utils.get_dataset("pubmedqa")
train_dataset_pmed = train_dataset_pmed.shuffle(seed=42).select(range(5000))

data_medqa = data_utils.get_dataset("medqa_4options")
train_dataset_medqa, test_dataset_medqa =  data_medqa["train"], data_medqa["validation"]
train_dataset_medqa = train_dataset_medqa.shuffle(seed=42).select(range(5000))

train_dataset_med = concatenate_datasets([train_dataset_pmed, train_dataset_medqa]).shuffle(seed=42)
test_dataset_med = concatenate_datasets([test_dataset_pmed, test_dataset_medqa]).shuffle(seed=42)

import pdb;pdb.set_trace()

## Math
math_dataset = data_utils.get_dataset("mathqa")
train_dataset_math, test_dataset_math = math_dataset["train"].shuffle(seed=42).select(range(5000)), math_dataset["test"].shuffle(seed=42).select(range(300))
## science
sci_dataset = data_utils.get_dataset("sciq")
train_dataset_sci, test_dataset_sci = sci_dataset["train"].shuffle(seed=42).select(range(5000)), sci_dataset["test"].shuffle(seed=42).select(range(300))
## General
dataset = data_utils.get_dataset("wikitext2")
train_dataset_gen, test_dataset_gen = dataset["train"].shuffle(seed=42).select(range(5000)), dataset["test"].shuffle(seed=42).select(range(300))










train_dataset = [train_dataset_legal, train_dataset_med, train_dataset_math, train_dataset_sci, train_dataset_gen]


for dataset in train_dataset:

    model = LlamaForCausalLM.from_pretrained(
    args.model,
    device_map="auto",              
    dtype=torch.float32,             
    token = args.auth_token)

    tokenizer = LlamaTokenizer.from_pretrained(args.model,token = args.auth_token)
    tokenizer.pad_token = tokenizer.eos_token 

    train_loader = data_utils.prepare_dataloader(
    dataset=dataset,
    tokenizer=tokenizer,
    max_seqlen=args.cal_max_seqlen,
    batch_size=args.cal_batch_size,
    nsamples=args.cal_nsamples,
    varied_seqlen=args.varied_seqlen,
    seed=args.seed) 

    ## calculating sparsity allocation
    data = next(iter(train_loader))
    data.pop("labels")

    data = {k: v.to("cuda:0") for k, v in data.items()}
   

    if args.sparsity_allocation ==1:
        gs = global_sparsity_allocation(model, data, args.sparsity)
        sparsity_layer = gs.gs
        sparsity_layer = [0.8 if i>1 else (i.item()) for i in gs.gs]
    else:
        sparsity_layer = []
        for _ in range(model.config.num_hidden_layers):
            sparsity_layer.append(args.sparsity)


    # Intatiating Pruner
    pruner = mode_gpt_algo1()

    ########
    pruner.prune(model, train_loader, sparsity_layer)
    ########
    import pdb;pdb.set_trace()
    del model
    gc.collect()
    torch.cuda.empty_cache()
