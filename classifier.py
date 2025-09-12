import torch
from typing import Optional
import torch.nn as nn
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.cache_utils import Cache
from transformers.masking_utils import create_causal_mask
from classifier_utils import classifier, classifier_training
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
    device_map="auto",              
    torch_dtype=torch.float32,             
    use_auth_token = args.auth_token
)


## Legal
train_dataset_legal, test_dataset_legal = data_utils.get_dataset("multilegalpile")
## Medical
train_dataset_med, test_dataset_med = data_utils.get_dataset("pubmedqa")
## Math
math_dataset = data_utils.get_dataset("mathqa")
train_dataset_math, test_dataset_math = math_dataset["train"].select(range(5000)), math_dataset["test"].select(range(300))
## science
sci_dataset = data_utils.get_dataset("sciq")
train_dataset_sci, test_dataset_sci = sci_dataset["train"].select(range(5000)), sci_dataset["test"].select(range(300))
## General
dataset = data_utils.get_dataset("wikitext2")
train_dataset_gen, test_dataset_gen = dataset["train"].select(range(5000)), dataset["test"].select(range(300))

# train_dataset_bill, test_dataset_bill = data_utils.get_dataset("billsum")
# data_medqa = data_utils.get_dataset("medqa_4options")
# train_dataset_medqa, test_dataset_medqa =  data_medqa["train"], data_medqa["validation"]


# ADDING LABELS

# training dataset
train_dataset_legal = train_dataset_legal.add_column("label", [0] * len(train_dataset_legal))
train_dataset_med   = train_dataset_med.add_column("label", [1] * len(train_dataset_med))
train_dataset_math = train_dataset_math.add_column("label", [2] * len(train_dataset_math))
train_dataset_sci = train_dataset_sci.add_column("label", [3] * len(train_dataset_sci))
train_dataset_gen = train_dataset_gen.add_column("label", [4] * len(train_dataset_gen))
# test dataset
test_dataset_legal = test_dataset_legal.add_column("label", [0] * len(test_dataset_legal))
test_dataset_med   = test_dataset_med.add_column("label", [1] * len(test_dataset_med))
test_dataset_math = test_dataset_math.add_column("label", [2] * len(test_dataset_math))
test_dataset_sci = test_dataset_sci.add_column("label", [3] * len(test_dataset_sci))
test_dataset_gen = test_dataset_gen.add_column("label", [4] * len(test_dataset_gen))

# test_dataset_bill = test_dataset_bill.add_column("label", [0] * len(test_dataset_bill))
# test_dataset_medqa   = test_dataset_medqa.add_column("label", [1] * len(test_dataset_medqa))

# Merge and shuffle
train_dataset = concatenate_datasets([train_dataset_legal, train_dataset_med, train_dataset_math, train_dataset_sci, train_dataset_gen]).shuffle(seed=42)

test_dataset = concatenate_datasets([test_dataset_legal, test_dataset_med, test_dataset_math, test_dataset_sci, test_dataset_gen]).shuffle(seed=42)

# test_dataset2 = concatenate_datasets([test_dataset_bill, test_dataset_medqa]).shuffle(seed=42)

def tokenize_function(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

tokenized_dataset_train = train_dataset.map(tokenize_function, batched=True)

tokenized_dataset_test = test_dataset.map(tokenize_function, batched=True)

# tokenized_dataset_test2 = test_dataset2.map(tokenize_function, batched=True)


def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_loader = DataLoader(tokenized_dataset_train, batch_size=32, shuffle=True, collate_fn=collate_fn)

test_loader = DataLoader(tokenized_dataset_test, batch_size=32, shuffle=True, collate_fn=collate_fn)

# test_loader2 = DataLoader(tokenized_dataset_test2, batch_size=32, shuffle=True, collate_fn=collate_fn)


classi = classifier(model, num_layers=5, num_labels=5)
train = classifier_training(classi)
train.training(train_loader,test_loader)

















def cosine_sim(a,b):
    # a_mean = a.mean(dim=0)
    # b_mean = b.mean(dim=0)
    # return torch.nn.functional.cosine_similarity(a_mean,b_mean,dim = 0)
    dot = (a.detach().cpu()*b.detach().cpu()).sum(dim=-1)
    xin_norm = a.detach().cpu().norm(dim=-1)
    xout_norm = b.detach().cpu().norm(dim=-1)
    cos_sim = dot / (xin_norm * xout_norm)
    mean_cos_sim = cos_sim.mean()
    return mean_cos_sim



# train_dataset_legal = train_dataset_legal.add_column("label", [0] * len(train_dataset_legal))
# train_dataset_med = train_dataset_med.add_column("label",[1]*len(train_dataset_med))
# train_dataset_bill = train_dataset_bill.add_column("label",[0]*len(train_dataset_bill))

# legal_tok = train_dataset_legal.map(tokenize_function, batched=True)
# med_tok = train_dataset_med.map(tokenize_function, batched=True)
# legal_tok2 = train_dataset_bill.map(tokenize_function, batched=True)

# loader_legal1 = DataLoader(legal_tok, batch_size=32, shuffle=True, collate_fn=collate_fn)
# loader_legal2 = DataLoader(legal_tok2, batch_size=32, shuffle=True, collate_fn=collate_fn)
# loader_med = DataLoader(med_tok, batch_size=32, shuffle=True, collate_fn=collate_fn)


# batch_legal1 = next(iter(loader_legal1))
# batch_legal2 = next(iter(loader_legal2))
# batch_med = next(iter(loader_med))


# hl1_hl2 = []
# hl1_hm = []
# hl2_hm = []
# for _ in tqdm(range(1,33,1)):
#     classi = classifier(model,num_layers=_,num_labels=2)
#     device = next(classi.parameters()).device
#     input_ids_l1 = batch_legal1["input_ids"].to(device)
#     am_l1 = batch_legal1["attention_mask"].to(device)
#     input_ids_l2 = batch_legal2["input_ids"].to(device)
#     am_l2 = batch_legal2["attention_mask"].to(device)

#     input_ids_med = batch_med["input_ids"].to(device)
#     am_med = batch_med["attention_mask"].to(device)



#     with torch.no_grad():
#         ol1,hl1 = classi(input_ids_l1,am_l1)
#         ol2,hl2 = classi(input_ids_l2,am_l2)
#         om,hm = classi(input_ids_med,am_med)

#     hl1_hl2.append(cosine_sim(hl1,hl2).item())
#     hl1_hm.append(cosine_sim(hl1,hm).item())
#     hl2_hm.append(cosine_sim(hl2,hm).item())
# import pdb;pdb.set_trace()





## compute mean of representations and dot product of data sets




