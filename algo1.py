import argparse
import logging

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
import torch
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM

import data_utils
import utils
from gs import   global_sparsity_allocation
from algo1_modular import mode_gpt_algo1

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


if args.cal_dataset not in ["billsum","multilegalpile","pubmedqa"]:
    dataset = data_utils.get_dataset(args.cal_dataset)

if args.cal_dataset == "hotpot":
    train_dataset, test_dataset = dataset["train"], dataset["validation"]
elif args.cal_dataset == "pubmedqa":
    train_dataset, test_dataset = data_utils.get_dataset(args.cal_dataset)
elif args.cal_dataset == "medqa_4options":
    train_dataset, test_dataset = dataset["train"], dataset["validation"]
elif args.cal_dataset == "billsum":
    train_dataset, test_dataset = data_utils.get_dataset(args.cal_dataset)
elif args.cal_dataset == "multilegalpile":
    train_dataset, test_dataset = data_utils.get_dataset(args.cal_dataset)
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



data = next(iter(train_loader))
data.pop("labels")

data = {k: v.to("cuda:0") for k, v in data.items()}

param = sum(int(p.nelement()) for p in model.parameters())
logging.info(f"unpruned model parameter count is :{param}")
if args.cal_dataset not in ["pubmedqa","billsum","medqa_4options","mathqa","sciq"]:
    logging.info("calculating unpruned model perplexity")
    dataset_ppl = utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    logging.info(f"unpruned model perplexity is :{dataset_ppl}")


if args.cal_dataset in ["pubmedqa","medqa_4options","mathqa","sciq"]:
    logging.info(f"calculating unpruned model accuracy for medical question answering")
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[args.cal_dataset],
        batch_size=4
    )
    print(make_table(results))

if args.cal_dataset == "billsum":
    logging.info("calculating unpruned model rouge scores")
    rouge_scores = utils.evaluate_rouge(model,test_dataset,tokenizer)
    logging.info(f"unpruned model rouge scores are :rouge1:{rouge_scores['rouge1']} rouge2:{rouge_scores['rouge2']}  rougeL:{rouge_scores['rougeL']}")

logging.info(f"calculating optimal sparsity allocation")
if args.sparsity_allocation ==1:
    gs = global_sparsity_allocation(model, data, args.sparsity)
    sparsity_layer = gs.gs
    sparsity_layer = [0.8 if i>1 else (i.item()) for i in gs.gs]
else:
    sparsity_layer = []
    for _ in range(model.config.num_hidden_layers):
        sparsity_layer.append(args.sparsity)

logging.info(f"calculating optimal sparsity allocation completed")

# Intatiating Pruner
pruner = mode_gpt_algo1()
    
#Pruning
pruner.prune(model,train_loader, sparsity_layer)


param = sum(int(p.nelement()) for p in model.parameters())
logging.info(f"pruned model parameter count is :{param}")
if args.cal_dataset not in ["pubmedqa","billsum","medqa_4options","mathqa","sciq"]:
    logging.info("calculating pruned model perplexity")
    dataset_ppl = utils.evaluate_ppl(model, model.config.pad_token_id, test_loader)
    logging.info(f"pruned model perplexity is :{dataset_ppl}")


if args.cal_dataset in ["pubmedqa","medqa_4options","mathqa","sciq"]:
    logging.info(f"calculating pruned model accuracy for medical question answering")
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[args.cal_dataset],
        batch_size=4
    )
    print(make_table(results))


if args.cal_dataset == "billsum":
    logging.info("calculating pruned model rouge scores")
    rouge_scores = utils.evaluate_rouge(model,test_dataset,tokenizer)
    logging.info(f"pruned model rouge scores are :rouge1:{rouge_scores['rouge1']} rouge2:{rouge_scores['rouge2']}  rougeL:{rouge_scores['rougeL']}")


