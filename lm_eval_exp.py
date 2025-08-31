from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.loggers import EvaluationTracker
from lm_eval.utils import make_table
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import LlamaTokenizer, GenerationConfig, LlamaConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaRMSNorm
import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

auth_token = ""

tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf",use_auth_token = auth_token)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    device_map="auto",              
    torch_dtype=torch.float32,             
    use_auth_token = auth_token
)

lm = HFLM(pretrained=model, tokenizer=tokenizer)
results = evaluator.simple_evaluate(
    model=lm,
    tasks=["humaneval"],   # <-- task name
    batch_size=4,confirm_run_unsafe_code=True
)
print(make_table(results))
   