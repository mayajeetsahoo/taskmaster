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
os.environ["TOKENIZERS_PARALLELISM"] = "false"


model = "meta-llama/Llama-2-7b-hf"
# model = "codellama/CodeLlama-7b-Instruct-hf"
# model = "WizardLM/WizardCoder-15B-V1.0"

tokenizer = LlamaTokenizer.from_pretrained(model,use_auth_token = auth_token)
tokenizer.pad_token = tokenizer.eos_token

model = LlamaForCausalLM.from_pretrained(
    model,
    device_map="auto",              
    torch_dtype=torch.float32,             
    use_auth_token = auth_token
)

# tokenizer = AutoTokenizer.from_pretrained(model, 
#                                         #   use_auth_token=auth_token
#                                           )
# tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(
#     model,
#     device_map="auto",              
#     torch_dtype="auto",   # Let HF choose bf16/fp16 automatically
#     # use_auth_token=auth_token
# )

lm = HFLM(pretrained=model, tokenizer=tokenizer)
results = evaluator.simple_evaluate(
    model=lm,
    tasks=["medqa_4options"],   # <-- task name
    batch_size=4,confirm_run_unsafe_code=True
)

# gen_kwargs = {
#     "temperature": 0.2,
#     "top_p": 0.95
# }

# results = evaluator.simple_evaluate(
#     model=lm,
#     tasks=["medqa_us"],
#     batch_size=4,
#     num_fewshot=0, gen_kwargs=gen_kwargs,
#     confirm_run_unsafe_code=True,
#     limit=None,              # Run full benchmark (set a number to debug faster)
#     bootstrap_iters=1000     # For stable confidence intervals
# )

print(make_table(results))
   