# CUDA_VISIBLE_DEVICES=6 
python3 task_prune_type2.py \
    --model meta-llama/Llama-2-7b-hf\
    --auth_token  \
    --hook_level 1\
    --cal-dataset wikitext2\
    --sparsity 0\
    --type2_engg 1
 
