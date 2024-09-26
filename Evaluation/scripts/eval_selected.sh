python eval.py \
    --data yifangong/saferlhf_evaluation_dataset \
    --model yifangong/tinyllama_selected \
    --batch_size 4 \
    --device "cuda:2" \
    --inference \
    --reward \
    --reward_model yifangong/TinyLlama-1.1B-Chat-v1.0-reward-model \