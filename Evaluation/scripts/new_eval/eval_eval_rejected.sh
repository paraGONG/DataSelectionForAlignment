python eval.py \
    --data yifangong/evaluation_data \
    --model yifangong/tinyllama_rejected \
    --batch_size 4 \
    --device "cuda:3" \
    --inference \
    --reward \
    --reward_model yifangong/TinyLlama-1.1B-Chat-v1.0-reward-model \