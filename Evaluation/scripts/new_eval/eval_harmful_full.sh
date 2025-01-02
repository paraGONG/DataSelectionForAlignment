python eval.py \
    --data yifangong/harmful_test \
    --model yifangong/tinyllama_full_candidate \
    --batch_size 4 \
    --device "cuda:0" \
    --inference \
    --reward \
    --reward_model yifangong/TinyLlama-1.1B-Chat-v1.0-reward-model \