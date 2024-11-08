python eval.py \
    --data yifangong/saferlhf_evaluation_dataset \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --batch_size 4 \
    --device "cuda:3" \
    --inference \
    --reward \
    --reward_model yifangong/TinyLlama-1.1B-Chat-v1.0-reward-model \