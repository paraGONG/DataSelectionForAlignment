deepspeed --module --hostfile=None openrlhf.cli.train_ppo \
  --pretrain  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --reward_pretrain yifangong/TinyLlama-1.1B-Chat-v1.0-reward-model \
  --save_path ./test \
  --ckpt_path  ../../tinyllama-warmup-ckpt  \
  --save_steps 10 \
  --max_ckpt_num 1000 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 4 \
  --train_batch_size 16 \
  --micro_rollout_batch_size 8 \
  --rollout_batch_size 64 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 0 \
  --bf16 \
  --actor_learning_rate 5e-4 \
  --critic_learning_rate 5e-4 \
  --init_kl_coef 0.01 \
  --prompt_data yifangong/candidate_dataset \
  --input_key prompt \
  --input_template "<|user|>\n{}</s>\n<|assistant|>\n" \
  --max_samples 100000 \
  --load_checkpoint \
  --actor_lora_rank 8 \
  --actor_lora_alpha 16 \
  --critic_lora_rank 8 \
  --critic_lora_alpha 16 \
  --flash_attn \
  --use_wandb edfbb872a90db03dcc70a6c53157b55525cbc8d0 \
  --wandb_project tt \
  --wandb_run_name tt \
  --select_policy chosen \
  --select_proportion 1 \
  --buffer_path ../buffer \
  --evaluation_data_path ../data/evaluation_data_harmless.jsonl \