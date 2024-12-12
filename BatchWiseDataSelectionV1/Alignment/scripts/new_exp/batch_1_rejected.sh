deepspeed --module --hostfile=None openrlhf.cli.train_ppo \
  --pretrain  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --reward_pretrain yifangong/TinyLlama-1.1B-Chat-v1.0-reward-model \
  --save_path ../checkpoint/tinyllama \
  --save_steps 1 \
  --max_ckpt_num 1000 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 2 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 8 \
  --rollout_batch_size 128 \
  --max_epochs 1 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 0 \
  --bf16 \
  --actor_learning_rate 5e-4 \
  --critic_learning_rate 5e-4 \
  --init_kl_coef 0.01 \
  --max_samples 100000 \
  --load_checkpoint \
  --actor_lora_rank 8 \
  --actor_lora_alpha 16 \
  --critic_lora_rank 8 \
  --critic_lora_alpha 16 \
  --window_num 1 \
  --select_policy rejected \
  --buffer_path ../buffer