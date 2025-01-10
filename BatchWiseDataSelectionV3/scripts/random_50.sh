deepspeed --module --hostfile=None --master_port 29501 --include localhost:4,5,6,7 openrlhf.cli.train_ppo \
  --pretrain  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --reward_pretrain yifangong/TinyLlama-1.1B-Chat-v1.0-reward-model \
  --save_path ../../checkpoint/selectionv3/random_50 \
  --ckpt_path  ../../tinyllama-warmup-ckpt  \
  --save_steps 10 \
  --max_ckpt_num 1000 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 4 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 8 \
  --rollout_batch_size 512 \
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
  --use_wandb dca3db2a5790b6d0133b22f305b32fb844c224ef \
  --wandb_project selectionv3_random_50 \
  --wandb_run_name selectionv3_random_50 \
  --select_policy random \
  --select_proportion 0.5 \
  --buffer_path ../buffer \
  --evaluation_data_path ../data/evaluation_data_harmless.jsonl \