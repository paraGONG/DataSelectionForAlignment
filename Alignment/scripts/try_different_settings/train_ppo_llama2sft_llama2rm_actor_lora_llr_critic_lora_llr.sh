deepspeed --module openrlhf.cli.train_ppo \
  --pretrain  OpenRLHF/Llama-2-7b-sft-model-ocra-500k \
  --reward_pretrain OpenRLHF/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt \
  --save_path ../checkpoint/llama2sft-llama2rm-rlhf-actor-lora-llr-critic-lora-llr \
  --ckpt_path  ../ckpt/llama2sft-llama2rm-rlhf-actor-lora-llr-critic-lora-llr \
  --save_steps 10 \
  --max_ckpt_num 500 \
  --logging_steps 1 \
  --eval_steps -1 \
  --micro_train_batch_size 2 \
  --train_batch_size 128 \
  --micro_rollout_batch_size 4 \
  --rollout_batch_size 512 \
  --max_epochs 2 \
  --prompt_max_len 1024 \
  --generate_max_len 1024 \
  --zero_stage 2 \
  --bf16 \
  --actor_learning_rate 5e-4 \
  --critic_learning_rate 5e-4 \
  --init_kl_coef 0.01 \
  --prompt_data yifangong/rlhf-prompt-collection-v1.0 \
  --input_key prompt \
  --input_template "User: {}\nAssistant: " \
  --max_samples 100000 \
  --load_checkpoint \
  --actor_lora_rank 8 \
  --actor_lora_alpha 16 \
  --critic_lora_rank 8 \
  --critic_lora_alpha 16 \
  --flash_attn \
  --gradient_checkpointing \
  --use_wandb b7f573ca98ce546e2a92a20e0602f5fb456156f2 \
  --wandb_project try_llama2sft-llama2rm-rlhf-actor-lora-llr-critic-lora-llr \
  --wandb_run_name try_llama2sft-llama2rm-rlhf-actor-lora-llr-critic-lora-llr