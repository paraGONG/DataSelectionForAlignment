set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain OpenRLHF/Llama-2-7b-sft-model-ocra-500k \
   --reward_pretrain OpenRLHF/Llama-2-7b-rm-anthropic_hh-lmsys-oasst-webgpt \
   --save_path ../checkpoint/llama-2-7b-rlhf \
   --ckpt_path  ../ckpt/checkpoints_ppo \
   --save_steps 10 \
   --max_ckpt_num 500 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 1024 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --max_samples 100000 \
   --load_checkpoint \
   --gradient_checkpointing
   --lora_rank 8 \
   --lora_alpha 16 \
   --target_modules q_proj v_proj \
   --use_wandb b7f573ca98ce546e2a92a20e0602f5fb456156f2 \
   --wandb_project OpenRLHF-PPO-without-reward-norm
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --remote_rm_url http://localhost:5000/get_reward

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
