set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_ppo \
   --pretrain /root/siton-object-46b8630eb56e449886cb89943ab6fe10/models/TinyLlama/TinyLlama_v1.1 \
   --reward_pretrain /root/siton-object-46b8630eb56e449886cb89943ab6fe10/DataSelectionForAlignment/checkpoint/tinyllama-rm \
   --save_path /root/siton-tmp/checkpoint/tinyllama-rlhf \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --micro_train_batch_size 1 \
   --train_batch_size 1 \
   --micro_rollout_batch_size 1 \
   --rollout_batch_size 1 \
   --max_epochs 1 \
   --prompt_max_len 512 \
   --generate_max_len 512 \
   --zero_stage 0 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data /root/siton-object-46b8630eb56e449886cb89943ab6fe10/dataset/OpenRLHF/prompt-collection-v0.1 \
   --input_key context_messages \
   --apply_chat_template \
   --max_samples 100000 \
   --normalize_reward \
   --load_checkpoint  \
   --lora_rank 8 \
   --lora_alpha 16
EOF
    # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
    # --remote_rm_url http://localhost:5000/get_reward

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
