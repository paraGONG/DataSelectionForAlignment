set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ../checkpoint/TinyLlamaChat_rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --pretrain TinyLlama/TinyLlama_v1.1 \
   --max_epochs 1 \
   --max_len 4096 \
   --zero_stage 0 \
   --learning_rate 9e-6 \
   --dataset OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --load_checkpoint \
   --gradient_checkpointing \
   --lora_rank 8 \
   --lora_alpha 16 \
   --target_modules q_proj v_proj \
   --use_wandb b7f573ca98ce546e2a92a20e0602f5fb456156f2 \
   --wandb_project DataSelectionForAlignment \

EOF
#   --lora_rank 8 \
#   --lora_alpha 16 \
#   --target_modules q_proj v_proj \
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
