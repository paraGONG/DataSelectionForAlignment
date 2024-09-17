set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_rm \
   --save_path ./checkpoint/tinyllama-rm \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --train_batch_size 1 \
   --micro_train_batch_size 1 \
   --pretrain /root/siton-object-46b8630eb56e449886cb89943ab6fe10/models/TinyLlama/TinyLlama_v1.1 \
   --max_epochs 1 \
   --max_len 512 \
   --zero_stage 0 \
   --learning_rate 9e-6 \
   --dataset /root/siton-object-46b8630eb56e449886cb89943ab6fe10/dataset/OpenRLHF/preference_dataset_mixture2_and_safe_pku \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --load_checkpoint \
   --lora_rank 8 \
   --lora_alpha 16
EOF
     # --use_wandb [WANDB_TOKENS] or True (use wandb login command)
     # --packing_samples


if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
