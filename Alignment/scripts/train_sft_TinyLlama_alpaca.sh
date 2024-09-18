set -x

read -r -d '' training_commands <<EOF
openrlhf.cli.train_sft \
   --max_len 512 \
   --dataset yifangong/processed_alpaca \
   --input_key input \
   --output_key output \
   --input_template "User: {}\nAssistant: " \
   --train_batch_size 128 \
   --micro_train_batch_size 2 \
   --max_samples 500000 \
   --pretrain TinyLlama/TinyLlama_v1.1 \
   --save_path ../checkpoint/TinyLlama-sft-alpaca \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 0 \
   --max_epochs 3 \
   --bf16 \
   --learning_rate 1e-5 \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb b7f573ca98ce546e2a92a20e0602f5fb456156f2 \
   --wandb_project DataSelectionForAlignment
EOF
    # --use_wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --module $training_commands
fi
