deepspeed --master_port 29500 --hostfile=None --module openrlhf.cli.calculate_gradient \
  --pretrain  TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --reward_pretrain yifangong/TinyLlama-1.1B-Chat-v1.0-reward-model \
  --save_path ../buffer \
  --micro_train_batch_size 1 \
  --train_batch_size 4 \
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
  --window_size 512 \
  --window_num 3 \
  --evaluation_data_path ../../../evaluation_data.jsonl \
  --select_policy chosen

# openrlhf.cli.calculate_gradient \
#    --pretrain /root/siton-object-46b8630eb56e449886cb89943ab6fe10/models/TinyLlama/TinyLlama_v1.1 \
#    --reward_pretrain /root/siton-object-46b8630eb56e449886cb89943ab6fe10/DataSelectionForAlignment/checkpoint/tinyllama-rm \
#    --save_path ./checkpoint/tinyllama-rlhf \
#    --micro_train_batch_size 1 \
#    --train_batch_size 1 \
#    --micro_rollout_batch_size 1 \
#    --rollout_batch_size 1 \
#    --max_epochs 1 \
#    --prompt_max_len 512 \
#    --generate_max_len 512 \
#    --zero_stage 0 \
#    --actor_learning_rate 5e-7 \
#    --critic_learning_rate 9e-6 \
#    --init_kl_coef 0.01 \
#    --prompt_data /root/siton-object-46b8630eb56e449886cb89943ab6fe10/dataset/OpenRLHF/prompt-collection-v0.1 \
#    --input_key context_messages \
#    --apply_chat_template \
#    --max_samples 100000 \
#    --actor_lora_rank 8 \
#    --actor_lora_alpha 16 \
#    --load_checkpoint