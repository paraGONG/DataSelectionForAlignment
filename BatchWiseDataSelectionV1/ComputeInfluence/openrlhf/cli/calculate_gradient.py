import argparse
import itertools
import math
import os
from copy import deepcopy
from datetime import datetime

import torch
from transformers.trainer import get_scheduler

from openrlhf.datasets import PromptDataset, SFTDataset
from openrlhf.models import Actor, get_llm_for_sequence_regression
from openrlhf.trainer import GradientCalculator
from openrlhf.utils import blending_datasets, get_strategy, get_tokenizer


def train(args):
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed()

    # configure model
    # load huggingface model
    actor = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.actor_lora_rank,
        lora_alpha=args.actor_lora_alpha,
        target_modules=args.actor_target_modules,
        lora_dropout=args.actor_lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )

    if args.actor_init_on_gpu:
        actor = actor.to(torch.cuda.current_device())

    critic = get_llm_for_sequence_regression(
        args.critic_pretrain,
        "critic",
        normalize_reward=args.normalize_reward,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        lora_rank=args.critic_lora_rank,
        lora_alpha=args.critic_lora_alpha,
        target_modules=args.critic_target_modules,
        lora_dropout=args.critic_lora_dropout,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        value_head_prefix=args.value_head_prefix,
        init_value_head=strategy.args.pretrain == strategy.args.critic_pretrain,
    )

    reward_model = get_llm_for_sequence_regression(
        args.reward_pretrain,
        "reward",
        normalize_reward=args.normalize_reward,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_train_config(is_actor=False),
        value_head_prefix=args.value_head_prefix,
    )
    get_tokenizer(args.reward_pretrain, reward_model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    
    # strategy.print("reward normalization status: {}".format(args.normalize_reward))
    # strategy.print("mean: {}, std {}".format(critic.mean, critic.std))
 
    # configure tokenizer
    tokenizer = get_tokenizer(args.pretrain, actor.model, "left", strategy, use_fast=not args.disable_fast_tokenizer)
    get_tokenizer(args.critic_pretrain, critic, "left", strategy, use_fast=not args.disable_fast_tokenizer)

    # strategy.print(actor)
    # strategy.print(critic)

    # load weights for reference actor
    initial_model = Actor(
        args.pretrain,
        use_flash_attention_2=args.flash_attn,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        ds_config=strategy.get_ds_eval_config(offload=False),
    )
    get_tokenizer(args.pretrain, initial_model.model, "left", strategy)

    # gradient_checkpointing
    if args.gradient_checkpointing:
        actor.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )
        critic.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # configure optimizer
    actor_optim = strategy.create_optimizer(
        actor, lr=args.actor_learning_rate, betas=args.adam_betas, weight_decay=args.l2
    )
    critic_optim = strategy.create_optimizer(
        critic, lr=args.critic_learning_rate, betas=args.adam_betas, weight_decay=args.l2
    )

    # prepare datasets
    prompts_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        return_eval=False,
        train_split=args.prompt_split,
    )

    # get the window
    prompts_data = prompts_data.select(range(args.window_num, args.window_num + args.window_size))

    # prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    prompts_dataset = PromptDataset(prompts_data, tokenizer, strategy, input_template=args.input_template)

    # prepare dataloader
    # For DataSelection, shuffle = False, drop_last = False
    prompts_dataloader = strategy.setup_dataloader(prompts_dataset, args.micro_rollout_batch_size, True, shuffle=False, drop_last=False)

    # configure scheduler
    num_update_steps_per_episodes = len(prompts_dataset) // args.train_batch_size * args.max_epochs
    max_steps = math.ceil(args.num_episodes * num_update_steps_per_episodes)

    actor_scheduler = get_scheduler(
        "constant",
        actor_optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
    )

    critic_scheduler = get_scheduler(
        "constant",
        critic_optim,
        num_warmup_steps=math.ceil(max_steps * 0.03),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.critic_learning_rate * 0.1},
    )

    # prepare models/optimizers...
    (
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        initial_model,
    ) = strategy.prepare(
        (actor, actor_optim, actor_scheduler),
        (critic, critic_optim, critic_scheduler),
        reward_model,
        initial_model,
        is_rlhf=True,
    )

    # load checkpoint
    consumed_samples = 0
    args.ckpt_path = f"../checkpoint/tinyllama_win_{args.window_num-1}_{args.select_policy}_ckpt/"
    print(os.path.exists(args.ckpt_path))
    # if args.load_checkpoint and os.path.exists(os.path.join(args.ckpt_path, "_actor")):
    _, states = strategy.load_ckpt(actor.model, os.path.join(args.ckpt_path, "_actor"), tag=args.ckpt_tag, load_module_only=True)
    strategy.load_ckpt(critic, os.path.join(args.ckpt_path, "_critic"), tag=args.ckpt_tag, load_module_only=True)
        # consumed_samples = states["consumed_samples"]
        # strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    # configure Trainer
    trainer = GradientCalculator(
        strategy,
        actor,
        critic,
        reward_model,
        initial_model,
        actor_optim,
        critic_optim,
        actor_scheduler,
        critic_scheduler,
        max_epochs=args.max_epochs,
        micro_train_batch_size=args.micro_train_batch_size,
        micro_rollout_batch_size=args.micro_rollout_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        tokenizer=tokenizer,
        prompt_max_len=args.prompt_max_len,
        value_clip=args.value_clip,
        eps_clip=args.eps_clip,
        gamma=args.gamma,
        lambd=args.lambd,
        init_kl_coef=args.init_kl_coef,
        kl_target=args.kl_target,
        ptx_coef=args.ptx_coef,
        max_norm=args.max_norm,
        save_path=os.path.join(args.save_path, f"window_{args.window_num}"),
        eval_data_path=args.evaluation_data_path,
        # fro GPT generation
        do_sample=True,
        max_new_tokens=args.generate_max_len,
        max_length=args.max_len,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    trainer.fit(args, prompts_dataloader, consumed_samples, num_update_steps_per_episodes)
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./gradients")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_ppo")
    parser.add_argument("--ckpt_tag", type=str, default="global_step10")
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # PPO
    parser.add_argument("--num_episodes", type=int, default=1) # to delete
    parser.add_argument("--rollout_batch_size", type=int, default=512)  # to delete
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8) # to delete
    parser.add_argument("--max_epochs", type=int, default=1) # to delete
    parser.add_argument("--prompt_max_len", type=int, default=1024, help="Max tokens for each prompt")
    parser.add_argument("--generate_max_len", type=int, default=1024, help="Max tokens to generate in PPO")
    parser.add_argument("--max_len", type=int, default=None, help="deprecated max_len") # to check
    parser.add_argument("--max_samples", type=int, default=1000000) # to check
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--l2", type=float, default=0.0, help="weight decay loss")
    parser.add_argument("--ptx_coef", type=float, default=0.05, help="PPO-ptx loss coef")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--value_clip", type=float, default=0.2, help="PPO value clip range")
    parser.add_argument("--lambd", type=float, default=0.95, help="PPO GAE lambd")
    parser.add_argument("--gamma", type=float, default=1, help="PPO GAE gamma")
    parser.add_argument("--micro_train_batch_size", type=int, default=4, help="batch size per GPU") # to delete
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size") # to delete
    parser.add_argument("--normalize_reward", action="store_true", default=False, help="Enable Reward Normazation")
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--freezing_actor_steps", type=int, default=-1, help="Used for critic initialization") # to delete
    parser.add_argument(
        "--n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation" # to check
    )
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=9e-6) # to delete
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--init_kl_coef", type=float, default=0.01, help="KL penalty in PPO")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # DeepSpeed
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=0, help="DeepSpeed ZeRO stage")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--actor_init_on_gpu", action="store_true", default=False)
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=False)

    # LoRA
    # Actor Lora
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--actor_lora_rank", type=int, default=0)
    parser.add_argument("--actor_lora_alpha", type=int, default=16)
    parser.add_argument("--actor_target_modules", type=str, nargs="*", default=None)
    parser.add_argument("--actor_lora_dropout", type=float, default=0)

    # Critic Lora
    parser.add_argument("--critic_lora_rank", type=int, default=0)
    parser.add_argument("--critic_lora_alpha", type=int, default=16)
    parser.add_argument("--critic_target_modules", type=str, nargs="*", default=None)
    parser.add_argument("--critic_lora_dropout", type=float, default=0)

    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--reward_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--critic_pretrain", type=str, default=None, help="HF model name or path")
    parser.add_argument("--value_head_prefix", type=str, default="value_head")

    # Custom dataset
    parser.add_argument("--prompt_data", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--prompt_data_probs",
        type=str,
        default="1.0",
        help="sampling probs for datasets",
    )
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--pretrain_data", type=str, default=None, help="HF dataset name or path")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default=None)
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )

    # sliding window
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--window_num", type=int, default=0)

    # evaluation data path
    parser.add_argument("--evaluation_data_path", type=str, default=None)
    parser.add_argument("--select_policy", type=str, default=None)

    args = parser.parse_args()

    if args.critic_pretrain is None:
        args.critic_pretrain = args.pretrain

    if args.input_template and not "{}" in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None
    train(args)
