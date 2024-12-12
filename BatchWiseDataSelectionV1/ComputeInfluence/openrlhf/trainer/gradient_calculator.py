import math
import json
import os.path
from abc import ABC
from typing import Any, Callable, Dict, List, Optional, Union

import ray
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from openrlhf.models import Actor, GPTLMLoss, PolicyLoss, ValueLoss
from openrlhf.models.utils import masked_mean
from openrlhf.utils.distributed_sampler import DistributedSampler

from .ppo_utils import AdaptiveKLController, Experience, FixedKLController, NaiveExperienceMaker, NaiveReplayBuffer


class GradientCalculator(ABC):
    """
        Trainer for PPO algorithm.

    Args:
        strategy (Strategy): the strategy to use for training
        actor (Actor): the actor model in ppo algorithm
        critic (nn.Module): the critic model in ppo algorithm
        reward_model (nn.Module): the reward model in rlhf algorithm to make reward of sentences
        initial_model (Actor): the initial model in rlhf algorithm to generate reference logits to limit the update of actor
        actor_optim (Optimizer): the optimizer to use for actor model
        critic_optim (Optimizer): the optimizer to use for critic model
        kl_coef (float, defaults to 0.1): the coefficient of kl divergence loss
        train_batch_size (int, defaults to 8): the batch size to use for training
        buffer_limit (int, defaults to 0): the max_size limitaiton of replay buffer
        buffer_cpu_offload (bool, defaults to True): whether to offload replay buffer to cpu
        eps_clip (float, defaults to 0.2): the clip coefficient of policy loss
        value_clip (float, defaults to 0.4): the clip coefficient of value loss
        experience_batch_size (int, defaults to 8): the batch size to use for experience generation
        max_epochs (int, defaults to 1): the number of epochs of training process
        tokenier (Callable, optional): the tokenizer to use for tokenizing the input
        sample_replay_buffer (bool, defaults to False): whether to sample from replay buffer
        dataloader_pin_memory (bool, defaults to True): whether to pin memory for data loader
        callbacks (List[Callback], defaults to []): the callbacks to call during training process
        generate_kwargs (dict, optional): the kwargs to use while model generating
        remote_rm_url (str, optional): function for reward model api
    """

    def __init__(
        self,
        strategy,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        actor_optim: Optimizer,
        critic_optim: Optimizer,
        actor_scheduler,
        critic_scheduler,
        init_kl_coef: float = 0.001,
        kl_target: float = None,
        kl_horizon: int = 10000,
        ptx_coef: float = 0,
        micro_train_batch_size: int = 8,
        buffer_limit: int = 0,
        buffer_cpu_offload: bool = True,
        eps_clip: float = 0.2,
        value_clip: float = 0.2,
        micro_rollout_batch_size: int = 8,
        gradient_checkpointing: bool = False,
        max_epochs: int = 1,
        max_norm: float = 1.0,
        save_path: str = None,
        eval_data_path: str = None,
        tokenizer: Optional[Callable[[Any], dict]] = None,
        prompt_max_len: int = 128,
        dataloader_pin_memory: bool = True,
        remote_rm_url: str = None,
        reward_fn: Callable[[List[torch.Tensor]], torch.Tensor] = None,
        **generate_kwargs,
    ) -> None:
        assert (
            not isinstance(reward_model, List) or len(reward_model) == 1 or reward_fn is not None
        ), "reward_fn must be specified if using multiple reward models"

        super().__init__()
        self.strategy = strategy
        self.args = strategy.args
        self.micro_rollout_batch_size = micro_rollout_batch_size
        self.max_epochs = max_epochs
        self.tokenizer = tokenizer
        self.generate_kwargs = generate_kwargs
        self.dataloader_pin_memory = dataloader_pin_memory
        self.max_norm = max_norm
        self.ptx_coef = ptx_coef
        self.micro_train_batch_size = micro_train_batch_size
        self.kl_target = kl_target
        self.prompt_max_len = prompt_max_len
        self.gradient_checkpointing = gradient_checkpointing
        self.reward_fn = reward_fn

        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.actor_optim = actor_optim
        self.critic_optim = critic_optim
        self.actor_scheduler = actor_scheduler
        self.critic_scheduler = critic_scheduler

        self.actor_loss_fn = PolicyLoss(eps_clip)
        # Mixtral 8x7b
        self.aux_loss = self.args.aux_loss_coef > 1e-8

        if self.kl_target:
            self.kl_ctl = AdaptiveKLController(init_kl_coef, kl_target, kl_horizon)
        else:
            self.kl_ctl = FixedKLController(init_kl_coef)

        self.experience_maker = NaiveExperienceMaker(
            actor,
            critic,
            reward_model,
            initial_model,
            tokenizer,
            prompt_max_len,
            self.kl_ctl,
            strategy,
            remote_rm_url,
            reward_fn,
        )
        self.replay_buffer = NaiveReplayBuffer(micro_train_batch_size, buffer_limit, buffer_cpu_offload)
        self.eval_replay_buffer = NaiveReplayBuffer(micro_train_batch_size, buffer_limit, buffer_cpu_offload)
        self.eval_data_path = eval_data_path
        self.save_path = os.path.join(save_path, f"device_{torch.cuda.current_device()}")
        os.makedirs(self.save_path, exist_ok=True)
        self.gradients_save_path = os.path.join(self.save_path, 'gradients')
        self.output_save_path = os.path.join(self.save_path, 'output')
        self.status_save_path = os.path.join(self.save_path, 'status')
        os.makedirs(self.gradients_save_path, exist_ok=True)
        os.makedirs(self.output_save_path, exist_ok=True)
        os.makedirs(self.status_save_path, exist_ok=True)

        self.eval_gradients = []
        self.influence_scores = []

    def fit(
        self,
        args,
        prompts_dataloader,
        consumed_samples=0,
        num_update_steps_per_episodes=1,
    ) -> None: 
        num_rollouts_per_episodes = (
            num_update_steps_per_episodes * args.train_batch_size // args.max_epochs // args.rollout_batch_size
        )
        update_timesteps = args.rollout_batch_size // (self.strategy.world_size * self.micro_rollout_batch_size)

        self.prompts_dataloader = prompts_dataloader

        # Restore step and start_epoch
        steps = consumed_samples // args.rollout_batch_size * update_timesteps + 1
        start_episode = consumed_samples // args.rollout_batch_size // num_rollouts_per_episodes
        consumed_samples = consumed_samples % (num_rollouts_per_episodes * args.rollout_batch_size)


        # prepare eval data
        eval_prompts = []
        with open(self.eval_data_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                eval_prompts.append(data["prompt"])
                break
        
        greedy_generate_kwargs = {
            "do_sample": False,
            "max_new_tokens": args.generate_max_len,
            "max_length": args.max_len,
            "temperature": 1.0,
            "top_p": 1.0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "gamma": self.generate_kwargs['gamma'],
            "lambd": self.generate_kwargs["lambd"]
        }

        for prompt in eval_prompts:
            experience = self.experience_maker.make_experience(prompt, **greedy_generate_kwargs)
            self.eval_replay_buffer.append(experience)

        # compute eval gratients
        torch.cuda.empty_cache()
        eval_dataloader = DataLoader(
            self.eval_replay_buffer,
            batch_size=self.eval_replay_buffer.sample_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.eval_replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        for experience in eval_dataloader:
            experience.to_device(device)
            self.actor.train()

            num_actions = experience.action_mask.size(1)
            # actor loss
            action_log_probs, output = self.actor(
                experience.sequences, num_actions, attention_mask=experience.attention_mask, return_output=True
            )

            # loss function
            actor_loss = self.actor_loss_fn(
                action_log_probs,
                experience.action_log_probs,
                experience.advantages,
                action_mask=experience.action_mask,
            )

            loss = actor_loss
            self.actor_optim.backward(loss, self.actor, self.actor_optim)
            # save gradient
            vectorized_grads = torch.cat([p.grad.view(-1) for p in self.actor.parameters() if p.grad is not None])
            print(vectorized_grads)
            self.eval_gradients.append(vectorized_grads)
            # clear gradient
            self.clear_gradient()
            # self.actor_optim.clear_hp_grads()
            # self.actor_optim.clear_lp_grads()
        
        torch.cuda.empty_cache()

        # clear
        # self.actor_optim.zero_grad()
        # self.actor_optim.clear_hp_grads()
        # self.actor_optim.clear_lp_grads()

        for episode in range(start_episode, args.num_episodes):
            if isinstance(self.prompts_dataloader.sampler, DistributedSampler):
                self.prompts_dataloader.sampler.set_epoch(
                    episode, consumed_samples=0 if episode > start_episode else consumed_samples
                )
            pbar = tqdm(
                range(self.prompts_dataloader.__len__()),
                desc=f"Episode [{episode + 1}/{args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )

            for rand_prompts in self.prompts_dataloader:
                experience = self.experience_maker.make_experience(rand_prompts, **self.generate_kwargs)
                # print prompt/answer in each update step
                # if steps % update_timesteps == 0:
                #     output = self.tokenizer.batch_decode(experience.sequences, skip_special_tokens=True)
                #     with open(os.path.join(self.output_save_path, 'output.jsonl'), 'a') as f:
                #         data = {'output': output[0]}
                #         json.dump(data, f)
                #         f.write('\n')
                #     self.strategy.print(output[0])
                self.replay_buffer.append(experience)

                if steps % update_timesteps == 0:
                    global_steps = steps // update_timesteps

                    torch.cuda.empty_cache()
                    self.replay_buffer.normalize("advantages", self.strategy)
                    # torch.save(self.replay_buffer.items, os.path.join(self.output_save_path, 'buffer_items.pth'))
                    status = self.ppo_train(global_steps)

                    self.replay_buffer.clear()
                    torch.cuda.empty_cache()

                    if "kl" in status:
                        self.kl_ctl.update(status["kl"], args.rollout_batch_size)
                    pbar.set_postfix(status)

                    # logs/checkpoints
                    client_states = {"consumed_samples": global_steps * args.rollout_batch_size}
                pbar.update()
                steps = steps + 1

        print(self.influence_scores)

    def ppo_train(self, global_steps=0):
        # replay buffer may be empty at first, we should rebuild at each training
        dataloader = DataLoader(
            self.replay_buffer,
            batch_size=self.replay_buffer.sample_batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=self.dataloader_pin_memory,
            collate_fn=self.replay_buffer.collate_fn,
        )
        device = torch.cuda.current_device()

        status_list = []
        status_mean = {}
        for epoch in range(self.max_epochs):
            pbar = tqdm(
                dataloader,
                desc=f"Train epoch [{epoch + 1}/{self.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            training_step = 0
            for experience in pbar:
                experience.to_device(device)
                status = self.training_step(experience, global_steps + training_step)

                training_step = training_step + 1

                # with open(os.path.join(self.status_save_path, 'status.jsonl'), 'a') as f:
                #     json.dump(status, f)
                #     f.write('\n')

                # for DP
                # weighted mean for kl
                if "kl" in status:
                    status["kl"] *= status["response_length"]
                    status = self.strategy.all_reduce(status)
                    status["kl"] /= status["response_length"]

                short_status = {}

                if "policy_loss" in status:
                    short_status = {
                        "pg": status["policy_loss"],
                        "rm": status["reward"],
                        "ret": status["return"],
                        "glen": status["response_length"],
                        "tlen": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                    }

                if "critic_loss" in status:
                    short_status["cri"] = status["critic_loss"]
                    short_status["vals"] = status["values"]
                    short_status["cri_lr"] = status["critic_lr"]

                if "ptx_loss" in status:
                    short_status["ptx"] = status["ptx_loss"]

                status_list.append(status)
                pbar.set_postfix(short_status)

        if status_list:
            status_mean = status_list[0]
            for m in status_list[1:]:
                for k, v in m.items():
                    status_mean[k] += v
            for k in status_mean.keys():
                status_mean[k] /= len(status_list)
        return status_mean

    def training_step(self, experience: Experience, idx) -> Dict[str, float]:
        status = {}
        status = self.training_step_actor(experience, idx)
        return status

    def training_step_actor(self, experience: Experience, idx) -> Dict[str, float]:
        self.actor.train()

        num_actions = experience.action_mask.size(1)
        # actor loss
        action_log_probs, output = self.actor(
            experience.sequences, num_actions, attention_mask=experience.attention_mask, return_output=True
        )

        # loss function
        actor_loss = self.actor_loss_fn(
            action_log_probs,
            experience.action_log_probs,
            experience.advantages,
            action_mask=experience.action_mask,
        )

        loss = actor_loss
        # self.strategy.backward(loss, self.actor, self.actor_optim)
        self.actor_optim.backward(loss, self.actor, self.actor_optim)
        # save gradient
        vectorized_grads = torch.cat([p.grad.view(-1) for p in self.actor.parameters() if p.grad is not None])
        print(vectorized_grads)
        influences = [torch.dot(vectorized_grads, g) for g in self.eval_gradients]
        mean_influences = torch.mean(torch.tensor(influences))
        self.influence_scores.append(mean_influences)

        # torch.save(vectorized_grads, os.path.join(self.gradients_save_path, f"gradient_{idx}.pt"))
        # clear gradient
        self.clear_gradient()
        # self.actor_optim.clear_hp_grads()
        # self.actor_optim.clear_lp_grads()
        # self.actor_optim.zero_grad()

        # status
        status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}
        for k, v in experience.info.items():
            if k == "kl":
                status[k] = (
                    (v * experience.info["response_length"]).sum() / experience.info["response_length"].sum()
                ).item()
            else:
                status[k] = v.mean().item()
        return status
    
    def clear_gradient(self):
        if self.actor.model.bfloat16_enabled():
            # TODO: Temporary until bf16_optimizer and zero_optimizer are integrated
            if self.actor.model.zero_optimization() and hasattr(self.actor.model.optimizer, "zero_grad"):
                self.actor.model.optimizer.zero_grad()
            else:
                pass
        elif self.actor.model.zero_optimization() or self.actor.model.fp16_enabled() or self.actor.model.amp_enabled():
            self.actor.model.optimizer.zero_grad()
        else:
            for param_name, param in self.actor.model.module.named_parameters():
                param.grad = None