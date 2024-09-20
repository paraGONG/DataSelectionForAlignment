import argparse
import json
import logging
import os
from typing import Any, Dict, List, Union, Optional

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict, Value, concatenate_datasets, load_dataset
from fastchat.conversation import Conversation
from transformers import PreTrainedTokenizer, AutoConfig

CORE_EVAL_SET = "allenai/reward-bench"
EXTRA_PREF_SETS = "allenai/pref-test-sets"


# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# reference for length bias categories
LENGTH_CATEGORIES = {
    "alpacaeval-easy": "True",
    "alpacaeval-hard": "True",
    "alpacaeval-length": "Neutral",
    "donotanswer": "False",
    "hep-cpp": "Neutral",
    "hep-go": "Neutral",
    "hep-java": "Neutral",
    "hep-js": "Neutral",
    "hep-python": "Neutral",
    "hep-rust": "Neutral",
    "llmbar-adver-GPTInst": "False",
    "llmbar-adver-GPTOut": "Neutral",
    "llmbar-adver-manual": "False",
    "llmbar-adver-neighbor": "False",
    "llmbar-natural": "Neutral",
    "math-prm": "Neutral",
    "mt-bench-easy": "False",
    "mt-bench-hard": "False",
    "mt-bench-med": "Neutral",
    "refusals-dangerous": "False",
    "refusals-offensive": "False",
    "xstest-should-refuse": "False",
    "xstest-should-respond": "True",
}

EXAMPLE_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

SUBSET_NAME_TO_PAPER_READY = {
    "alpacaeval-easy": "AlpacaEval Easy",
    "alpacaeval-length": "AlpacaEval Length",
    "alpacaeval-hard": "AlpacaEval Hard",
    "mt-bench-easy": "MT Bench Easy",
    "mt-bench-med": "MT Bench Medium",
    "mt-bench-hard": "MT Bench Hard",
    "llmbar-natural": "LLMBar Natural",
    "llmbar-adver-neighbor": "LLMBar Adver. Neighbor",
    "llmbar-adver-GPTInst": "LLMBar Adver. GPTInst",
    "llmbar-adver-GPTOut": "LLMBar Adver. GPTOut",
    "llmbar-adver-manual": "LLMBar Adver. Manual",
    "refusals-dangerous": "Refusals Dangerous",
    "refusals-offensive": "Refusals Offensive",
    "xstest-should-refuse": "XSTest Should Refuse",
    "xstest-should-respond": "XSTest Should Respond",
    "donotanswer": "Do Not Answer",
    "math-prm": "PRM Math",
    "hep-cpp": "HumanEvalPack CPP",
    "hep-go": "HumanEvalPack Go",
    "hep-java": "HumanEvalPack Java",
    "hep-js": "HumanEvalPack Javascript",
    "hep-python": "HumanEvalPack Python",
    "hep-rust": "HumanEvalPack Rust",
    "anthropic_harmless": "Anthropic Harmless",
    "anthropic_helpful": "Anthropic Helpful",
    "anthropic_hhh": "Anthropic HHH",
    "mtbench_gpt4": "MT Bench GPT-4",
    "mtbench_human": "MT Bench Human",
    "shp": "SHP",
    "summarize": "Summarize",
}

def load_eval_dataset(
    core_set: bool = True,
    custom_dialogue_formatting: bool = False,
    conv: Conversation = None,
    tokenizer: PreTrainedTokenizer = None,
    logger: logging.Logger = None,
    keep_columns: List[str] = ["text_chosen", "text_rejected", "id"],
    max_turns: int = None,
) -> tuple[Dataset, list[str]]:
    """
    Loads either the core eval set for HERM or the existing preference data test sets.

    Args:
        core_set: if True, load the core eval set for HERM.
        custom_dialogue_formatting: if True, format the dialogue as needed for custom models (e.g. SHP and PairRM).
        conv: fastchat conversation template.
                If None (default) the passed tokenizer needs to have a usable chat template.
        tokenizer: HuggingFace tokenizer to use. The tokenizer's chat template, if available, has precedence over conv.
        logger: logger to use for logging. If None (default), no logging is done.
        keep_columns: list of columns to keep in the dataset.
        max_turns: maximum number of turns in the dialogue (usually even). If None (default), no filtering is done.

    Returns:
        dataset: loaded dataset with required properties.
        subsets: list of subsets for the corresponding samples in the dataset.
    """
    if core_set:
        raw_dataset = load_dataset(CORE_EVAL_SET, split="filtered")
    else:
        raw_dataset = load_dataset(EXTRA_PREF_SETS)
        modified_datasets = []

        # Iterate over each subset in the DatasetDict
        for subset_name, subdataset in raw_dataset.items():
            # if subset column exists, move to subsubset (for pref sets)
            if "subset" in subdataset.column_names:
                subdataset = subdataset.rename_column("subset", "subsubset")

            # Add a new column 'subset' to the dataset with the subset name
            subdataset = subdataset.add_column("subset", [subset_name] * len(subdataset))

            # Append the modified dataset to the list
            # remove pku_safer and pku_better from the dict, no longer part of the benchmark
            if subset_name not in ["pku_safer", "pku_better"]:
                modified_datasets.append(subdataset)

        # Concatenate all the modified datasets into one dataset
        raw_dataset = concatenate_datasets(modified_datasets)

    # Apply chat template
    if not custom_dialogue_formatting:
        usable_tokenizer = check_tokenizer_chat_template(tokenizer)

        # assert either conv is passed or tokenizer has chat_template
        assert conv is not None or usable_tokenizer

        if usable_tokenizer:
            if logger is not None:
                logger.info("*** Preparing dataset with HF Transformers ***")
            # docs https://huggingface.co/docs/transformers/main/en/chat_templating
            dataset = raw_dataset.map(
                prepare_dialogue_from_tokenizer,
                fn_kwargs={"tokenizer": tokenizer},
                num_proc=8,
                load_from_cache_file=False,
            )

        # else use FastChat to get chat template
        else:
            if logger is not None:
                logger.info("*** Preparing dataset with FastChat ***")
            dataset = raw_dataset.map(
                prepare_dialogue,
                fn_kwargs={"dialogue_template": conv},
                num_proc=8,  # using >1 process causes issues with re-assigning prompt in example
                load_from_cache_file=False,
            )
    else:
        if logger is not None:
            logger.info("*** Preparing dataset with custom formatting ***")

        def map_conversations(example, core_set=True):
            if core_set:
                example["text_chosen"] = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["chosen"]},
                ]
                example["text_rejected"] = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["rejected"]},
                ]
            else:
                prompt = example["prompt"]
                example["text_chosen"] = prompt + [{"role": "assistant", "content": example["chosen"]}]
                example["text_rejected"] = prompt + [{"role": "assistant", "content": example["rejected"]}]
            return example

        dataset = raw_dataset.map(
            map_conversations,
            fn_kwargs={"core_set": core_set},
            num_proc=8,
        )

    if max_turns is not None:
        assert max_turns > 0, "max_turns must be greater than 0"

        # filter long answers (MT Bench prompt as 1 or 2 turn examples)
        def filter_long_turns(batch):
            return len(batch["text_chosen"]) <= max_turns

        dataset = dataset.filter(filter_long_turns)

    # take column subset from dataset
    subsets = dataset["subset"]

    # remove columns if set and not custom_dialogue_formatting
    all_cols = dataset.column_names
    dataset = dataset.remove_columns([c for c in all_cols if c not in keep_columns])

    return dataset, subsets


def check_tokenizer_chat_template(tokenizer):
    """
    Check if tokenizer has non none chat_template attribute.
    """
    if hasattr(tokenizer, "chat_template"):
        if tokenizer.chat_template is not None:
            return True
    return False


def prepare_dialogue_from_tokenizer(
    example: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    ift: bool = False,
) -> Dict[str, Any]:
    if all(k in example.keys() for k in ("chosen", "rejected")):
        # multi turn
        if isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
            # iterate through prompt messages, alternate user and assistant, end with example["chosen"]/rejected
            messages = []
            for i, (line) in enumerate(example["prompt"]):
                p = line["content"]
                _ = line["role"]
                if (i + 1) % 2 == 1:
                    messages.append({"role": "user", "content": p})
                else:
                    messages.append({"role": "assistant", "content": p})
            # assert that the last message before this is user
            assert messages[-1]["role"] == "user"

            # required for DPO code only, otherwise discarded
            temp_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            # end with chosen/rejected
            messages.append({"role": "assistant", "content": example["chosen"]})
            example["text_chosen"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            messages[-1] = {"role": "assistant", "content": example["rejected"]}
            example["text_rejected"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            example["prompt"] = temp_prompt
        # single turn
        else:
            # needed for DPO
            messages = [
                {"role": "user", "content": example["prompt"]},
            ]
            temp_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )

            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["chosen"]},
            ]
            example["text_chosen"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["rejected"]},
            ]
            example["text_rejected"] = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            example["prompt"] = temp_prompt
    elif ift:
        # TODO adapt this for DPO models with tokenize_row function
        messages = [
            {"role": "user", "content": example["prompt"]},
            {"role": "assistant", "content": example["input"]},
        ]
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
        )
    else:
        raise ValueError(
            "Could not format example as dialogue for `rm` task!"
            f"Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example


def prepare_dialogue(
    example: Dict[str, Any],
    dialogue_template: Conversation,
    ift: bool = False,
) -> Dict[str, Any]:
    """Format example to single- or multi-turn dialogue."""
    if all(k in example.keys() for k in ("chosen", "rejected")):
        # multi turn
        if isinstance(example["prompt"], list) and len(example["prompt"]) > 0:
            # iterate through prompt messages, alternate user and assistant, end with example["chosen"]/rejected
            dialogue_template.messages = []
            for i, (line) in enumerate(example["prompt"]):
                p = line["content"]
                _ = line["role"]
                if (i + 1) % 2 == 1:
                    dialogue_template.messages.append([dialogue_template.roles[0], p])
                else:
                    dialogue_template.messages.append([dialogue_template.roles[1], p])
            # assert that the last message before this is user
            assert dialogue_template.messages[-1][0] == dialogue_template.roles[0]

            # needed for DPO
            temp_prompt = dialogue_template.get_prompt()

            # end with chosen/rejected
            dialogue_template.messages.append([dialogue_template.roles[1], example["chosen"]])
            example["text_chosen"] = dialogue_template.get_prompt()

            dialogue_template.messages[-1] = [dialogue_template.roles[1], example["rejected"]]
            example["text_rejected"] = dialogue_template.get_prompt()

            example["prompt"] = temp_prompt

        # single turn
        else:
            if isinstance(example["prompt"], list):
                example["prompt"] = example["prompt"][0]
            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
            ]
            temp_prompt = dialogue_template.get_prompt()

            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
                [dialogue_template.roles[1], example["chosen"]],
            ]
            example["text_chosen"] = dialogue_template.get_prompt()
            dialogue_template.messages = [
                [dialogue_template.roles[0], example["prompt"]],
                [dialogue_template.roles[1], example["rejected"]],
            ]
            example["text_rejected"] = dialogue_template.get_prompt()

            example["prompt"] = temp_prompt
    elif ift:
        if isinstance(example["prompt"], list):
            example["prompt"] = example["prompt"][0]

        dialogue_template.messages = [
            [dialogue_template.roles[0], example["prompt"]],
        ]
        temp_prompt = dialogue_template.get_prompt()
        dialogue_template.messages = [
            [dialogue_template.roles[0], example["prompt"]],
            [dialogue_template.roles[1], example["input"]],
        ]
        example["text"] = dialogue_template.get_prompt()
        example["prompt"] = temp_prompt  # needed for DPO

    else:
        raise ValueError(
            "Could not format example as dialogue for `rm` task!"
            f"Require `[chosen, rejected]` keys but found {list(example.keys())}"
        )
    return example

def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    """
    Helper function for immediately logging RewardBench scores.
    """
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                total_weighted_score += metrics[test] * example_counts[test]
                total_examples += example_counts[test]
        if total_examples > 0:
            section_scores[section] = total_weighted_score / total_examples
        else:
            section_scores[section] = 0
    return section_scores


def _get_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="value_head", packing_samples=False):
    class RewardModel(base_pretrained_model):
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            super().__init__(config)
            setattr(self, self.base_model_prefix, base_llm_model(config))

            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            self.packing_samples = packing_samples

            # mean std
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # load mean/std from config.json
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            return_output=False,
        ) -> torch.Tensor:
            if not self.packing_samples:
                # https://github.com/OpenRLHF/OpenRLHF/issues/217
                position_ids = attention_mask.long().cumsum(-1) - 1
            else:
                # reset the positions for packed samples
                position_ids = reset_position_ids(attention_mask)
            position_ids.masked_fill_(attention_mask == 0, 1)

            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)
            print(values.device)
            if self.packing_samples:
                reward = values
            else:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            return (reward, outputs) if return_output else reward

    return RewardModel

def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


class RewardPipeline:
    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer

    def __call__(self, samples, return_inputs=False, **kwargs):
        _ = kwargs.get("batch_size", 1)
        truncation = kwargs.get("truncation", True)
        padding = kwargs.get("padding", True)
        max_length = kwargs.get("max_length", 2048)
        inputs = self.tokenizer(
            samples,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            return_tensors="pt",
        ).to("cuda")

        if self.tokenizer.bos_token:
            bos_token_id = self.tokenizer.bos_token_id
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Ensure input_ids is 2D
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                attention_mask = attention_mask.unsqueeze(0)

            # Find the start of each sequence (first non-pad token)
            seq_starts = attention_mask.argmax(dim=1)

            # Check for double BOS tokens
            seq_second = torch.clamp(seq_starts + 1, max=input_ids.size(1) - 1)
            double_bos_mask = (input_ids[torch.arange(input_ids.size(0)), seq_starts] == bos_token_id) & (
                input_ids[torch.arange(input_ids.size(0)), seq_second] == bos_token_id
            )

            # Set attention mask to 0 for the first BOS token where double BOS is detected
            if double_bos_mask.any():
                attention_mask[
                    torch.arange(attention_mask.size(0), device=attention_mask.device)[double_bos_mask],
                    seq_starts[double_bos_mask],
                ] = torch.tensor(0, device=attention_mask.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        if return_inputs:
            return outputs, inputs
        else:
            return outputs