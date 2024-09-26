import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, AutoModel, pipeline

from utils.model_utils import _get_reward_model
from utils.data_utils import build_prompt, load_local_dataset, load_remote_dataset


def parse_arguments():
    parser = argparse.ArgumentParser(description='Value Inference')
    parser.add_argument("-d", "--data", type=str, default=None, help="evaluation data path")
    parser.add_argument("-m", "--model", type=str, default=None, help="evaluation model path")
    parser.add_argument("-b", "--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("-i", "--device", type=str, default="cuda:0", help="device")
    parser.add_argument("-o", "--output_dir", type=str, default="./evaluation_results", help="output dir")
    parser.add_argument("-f", "--inference", action="store_true", help="if inference model or use a existing file")
    parser.add_argument("-r", "--reward", action="store_true", help="if compute reward")
    parser.add_argument("-rm", "--reward_model", type=str, default="yifangong/TinyLlama-1.1B-Chat-v1.0-reward-model", help="reward model")
    args = parser.parse_args()
    return args


class TextDataset(Dataset):
    def __init__(self, args, tokenizer, data, max_length=2048):
        self.args = args
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = build_prompt(self.data[idx]["prompt"], self.args.model)
        encoding = self.tokenizer(
            prompt,
            add_special_tokens=False,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        return {
            "prompt": prompt,
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }


def model_inference(args, infer_data):
    results = [x for x in infer_data if "answer" in x]
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model,
        model_max_length = 2048,
        padding_side = "left",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm_config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True
    )
    llm_config.pad_token_id = tokenizer.pad_token_id
    llm_config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=llm_config,
        trust_remote_code=True
    )
    model = model.to(device)

    generation_config = GenerationConfig.from_pretrained(
        args.model,
        max_new_tokens = 1024,
    )
    
    dataset = TextDataset(args, tokenizer, infer_data[len(results):], max_length=2048)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model.eval()
    for batch in tqdm(dataloader, desc="inference"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # print(batch["prompt"])
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config)
            answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # print(answers)
        for answer in answers:
            line = infer_data[len(results)]
            line["answer"] = answer
            results.append(line)
    return results


def reward_computation(args, infer_data):
    device = torch.device(args.device)
    # reward_model, tokenizer = AutoModelForSequenceClassification.from_pretrained(args.reward_model), AutoTokenizer.from_pretrained(args.reward_model)
    # reward_model = reward_model.to(device)
    # rewards = []
    # for idx, line in tqdm(enumerate(infer_data), desc="computing the reward for inferred data."):
    #     question = line["prompt"]
    #     answer = line["answer"].replace(build_prompt(question, args.model), "").lstrip()
    #     inputs = tokenizer(question, answer, return_tensors='pt').to(device)
    #     score = reward_model(**inputs).logits[0].cpu().detach().item()
    #     rewards.append(score)
    #     line["reward"] = score
    # print("avg reward score: ", np.mean(rewards))
    tokenizer = AutoTokenizer.from_pretrained(args.reward_model)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    cls_class = _get_reward_model(base_pretrained_class, base_class)
    reward_model = cls_class.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    rewards = []
    for idx, line in tqdm(enumerate(infer_data), desc="computing the reward for inferred data."):
        question = line["prompt"]
        answer = line["answer"].replace(build_prompt(question, args.model), "").lstrip()
        inputs = tokenizer(question, answer, return_tensors='pt').to(device)
        score = reward_model(**inputs).cpu().detach().item()
        rewards.append(score)
        line["reward"] = score
    print("avg reward score: ", np.mean(rewards))
    return infer_data


def save_results(output_file, results):
    with open(output_file, "w") as fw:
        for result in results:
            fw.write(json.dumps(result) + "\n")


if __name__ == '__main__':
    args = parse_arguments()
    print("Running with arguments: ", args)

    # infer model outputs on test prompts
    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model.split("/")[-1]
    data_name = args.data.split("/")[-1]
    reward_model_name = args.reward_model.split("/")[-1]
    output_file = os.path.join(args.output_dir, f"{data_name}_{model_name}_{reward_model_name}.jsonl")
    infer_data = []
    if os.path.exists(output_file):
        lines = open(output_file, "r").readlines()
        infer_data = [json.loads(line) for line in lines]

    if os.path.exists(args.data):
        data = load_local_dataset(args.data)
    else:
        data = load_remote_dataset(args.data)
    infer_data = infer_data[:] + data[len(infer_data):]
    infer_data = infer_data[:2]

    if args.inference:
        print(f"inference on dataset {args.data} with model {args.model}...")
        results = model_inference(args, infer_data)
        
        with open(output_file, "w") as fw:
            for result in results:
                fw.write(json.dumps(result) + "\n")
    else:
        lines = open(output_file, "r").readlines()
        results = [json.loads(line) for line in lines]
    
    # compute reward
    if args.reward:
        print("reward computation...")
        results = reward_computation(args, results)
        save_results(output_file, results)
    