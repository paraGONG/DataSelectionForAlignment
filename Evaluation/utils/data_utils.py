import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset, Dataset
from fastchat.conversation import get_conv_template


def build_prompt(text, model_name):
    conv = None
    if "llama-2" in model_name.lower():
        conv = get_conv_template("llama-2")
    elif "tinyllama" in model_name.lower():
        # conv = get_conv_template("TinyLlama")
        return f"<|user|>\n{text}</s>\n<|assistant|>\n"
    else:
        return text
    
    if conv:
        conv.messages = []
        text = [text]
        for i, message in enumerate(text):
            conv.append_message(conv.roles[i%2], message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    return prompt


def load_local_dataset(data_path, start=0):
    data = []
    with open(data_path, "r") as f:
        for index, line in enumerate(f):
            if index < start:
                continue
            line = json.loads(line)
            data.append({"prompt": line["prompt"], "source": line["source"]})
    return data

def load_remote_dataset(data_path, start=0):
    dataset = load_dataset(data_path, split="test")
    data = []
    for line in dataset:
        data.append({"prompt": line["prompt"], "source": line["source"]})
    return data