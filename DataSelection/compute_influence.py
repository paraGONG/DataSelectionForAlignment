import os
import json
import torch
import tqdm

def prepare_gradients_train():
    gradients_train = []
    for i in range(4):
        gradients_path = f"tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_0-3/tinyllamachat_global_step10_gradients_train_part_{i}/gradients"
        num = len([f for f in os.listdir(gradients_path)])
        print(num)

prepare_gradients_train()