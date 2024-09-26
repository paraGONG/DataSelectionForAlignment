import os
import json
import torch
import tqdm

def prepare_gradients_train():
    gradients_train = []
    for i in range(4):
        gradients_path = f"../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_0-3/tinyllamachat_global_step10_gradients_train_part_{i}/gradients"
        num = len([f for f in os.listdir(gradients_path)])
        for j in range(num):
            gradients_train.append(os.path.join(gradients_path, f"gradient_{j+1}.pt"))
    for i in range(4):
        gradients_path = f"../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_4-7/tinyllamachat_global_step10_gradients_train_part_{i+4}/gradients"
        num = len([f for f in os.listdir(gradients_path)])
        for j in range(num):
            gradients_train.append(os.path.join(gradients_path, f"gradient_{j+1}.pt"))
    for i in range(2):
        gradients_path = f"../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_8-9/tinyllamachat_global_step10_gradients_train_part_{i+8}/gradients"
        num = len([f for f in os.listdir(gradients_path)])
        for j in range(num):
            gradients_train.append(os.path.join(gradients_path, f"gradient_{j+1}.pt"))
    print(len(gradients_path))
prepare_gradients_train()