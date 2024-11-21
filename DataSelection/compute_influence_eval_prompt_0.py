import os
import json
import torch
from tqdm import tqdm

def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score.

    Args:
        training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
        validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
    """
    # N x N_VALID
    influence_scores = torch.matmul(
        training_info, validation_info)
    return influence_scores.item()


def prepare_gradients_train():
    gradients_train = []
    for i in range(1):
        gradients_path = f"../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_0-3/tinyllamachat_global_step10_gradients_train_part_{i}/gradients"
        num = len([f for f in os.listdir(gradients_path)])
        for j in range(2000):
            gradients = torch.load(os.path.join(gradients_path, f"gradient_{j+1}.pt"), map_location="cpu")
            # gradients = torch.load(os.path.join(gradients_path, f"gradient_{j+1}.pt"))

            gradients_train.append(gradients)
    for i in range(1):
        gradients_path = f"../tinyllamachat_global_step10_gradients_eval_prompt_0/gradients"
        num = len([f for f in os.listdir(gradients_path)])
        for j in range(num):
            gradients = torch.load(os.path.join(gradients_path, f"gradient_{j+1}.pt"), map_location="cpu")
            # gradients = torch.load(os.path.join(gradients_path, f"gradient_{j+1}.pt"))

            gradients_train.append(gradients)
    return gradients_train


def prepare_gradients_evaluation():
    gradients_eval = []
    gradients_path = f"../tinyllamachat_global_step10_gradients_evaluation_saferlhf/gradients"
    num = len([f for f in os.listdir(gradients_path)])
    for j in range(num):
        gradients = torch.load(os.path.join(gradients_path, f"gradient_{j+1}.pt"), map_location="cpu")
        # gradients = torch.load(os.path.join(gradients_path, f"gradient_{j+1}.pt"))
        gradients_eval.append(gradients)
    return gradients_eval[7]


def compute_influence(gradients_train, gradients_eval, save_path):
    os.makedirs(save_path, exist_ok=True)
    for i, eval_gradient in enumerate(gradients_eval):
        influence_scores = []
        for train_gradient in tqdm(gradients_train, desc=f"evaluation_data_{i}"):
            influence_score = calculate_influence_score(train_gradient, eval_gradient)
            influence_scores.append(influence_score)
        with open(os.path.join(save_path, f"scores_{i}"), 'w') as f:
            json.dump(influence_scores, f)

gradients_train = prepare_gradients_train()
gradients_eval = prepare_gradients_evaluation()
compute_influence(gradients_train, gradients_eval, "../eval_influence_0")