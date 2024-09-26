# import torch

# data = torch.load('../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_0-3/tinyllamachat_global_step10_gradients_train_part_0/gradients/gradient_1.pt')

# if isinstance(data, dict):
#     for key, value in data.items():
#         print(f"{key}: {value.shape}")
# else:
#     print(data.shape)
import torch


def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score.

    Args:
        training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
        validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
    """
    # N x N_VALID
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores

data1 = torch.load('../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_0-3/tinyllamachat_global_step10_gradients_train_part_0/gradients/gradient_1.pt')
data2 = torch.load('../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_0-3/tinyllamachat_global_step10_gradients_train_part_0/gradients/gradient_2.pt')
influence_score = calculate_influence_score(data1, data2)
print(influence_score.shape)
print(influence_score)