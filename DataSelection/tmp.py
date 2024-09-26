import torch

data = torch.load('../tinyllamachat_global_step10_gradients_train/tinyllamachat_global_step10_gradients_train_part_0-3/tinyllamachat_global_step10_gradients_train_part_0/gradients/gradient_1.pt')

if isinstance(data, dict):
    for key, value in data.items():
        print(f"{key}: {value.shape}")
else:
    print(data.shape)