import torch

data = torch.load('.pt')

if isinstance(data, dict):
    for key, value in data.items():
        print(f"{key}: {value.shape}")
else:
    print(data.shape)