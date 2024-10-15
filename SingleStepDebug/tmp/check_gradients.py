import torch

# 加载 .pt 文件
vectorized_grads = torch.load('/root/siton-object-46b8630eb56e449886cb89943ab6fe10/ComputeInfluence/grads/test_gradients5.pt')
print(vectorized_grads)
print("Shape:", vectorized_grads.shape)
print("Data type:", vectorized_grads.dtype)
print("Min value:", vectorized_grads.min().item())
print("Max value:", vectorized_grads.max().item())
