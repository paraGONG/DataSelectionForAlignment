import torch

# 加载梯度文件
gradients_path = "/root/siton-object-46b8630eb56e449886cb89943ab6fe10/DataSelectionForAlignment/grads/test_gradients.pt"
gradients = torch.load(gradients_path)

# 查看梯度
print("Gradients loaded from gradients.pt:")
print(gradients)

# 如果你想查看具体的形状
print(f"Gradient shape: {gradients.shape}")
