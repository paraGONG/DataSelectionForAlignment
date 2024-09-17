import torch

# 加载 checkpoint 文件
checkpoint_path = "/root/siton-object-46b8630eb56e449886cb89943ab6fe10/DataSelectionForAlignment/ckpt/checkpoints_ppo/_actor/global_step5/mp_rank_00_model_states.pt"
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))  # 使用 map_location 将其加载到 CPU

# # 查看 checkpoint 中的键
# print("Keys in the checkpoint:")
# for key in checkpoint.keys():
#     print(key)

# # 获取模型的状态字典
# model_state_dict = checkpoint.get('module', checkpoint)  # 如果是使用多GPU，可能需要用 'module'

# # 打印模型中的参数
# for param_tensor in model_state_dict:
#     print(f"Tensor name: {param_tensor}, Size: {model_state_dict[param_tensor].size()}")

# # 查看训练的全局步骤数或其他信息
# if 'global_steps' in checkpoint:
#     print(f"Global steps: {checkpoint['global_steps']}")

# # 如果包含优化器状态，可以这样查看
# if 'optimizer' in checkpoint:
#     optimizer_state_dict = checkpoint['optimizer']
#     for key in optimizer_state_dict.keys():
#         print(key)