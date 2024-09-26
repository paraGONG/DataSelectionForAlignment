### **上传模型/权重到huggingface**

1、查看文件

```cmd
cd ~
cd yifan
cd DataSelectionForAlignment
ls
```

2、clone huggingface模型仓库，将数据cp到仓库中

```cmd
# A100 old
cd ~/yifan/hf
git clone https://huggingface.co/yifangong/tinyllamachat_global_step10_gradients_train_part_0-3
cd tinyllamachat_global_step10_gradients_train_part_0-3
cp -r ../../DataSelectionForAlignment/tinyllamachat_global_step10_gradients_train_part_0 .
cp -r ../../DataSelectionForAlignment/tinyllamachat_global_step10_gradients_train_part_1 .
cp -r ../../DataSelectionForAlignment/tinyllamachat_global_step10_gradients_train_part_2 .
cp -r ../../DataSelectionForAlignment/tinyllamachat_global_step10_gradients_train_part_3 .

# A100 new
cd ~/yifan/hf
git clone https://huggingface.co/yifangong/tinyllamachat_global_step10_gradients_train_part_4-7
cd tinyllamachat_global_step10_gradients_train_part_4-7
cp -r ../../DataSelectionForAlignment/tinyllamachat_global_step10_gradients_train_part_4 .
cp -r ../../DataSelectionForAlignment/tinyllamachat_global_step10_gradients_train_part_5 .
cp -r ../../DataSelectionForAlignment/tinyllamachat_global_step10_gradients_train_part_6 .
cp -r ../../DataSelectionForAlignment/tinyllamachat_global_step10_gradients_train_part_7 .

# 2*H100
cd ~/yifan/hf
git clone https://huggingface.co/yifangong/tinyllamachat_global_step10_gradients_train_part_8-9
cd tinyllamachat_global_step10_gradients_train_part_8-9
cp -r ../../DataSelectionForAlignment/tinyllamachat_global_step10_gradients_train_part_8 .
cp -r ../../DataSelectionForAlignment/tinyllamachat_global_step10_gradients_train_part_9 .
```

3、可能要登录huggingface账户

```cmd
huggingface-cli login
# token
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
```

4、上传

```cmd
sudo apt-get install git-lfs
git lfs install
git lfs track "*.jsonl"  
git add .gitattributes
git add .
git commit -m "add file"
git push

# 可能遇到设置git和huggingface登录
git config --global user.email "1240138092@qq.com"
git config --global user.name "paraGONG"
# huggingface用户名
yifangong
# 密码
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
```



### 计算梯度

```cmd
cd ~
cd yifan/DataSelectionForAlignment/ComputeInfluence
bash scripts\new_exp\gcr_2H100\saferlhf_eval.sh
```



### Alignment

```cmd
cd ~
cd yifan/DataSelectionForAlignment/Alignment
git pull
# 这里要git clone一下checkpoint
git clone https://huggingface.co/yifangong/tinyllama-warmup-ckpt
# 然后运行
bash scripts\new_exp\selected.sh
```

saferlhf_evaluation_dataset_tinyllama_random_TinyLlama-1.1B-Chat-v1.0-reward-model.jsonl

