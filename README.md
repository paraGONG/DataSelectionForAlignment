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
git lfs install
git add .
git commit -m "add file"
git push
```

