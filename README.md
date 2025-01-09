git clone git@github.com:paraGONG/DataSelectionForAlignment.git

conda create -n openrlhf python=3.10

cd DataSelectionForAlignment

cd ref\OpenRLHF-main

pip install -e .

pip install flash-attn

pip uninstall openrlhf

cd ../../../

sudo apt-get install git-lfs

git lfs install

git clone https://huggingface.co/yifangong/tinyllama-warmup-ckpt

cd DataSelectionForAlignment

cd BatchWiseDataSelectionV3

bash scripts\random_90.sh
