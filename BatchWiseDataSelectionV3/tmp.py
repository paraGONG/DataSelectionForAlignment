import numpy as np
import matplotlib.pyplot as plt
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="path")
    parser.add_argument('--path', type=str, required=True, help="path")
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 调用保存图像的函数并传递保存路径
    # 加载影响分数
    influence_scores = np.load(f'{args.path}.npy')

    # 使用matplotlib查看分布，通常使用直方图
    plt.hist(influence_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title(f'{args.path}')
    plt.xlabel('Influence Score')
    plt.ylabel('Frequency')
    plt.savefig(f'../../tmp/{args.path}.png')


if __name__ == "__main__":
    main()