import numpy as np
import matplotlib.pyplot as plt

# 加载影响分数
influence_scores = np.load('influence_scores_0_step_32.npy')

# 使用matplotlib查看分布，通常使用直方图
plt.hist(influence_scores, bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('influence_scores_0_step_32')
plt.xlabel('Influence Score')
plt.ylabel('Frequency')
plt.savefig('influence_scores_0_step_32.png')
