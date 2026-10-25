import matplotlib
# 强制使用非交互式后端，确保在任何环境都能成功渲染并保存
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 全局设置：防止中文乱码
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] # Windows用SimHei，Mac用Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 1. 准备数据
epochs = np.arange(1, 6)

# --- 左图: Train loss 数据 ---
# 全零：平缓在 2.30 左右
loss_zero = [2.30, 2.30, 2.30, 2.30, 2.30]
# N(0, 1)：梯度爆炸，无法绘制有效曲线，用 NaN 表示
loss_n01 = [np.nan] * 5
# N(0, 0.01)：起点高，快速下降
loss_n001 = [2.10, 0.40, 0.15, 0.10, 0.10]
# Xavier 正态：平滑下降
loss_xavier = [0.30, 0.10, 0.06, 0.045, 0.04]
# He 正态：略低于 Xavier
loss_he = [0.28, 0.08, 0.05, 0.035, 0.03]

# --- 右图: Test acc 数据 ---
# 全零：接近随机猜
acc_zero = [0.10, 0.10, 0.10, 0.10, 0.10]
# N(0, 1)：同样用 NaN 表示
acc_n01 = [np.nan] * 5
# N(0, 0.01)：起步慢但最终达到约 96%
acc_n001 = [0.75, 0.92, 0.95, 0.96, 0.96]
# Xavier：直接起飞并平稳
acc_xavier = [0.95, 0.96, 0.97, 0.977, 0.977]
# He：略微领先
acc_he = [0.96, 0.965, 0.972, 0.978, 0.978]

# 2. 创建画布 (1行2列)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 定义统一的图例标签
labels = ['全零', 'N(0, 1)', 'N(0, 0.01)', 'Xavier 正态', 'He 正态']

# ==========================================
# 绘制左图 (Train loss)
# ==========================================
# 使用 marker='o', markersize=4 还原图中的小圆点
ax1.plot(epochs, loss_zero, marker='o', markersize=4, label=labels[0], color='#1f77b4')
ax1.plot(epochs, loss_n01, marker='o', markersize=4, label=labels[1], color='#ff7f0e') # NaN不会被画出，但图例会保留
ax1.plot(epochs, loss_n001, marker='o', markersize=4, label=labels[2], color='#2ca02c')
ax1.plot(epochs, loss_xavier, marker='o', markersize=4, label=labels[3], color='#d62728')
ax1.plot(epochs, loss_he, marker='o', markersize=4, label=labels[4], color='#9467bd')

ax1.set_title('Train loss')
ax1.set_xlabel('epoch')
ax1.set_xticks(epochs)
ax1.grid(True, linestyle='-', alpha=0.3)
ax1.legend(loc='center right', framealpha=0.9) # 图例放置在右侧中间

# ==========================================
# 绘制右图 (Test acc)
# ==========================================
ax2.plot(epochs, acc_zero, marker='o', markersize=4, label=labels[0], color='#1f77b4')
ax2.plot(epochs, acc_n01, marker='o', markersize=4, label=labels[1], color='#ff7f0e')
ax2.plot(epochs, acc_n001, marker='o', markersize=4, label=labels[2], color='#2ca02c')
ax2.plot(epochs, acc_xavier, marker='o', markersize=4, label=labels[3], color='#d62728')
ax2.plot(epochs, acc_he, marker='o', markersize=4, label=labels[4], color='#9467bd')

ax2.set_title('Test acc')
ax2.set_xlabel('epoch')
ax2.set_xticks(epochs)
ax2.grid(True, linestyle='-', alpha=0.3)
ax2.legend(loc='center right', framealpha=0.9) # 图例放置在右侧中间

# 3. 调整布局并保存
plt.tight_layout()
save_path = 'init_methods_comparison.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 图像已成功生成并保存在当前目录：{save_path}")