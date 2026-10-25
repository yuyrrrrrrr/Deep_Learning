import matplotlib
# 强制使用非交互式后端，确保在任何环境（如IDE、终端）都能成功渲染并保存
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 全局设置：防止中文乱码
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] # Windows用SimHei，Mac用Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 1. 准备数据
# --- 左图: lr 曲线 ---
epochs_lr = np.linspace(0, 15, 300)
warmup_epochs = 1.5 # 前10%线性升 (15的10%是1.5)

# 固定 lr
lr_fixed = np.full_like(epochs_lr, 0.1)

# Warmup + Cosine
lr_warmup_cosine = np.zeros_like(epochs_lr)
for i, t in enumerate(epochs_lr):
    if t <= warmup_epochs:
        # 线性上升阶段
        lr_warmup_cosine[i] = (t / warmup_epochs) * 0.1
    else:
        # 余弦退火阶段
        progress = (t - warmup_epochs) / (15 - warmup_epochs)
        lr_warmup_cosine[i] = 0.5 * 0.1 * (1 + np.cos(np.pi * progress))

# --- 右图: test accuracy ---
epochs_acc = np.arange(1, 16)
# 根据图片视觉特征估算的数据点
acc_fixed = [0.959, 0.962, 0.967, 0.974, 0.972, 0.966, 0.973, 0.969, 0.975, 0.972, 0.970, 0.976, 0.971, 0.974, 0.976]
acc_warmup = [0.952, 0.957, 0.960, 0.971, 0.973, 0.975, 0.979, 0.981, 0.982, 0.983, 0.983, 0.983, 0.983, 0.983, 0.983]

# 颜色定义 (尽量提取原图颜色)
color_fixed = '#7f8c8d'  # 灰色
color_warmup = '#c0392b' # 深红色

# 2. 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ==========================================
# 绘制左图 (LR 曲线)
# ==========================================
ax1.plot(epochs_lr, lr_fixed, linestyle='--', color=color_fixed, label='固定 lr=0.1', linewidth=1.5)
ax1.plot(epochs_lr, lr_warmup_cosine, color=color_warmup, label='Warmup + Cosine', linewidth=2)

ax1.set_title('lr 曲线 ( 前 10% 线性升 + 余弦退火 )')
ax1.set_xlabel('epoch')
ax1.set_ylabel('lr')
ax1.set_xlim(0, 15)
ax1.set_ylim(-0.005, 0.105)
ax1.set_xticks(np.arange(0, 16, 2))
ax1.set_yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
ax1.grid(True, linestyle='-', alpha=0.3)
# 图例放置位置
ax1.legend(loc='center right', framealpha=0.9)

# ==========================================
# 绘制右图 (Accuracy 曲线)
# ==========================================
ax2.plot(epochs_acc, acc_fixed, marker='o', color=color_fixed, label='固定 lr=0.1', linewidth=1.5, markersize=5)
ax2.plot(epochs_acc, acc_warmup, marker='o', color=color_warmup, label='Warmup + Cosine', linewidth=1.5, markersize=5)

ax2.set_title('test accuracy by epoch')
ax2.set_xlabel('epoch')
ax2.set_ylabel('test acc')
ax2.set_xlim(0, 16)
ax2.set_ylim(0.949, 0.985) # 设置合适的Y轴范围以凸显差异
ax2.set_xticks(np.arange(1, 16, 1)) # 设置X轴刻度为1到15
ax2.set_yticks([0.955, 0.960, 0.965, 0.970, 0.975, 0.980])
ax2.grid(True, linestyle='-', alpha=0.3)
# 图例放置位置
ax2.legend(loc='upper left', framealpha=0.9)

# 3. 调整布局并保存
plt.tight_layout()
save_path = 'lr_acc_comparison.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 图像已成功生成并保存在当前目录：{save_path}")