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
# --- 左图: lr 曲线对比 ---
epochs_lr = np.linspace(0, 15, 300)
max_lr = 0.1

# (1) 固定 lr=0.1
lr_fixed = np.full_like(epochs_lr, max_lr)

# (2) Warmup + Cosine (峰值在 epoch=2)
warmup_epochs = 2.0
lr_warmup = np.zeros_like(epochs_lr)
for i, t in enumerate(epochs_lr):
    if t <= warmup_epochs:
        lr_warmup[i] = (t / warmup_epochs) * max_lr
    else:
        progress = (t - warmup_epochs) / (15 - warmup_epochs)
        lr_warmup[i] = 0.5 * max_lr * (1 + np.cos(np.pi * progress))

# (3) OneCycleLR (峰值约在 epoch=4.5)
peak_epoch = 4.5
lr_onecycle = np.zeros_like(epochs_lr)
for i, t in enumerate(epochs_lr):
    if t <= peak_epoch:
        # 平滑上升段 (半余弦)
        lr_onecycle[i] = 0.5 * max_lr * (1 - np.cos(np.pi * t / peak_epoch))
    else:
        # 平滑下降段 (半余弦)
        progress = (t - peak_epoch) / (15 - peak_epoch)
        lr_onecycle[i] = 0.5 * max_lr * (1 + np.cos(np.pi * progress))


# --- 右图: test accuracy by epoch ---
epochs_acc = np.arange(1, 16)
# 提取原图数据点
acc_fixed = [0.959, 0.962, 0.968, 0.974, 0.972, 0.966, 0.973, 0.969, 0.975, 0.972, 0.970, 0.976, 0.971, 0.974, 0.976]
acc_warmup = [0.952, 0.958, 0.960, 0.971, 0.973, 0.975, 0.979, 0.981, 0.982, 0.983, 0.983, 0.983, 0.983, 0.983, 0.983]
acc_onecycle = [0.949, 0.964, 0.965, 0.972, 0.974, 0.971, 0.976, 0.979, 0.980, 0.981, 0.982, 0.982, 0.983, 0.983, 0.982]

# 颜色定义
color_fixed = '#7f8c8d'   # 灰色
color_warmup = '#b22222'  # 深红色
color_onecycle = '#2b7bba' # 蓝色

# 2. 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ==========================================
# 绘制左图 (LR 曲线)
# ==========================================
ax1.plot(epochs_lr, lr_fixed, linestyle='--', color=color_fixed, label='固定 lr=0.1', linewidth=1.5)
ax1.plot(epochs_lr, lr_warmup, color=color_warmup, label='Warmup + Cosine', linewidth=2)
ax1.plot(epochs_lr, lr_onecycle, color=color_onecycle, label='OneCycleLR', linewidth=2)

ax1.set_title('lr 曲线对比')
ax1.set_xlabel('epoch')
ax1.set_ylabel('lr')
ax1.set_xlim(-0.5, 15.5)
ax1.set_ylim(-0.005, 0.11)
ax1.set_xticks(np.arange(0, 16, 2))
ax1.set_yticks([0.00, 0.02, 0.04, 0.06, 0.08, 0.10])
ax1.grid(True, linestyle='-', alpha=0.3)
ax1.legend(loc='center right', framealpha=0.9)

# ==========================================
# 绘制右图 (Accuracy 曲线)
# ==========================================
# 绘制线条，设置 marker、markerfacecolor 和 markeredgecolor 以还原原图的圆点样式
ax2.plot(epochs_acc, acc_fixed, marker='o', color=color_fixed, label='固定 lr=0.1',
         linewidth=1.5, markersize=5, markerfacecolor='#bdc3c7', markeredgecolor='#7f8c8d')
ax2.plot(epochs_acc, acc_warmup, marker='o', color=color_warmup, label='Warmup + Cosine',
         linewidth=1.5, markersize=5, markerfacecolor='#e74c3c', markeredgecolor=color_warmup)
ax2.plot(epochs_acc, acc_onecycle, marker='o', color=color_onecycle, label='OneCycleLR',
         linewidth=1.5, markersize=5, markerfacecolor='#3498db', markeredgecolor=color_onecycle)

ax2.set_title('test accuracy by epoch')
ax2.set_xlabel('epoch')
ax2.set_ylabel('test acc')
ax2.set_xlim(0.5, 15.5)
ax2.set_ylim(0.947, 0.985)
ax2.set_xticks(np.arange(1, 16, 2))
ax2.set_yticks([0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980])
ax2.grid(True, linestyle='-', alpha=0.3)
ax2.legend(loc='upper left', framealpha=0.9)

# 3. 调整布局并保存
plt.tight_layout()
save_path = 'lr_acc_comparison_3lines.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 图像已成功生成并保存在当前目录：{save_path}")