import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，防止中文显示为方块
# Windows 用户可以使用 'SimHei'，Mac 用户可以使用 'Arial Unicode MS' 或 'PingFang HK'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 设置随机种子，保证每次生成的模拟数据形状一致
np.random.seed(42)

# ==========================================
# 1. 模拟生成数据
# ==========================================

# --- 图 1: lr 曲线 ---
steps = np.arange(0, 2500)
# 固定 lr=0.02
lr_fixed = np.full_like(steps, 0.02, dtype=float)
# Warmup -> 0.02 (约 200 步线性上升)
warmup_steps = 200
lr_warmup = np.minimum(steps / warmup_steps * 0.02, 0.02)

# --- 图 2: 前 500 个 batch 的 loss ---
batches = np.arange(1, 501)
# 构造基础下降趋势
base_trend = 2.5 * np.exp(-batches / 60) + 0.3

# 模拟 Warmup 的 Loss (红线)：平滑下降，伴随较小震荡
noise_warmup = np.random.normal(0, 0.15, size=len(batches))
loss_warmup = base_trend * np.exp(noise_warmup)

# 模拟固定 lr=0.02 的 Loss (灰线)：开头极高尖峰，随后震荡更剧烈
loss_fixed = base_trend * np.exp(np.random.normal(0, 0.25, size=len(batches)))
# 注入开头的尖峰 (根据文案，最大为 17.73)
loss_fixed[0] = 17.73
loss_fixed[1] = 8.5
loss_fixed[2] = 5.0
# 让灰线整体略微低一点点（符合原图视觉）
loss_fixed[3:] = loss_fixed[3:] * 0.85

# --- 图 3: test accuracy by epoch ---
epochs = np.array([1, 2, 3, 4, 5])
# 数据根据图片目测提取
acc_fixed = np.array([0.921, 0.895, 0.926, 0.928, 0.918])  # 灰色
acc_warmup = np.array([0.921, 0.944, 0.946, 0.932, 0.949]) # 红色

# 颜色定义（尽量还原原图）
color_fixed = '#7f8c8d'  # 灰色
color_warmup = '#c0392b' # 砖红色

# ==========================================
# 2. 绘图
# ==========================================

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# --- 绘制图 1: lr 曲线 ---
ax1.plot(steps, lr_fixed, color=color_fixed, label='固定 lr=0.02', linewidth=1.5)
ax1.plot(steps, lr_warmup, color=color_warmup, label='Warmup -> 0.02', linewidth=1.5)
ax1.set_title('lr 曲线')
ax1.set_xlabel('batch step')
ax1.set_ylabel('lr')
ax1.set_yticks(np.arange(0, 0.0225, 0.0025)) # 设置Y轴刻度间隔
ax1.grid(True, alpha=0.3)
ax1.legend(loc='lower right')

# --- 绘制图 2: 前 500 个 batch 的 loss ---
ax2.plot(batches, loss_fixed, color=color_fixed, label='固定 lr=0.02', linewidth=1.2, alpha=0.8)
ax2.plot(batches, loss_warmup, color=color_warmup, label='Warmup', linewidth=1.2, alpha=0.8)
ax2.set_title('前 500 个 batch 的 loss——固定那条开头被冲飞了')
ax2.set_xlabel('batch step')
ax2.set_ylabel('train loss (log)')
ax2.set_yscale('log')  # 设置 Y 轴为对数坐标
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# --- 绘制图 3: test accuracy by epoch ---
ax3.plot(epochs, acc_fixed, marker='o', color=color_fixed, label='固定 lr=0.02', linewidth=1.5)
ax3.plot(epochs, acc_warmup, marker='o', color=color_warmup, label='Warmup', linewidth=1.5)
ax3.set_title('test accuracy by epoch')
ax3.set_xlabel('epoch')
ax3.set_ylabel('test acc')
ax3.set_xticks(epochs)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='lower right')

# 调整布局并显示
plt.tight_layout()
plt.show()