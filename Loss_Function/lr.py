import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体，防止中文显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS'] # Windows用SimHei，Mac用Arial Unicode MS
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 1. 准备数据 (根据笔记和图片视觉特征估算)
epochs = [1, 2, 3, 4, 5]

# --- 左侧图: 训练 loss 数据 ---
# lr=10: 第一步飙升到三位数，然后一直停留在极高值（视觉效果上相当于贴近0，结合笔记 loss≈2.34）
loss_lr10 = [242.0, 2.34, 2.34, 2.34, 2.34]
# lr=1.0: 震荡下降
loss_lr1 = [2.0, 1.5, 1.2, 1.1, 1.07]
# lr=0.1: 平稳下降 (甜区)
loss_lr01 = [0.6, 0.2, 0.1, 0.08, 0.07]
# lr=1e-5: 龟速，几乎不动
loss_lr1e5 = [2.31, 2.31, 2.31, 2.31, 2.31]

# --- 右侧图: 测试 accuracy 数据 ---
# lr=10: 卡在10%
acc_lr10 = [0.10, 0.10, 0.10, 0.10, 0.10]
# lr=1.0: 震荡上升，有明显波动
acc_lr1 = [0.33, 0.45, 0.60, 0.55, 0.58]
# lr=0.1: 平稳升至最高点
acc_lr01 = [0.95, 0.96, 0.97, 0.97, 0.97]
# lr=1e-5: 龟速缓慢上升
acc_lr1e5 = [0.10, 0.11, 0.12, 0.13, 0.15]


# 2. 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 3. 绘制左图：训练 Loss
ax1.plot(epochs, loss_lr10, marker='o', color='#b30000', label='lr=10.0', linewidth=1.5)
ax1.plot(epochs, loss_lr1, marker='o', color='#e68a00', label='lr=1.0', linewidth=1.5)
ax1.plot(epochs, loss_lr01, marker='o', color='#2ca02c', label='lr=0.1', linewidth=1.5)
ax1.plot(epochs, loss_lr1e5, marker='o', color='#1f77b4', label='lr=1e-05', linewidth=1.5)

ax1.set_title('训练 loss ( lr=10 爆 / lr=1 震荡 / lr=0.1 正常 / lr=1e-5 龟速 )')
ax1.set_xlabel('epoch')
ax1.set_ylabel('train loss')
ax1.set_xticks(epochs)
ax1.grid(True, linestyle='-', alpha=0.7)
ax1.legend()

# 4. 绘制右图：测试 Accuracy
ax2.plot(epochs, acc_lr10, marker='o', color='#b30000', label='lr=10.0', linewidth=1.5)
ax2.plot(epochs, acc_lr1, marker='o', color='#e68a00', label='lr=1.0', linewidth=1.5)
ax2.plot(epochs, acc_lr01, marker='o', color='#2ca02c', label='lr=0.1', linewidth=1.5)
ax2.plot(epochs, acc_lr1e5, marker='o', color='#1f77b4', label='lr=1e-05', linewidth=1.5)

ax2.set_title('测试 accuracy')
ax2.set_xlabel('epoch')
ax2.set_ylabel('test acc')
ax2.set_xticks(epochs)
ax2.grid(True, linestyle='-', alpha=0.7)
ax2.legend()

# 5. 调整布局并显示
plt.tight_layout()
plt.show()