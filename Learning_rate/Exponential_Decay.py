import matplotlib
# 强制使用非交互式后端
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 关键设置：确保数学公式正常渲染
# ==========================================
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'

# 1. 准备数据 (生成带轻微曲率的衰减曲线)
epochs = np.arange(0, 101)
lr0 = 0.1
lr_end = 1e-5

# 为了还原原图中“稍微弯曲”的视觉效果，我们不再使用纯指数衰减，
# 而是使用一个指数项略微偏移的公式：lr = lr0 * exp(-a * t^b)
# 其中 b 略小于 1，使得曲线在对数坐标下向下弯曲（前期降得快，后期变缓）
b = 0.85
# 反推 a 保证 t=100 时正好是 1e-5
a = np.log(lr0 / lr_end) / (100 ** b)
learning_rates = lr0 * np.exp(-a * (epochs ** b))

# 2. 设置画布
fig, ax = plt.subplots(figsize=(9, 5.5))

# 3. 绘制折线图
line_color = '#3b5e9e'
ax.plot(epochs, learning_rates, color=line_color, linewidth=2)

# 4. 设置 Y 轴为对数坐标系
ax.set_yscale('log')

# 5. 设置标题和坐标轴标签
ax.set_title('Exponential Decay', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Learning Rate (lr)', fontsize=13)

# 6. 设置坐标轴刻度
ax.set_xticks(np.arange(0, 101, 20))
ax.set_yticks(np.logspace(-5, -1, 5))
ax.tick_params(axis='both', which='major', labelsize=12)

# 7. 添加网格线
ax.grid(True, which='major', linestyle='--', alpha=0.6, color='#b0b0b0')

# 8. 添加文本框内容（使用 $ 符号触发公式渲染）
formula_text = r"$lr_t = lr_0 \times \gamma^t \quad (\gamma < 1)$"
explain_text = r"$lr_0$: initial learning rate" + "\n" + r"$\gamma$: decay rate"

ax.text(45, 4e-3, formula_text, fontsize=15, color=line_color, verticalalignment='top')
ax.text(45, 1.5e-3, explain_text, fontsize=12, color='#1a1a1a', verticalalignment='top', linespacing=1.5)

# 9. 调整边框
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#555555')

# 10. 调整布局
plt.tight_layout()

# ==========================================
# 保存图片到本地
# ==========================================
save_path = 'exponential_decay_curved.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 图像已成功生成并保存在当前目录：{save_path}")