import matplotlib
# 强制使用非交互式后端，确保不弹窗也能成功渲染保存
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 全局设置：确保数学公式正常渲染
# ==========================================
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm' # 使用标准的数学字体

# 1. 准备数据
epochs = np.linspace(0, 100, 200)
lr_max = 0.1
lr_min = 0.0
# 标准余弦退火公式：lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T_max))
learning_rates = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epochs / 100))

# 2. 设置画布
fig, ax = plt.subplots(figsize=(8, 5))

# 3. 绘制主要曲线 (使用类似的绿色)
line_color = '#418a53'
ax.plot(epochs, learning_rates, color=line_color, linewidth=2.2)

# 4. 绘制特殊标记点 (Epoch=40 处的圆点和叉号)
# 根据图片目测，该点在 x=40，y 大约在 0.072 左右
mark_x = 40
mark_y = 0.072
# 画外圈白底黑边的圆点
ax.plot(mark_x, mark_y, marker='o', markerfacecolor='white', markeredgecolor='black',
        markersize=9, markeredgewidth=1.5, zorder=5)
# 画内部的黑色叉号
ax.plot(mark_x, mark_y, marker='x', color='black', markersize=6, markeredgewidth=1.5, zorder=6)

# 5. 设置坐标轴格式
ax.set_xlim(0, 100)
ax.set_ylim(0, 0.102)
ax.set_xticks(np.arange(0, 101, 20))
ax.set_yticks([0.000, 0.025, 0.050, 0.075, 0.100])
# 强制 Y 轴保留三位小数
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
ax.tick_params(axis='both', which='major', labelsize=11)

# 6. 设置标题和标签
ax.set_title('Cosine Annealing', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Learning Rate (lr)', fontsize=12)

# 7. 添加网格线 (浅灰色虚线)
ax.grid(True, which='major', linestyle='--', alpha=0.5, color='#a0a0a0')

# 8. 添加数学公式和文字说明
# 拆成三行用 $ 包裹，避免公式过长导致排版错乱
formula_line1 = r"$lr_t = \frac{1}{2}lr_{min} + \frac{1}{2}(lr_{max} - lr_{min})$"
formula_line2 = r"$\quad \times \cos\left(\frac{\pi t}{T_{max}}\right)$"
explain_text = r"\(T_{max}\): total number of epochs"

# 在指定位置放置文本
ax.text(58, 0.082, formula_line1, fontsize=13, color='#333333')
ax.text(58, 0.069, formula_line2, fontsize=13, color='#333333')
ax.text(58, 0.055, explain_text, fontsize=11, color='#333333')

# 9. 调整边框粗细 (更贴近原图质感)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#555555')

# 10. 调整布局并保存
plt.tight_layout()
save_path = 'cosine_annealing_plot.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 图像已成功生成并保存在当前目录：{save_path}")