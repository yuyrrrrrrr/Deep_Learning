import matplotlib

# 强制使用非交互式后端，确保不弹窗也能成功渲染保存
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 全局设置：确保数学公式正常渲染
# ==========================================
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'  # 使用标准的数学字体

# 1. 准备数据 (分段构造热重启曲线)
epochs = np.linspace(0, 100, 500)
learning_rates = np.zeros_like(epochs)
lr_max = 0.1

# 根据原图目测的重启节点 (每个周期的终点)
restart_points = [0, 20, 42, 65, 86]

# 循环填充每个周期的余弦曲线
for i in range(len(restart_points) - 1):
    start = restart_points[i]
    end = restart_points[i + 1]
    T_i = end - start  # 当前周期的长度

    # 获取当前周期内的所有点
    mask = (epochs >= start) & (epochs < end)

    # 标准余弦退火公式：0.5 * lr_max * (1 + cos(pi * t / T_i))
    learning_rates[mask] = 0.5 * lr_max * (1 + np.cos(np.pi * (epochs[mask] - start) / T_i))

# 最后一个周期（86之后）保持为 0（根据原图特征）
learning_rates[epochs >= 86] = 0.0

# 2. 设置画布
fig, ax = plt.subplots(figsize=(8, 5))

# 3. 绘制主要曲线 (深紫色)
line_color = '#5c1a8a'
ax.plot(epochs, learning_rates, color=line_color, linewidth=2)

# 4. 设置坐标轴格式
ax.set_xlim(0, 102)
ax.set_ylim(0, 0.105)
ax.set_xticks(np.arange(0, 101, 20))
ax.set_yticks([0.000, 0.025, 0.050, 0.075, 0.100])
# 强制 Y 轴保留三位小数
ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
ax.tick_params(axis='both', which='major', labelsize=11)

# 5. 设置标题和标签
ax.set_title('Cosine Annealing with Warm Restarts', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Learning Rate (lr)', fontsize=12)

# 6. 添加网格线 (浅灰色虚线)
ax.grid(True, which='major', linestyle='--', alpha=0.5, color='#a0a0a0')

# 7. 添加文字说明 (右上角紫色文本)
# 使用 $ 符号包裹 T_i 以确保下标渲染正常
text_str = "Cosine Annealing\nrestarts every \(T_i\) epochs"
ax.text(68, 0.085, text_str, fontsize=11, color=line_color,
        verticalalignment='top', horizontalalignment='center', linespacing=1.5)

# 8. 调整边框粗细
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#555555')

# 9. 调整布局并保存
plt.tight_layout()
save_path = 'cosine_annealing_warm_restarts.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 图像已成功生成并保存在当前目录：{save_path}")