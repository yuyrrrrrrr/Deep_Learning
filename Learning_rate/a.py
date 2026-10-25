import matplotlib

# 强制使用非交互式后端，确保在任何环境都能成功渲染并保存
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 全局设置：确保数学公式正常渲染
# ==========================================
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'  # 使用最标准的 Computer Modern 字体

# 1. 准备数据
epochs = np.linspace(0, 100, 500)
eta_max = 1.0
eta_min = 0.0

# 策略 1: Exponential Decay
gamma = 0.95
lr_exp = eta_max * (gamma ** epochs)

# 策略 2: Cosine Annealing
lr_cos = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * epochs / 100))

# 策略 3: Cosine Annealing with Warm Restarts
lr_warm = np.zeros_like(epochs)
T_i = 20
for i in range(5):
    start = i * T_i
    end = (i + 1) * T_i
    mask = (epochs >= start) & (epochs <= end)
    lr_warm[mask] = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * (epochs[mask] - start) / T_i))

# 2. 设置画布
fig, ax = plt.subplots(figsize=(14, 7))

# 颜色定义
color_exp = '#4A72B0'
color_cos = '#E58757'
color_warm = '#5C9E6C'

# 3. 绘制三条曲线
ax.plot(epochs, lr_exp, color=color_exp, linewidth=2.5, label='Exponential Decay')
ax.plot(epochs, lr_cos, color=color_cos, linewidth=2.5, label='Cosine Annealing')
ax.plot(epochs, lr_warm, color=color_warm, linewidth=2.5, label='Cosine Annealing\nwith Warm Restarts')

# 4. 设置坐标轴
ax.set_xlim(0, 100)
ax.set_ylim(-0.05, 1.08)
ax.set_xticks(np.arange(0, 101, 20))
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis='both', which='major', labelsize=12)

# 5. 添加标题和标签 (使用 $ 包裹 η)
ax.set_title('Learning Rate Schedules Comparison', fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Epoch', fontsize=14, labelpad=30)
ax.set_ylabel(r'Learning Rate ($\eta$)', fontsize=14)

# 6. 添加网格线和垂直边界线
ax.grid(True, which='major', linestyle='--', alpha=0.4, color='#a0a0a0')
for x in [20, 40, 60, 80]:
    ax.axvline(x=x, color='#a0a0a0', linestyle='--', linewidth=1, alpha=0.8)

# 7. 添加公式注释 (全部使用 $ 包围，并采用 r"" 避免转义)
bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="none", alpha=0.85)

# 指数衰减注释
text_exp = "Exponential Decay\n" + r"$\eta_t = \eta_0 \cdot \gamma^t$" + "\n" + r"$(0 < \gamma < 1)$"
ax.annotate(text_exp,
            xy=(15, 0.45), xytext=(10, 0.85),
            color=color_exp, fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle="-|>", color=color_exp, alpha=0.7),
            bbox=bbox_props, ha='center')

# 余弦退火注释
text_cos = "Cosine Annealing\n" + \
           r"$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})$" + "\n" + \
           r"$\times (1 + \cos(\frac{T_{cur}}{T_{max}}\pi))$"
ax.annotate(text_cos,
            xy=(45, 0.72), xytext=(30, 0.85),
            color=color_cos, fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle="-|>", color=color_cos, alpha=0.7),
            bbox=bbox_props, ha='center')

# 热重启余弦退火注释 (注意 T_i 和 i 的处理)
text_warm = "Cosine Annealing\nwith Warm Restarts\n" + \
            r"$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})$" + "\n" + \
            r"$\times (1 + \cos(\frac{T_{cur}}{T_i}\pi))$" + "\n" + \
            r"$T_i$: length of $i$-th cycle"
ax.annotate(text_warm,
            xy=(75, 0.6), xytext=(75, 0.8),
            color=color_warm, fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle="-|>", color=color_warm, alpha=0.7),
            bbox=bbox_props, ha='center')

# 8. 添加图例
legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.9, edgecolor='#555555')
legend.get_frame().set_linewidth(1.2)

# 9. 调整边框
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#333333')

# 10. 顶部 η_max 标记 (严格使用 $ 包裹)
ax.text(-2, 1.02, r'$\eta_{max}$', fontsize=12, ha='right')

# ==========================================
# 11. 绘制底部的周期标记条
# ==========================================
y_bar = -0.1
for i in range(5):
    start = i * 20
    end = (i + 1) * 20
    mid = (start + end) / 2

    ax.plot([start, end], [y_bar, y_bar], color=color_warm, linestyle='--', linewidth=1.5,
            marker='*', markersize=8, clip_on=False)

    # 文字中的公式也严格用 $ 包裹
    ax.text(mid, y_bar - 0.05, r"Cycle " + f"{i + 1} " + r"($T_{" + f"{i + 1}" + r"} = 20$)",
            color=color_warm, fontsize=11, ha='center', clip_on=False)

# 12. 保存
plt.tight_layout()
save_path = 'lr_schedules_comparison_fixed.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 图像已成功生成并保存在当前目录：{save_path}")