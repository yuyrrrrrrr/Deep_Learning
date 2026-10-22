import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

# 1. 设置中文字体和暗色主题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# 设置整体画布背景色（深灰/黑色）
fig = plt.figure(figsize=(14, 7.5), facecolor='#0e0e0e')

# 2. 生成数据
# 注意：p 不能等于 0，否则 log(0) 会报错。我们取一个极小值开始。
p = np.linspace(0.001, 1, 1000)
y1 = 1 - p
y2 = -np.log(p)

# ==========================================
# 3. 绘制主图 (左侧)
# ==========================================
# 添加坐标系 [左, 下, 宽, 高]
ax1 = fig.add_axes([0.05, 0.25, 0.55, 0.65])
ax1.set_facecolor('#0e0e0e')  # 图表内部背景色

# 绘制线条
ax1.plot(p, y1, label='1 - p', color='#5b8cff', linewidth=2.5)
ax1.plot(p, y2, label='-log(p) (自然对数)', color='#ff9430', linewidth=2.5)

# 设置标题、坐标轴标签
ax1.set_title('1 - p 与 -log(p) 的图像对比', color='white', fontsize=16, pad=15)
ax1.set_xlabel('p (概率)', color='white', fontsize=12)
ax1.set_ylabel('数值', color='white', fontsize=12)

# 设置坐标轴范围和刻度
ax1.set_xlim(0, 1.05)
ax1.set_ylim(0, 6.5)
ax1.set_xticks(np.arange(0, 1.1, 0.2))
ax1.set_yticks(np.arange(0, 7, 1))

# 设置刻度标签颜色
ax1.tick_params(axis='both', colors='white', labelsize=11)

# 设置网格线（虚线，低透明度）
ax1.grid(True, linestyle='--', color='#333333', alpha=0.8)

# 设置边框颜色
for spine in ax1.spines.values():
    spine.set_color('#333333')

# 图例
legend = ax1.legend(loc='upper right', facecolor='#1c1c1c', edgecolor='#444444', fontsize=11)
for text in legend.get_texts():
    text.set_color('white')

# 添加文字注释
# 注释 1：左上角
ax1.annotate('p → 0⁺ 时，\n-log(p) → +∞', xy=(0.02, 5.5), xytext=(0.1, 5.5),
             color='white', fontsize=11,
             arrowprops=dict(arrowstyle="->", color='#ff9430'))

# 注释 2：右下角
ax1.annotate('p = 1 时，\n两者都为 0', xy=(1.0, 0), xytext=(0.75, 1.0),
             color='white', fontsize=11,
             arrowprops=dict(arrowstyle="->", color='white'))


# ==========================================
# 4. 绘制表格 (右侧)
# ==========================================
ax2 = fig.add_axes([0.68, 0.35, 0.28, 0.45])
ax2.set_facecolor('#0e0e0e')
ax2.axis('off') # 关闭坐标轴

# 准备表格数据
p_vals = [0.001, 0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 0.999]
data = []
for val in p_vals:
    one_minus_p = 1 - val
    neg_log_p = -np.log(val)
    # 格式化保留 3 位小数
    data.append([f"{val:g}", f"{one_minus_p:.3f}", f"{neg_log_p:.3f}"])

# 列名
columns = ['p', '1 - p', '-log(p)']

# 创建表格
table = ax2.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')

# 美化表格
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.8) # 调整行高

# 设置表格颜色样式（模拟暗黑模式）
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#333333') # 边框颜色
    if row == 0: # 表头
        cell.set_text_props(color='white', weight='bold')
        cell.set_facecolor('#1c1c1c')
    else: # 数据行
        cell.set_text_props(color='#e0e0e0')
        cell.set_facecolor('#0e0e0e')


# ==========================================
# 5. 绘制底部文本框
# ==========================================
# 左侧文本框
ax_text1 = fig.add_axes([0.05, 0.02, 0.45, 0.12])
ax_text1.set_facecolor('#161616')
ax_text1.set_xticks([])
ax_text1.set_yticks([])
for spine in ax_text1.spines.values():
    spine.set_color('#333333')
ax_text1.text(0.02, 0.7, "• 1 - p 是直线，从 (0, 1) 线性下降到 (1, 0)", color='white', fontsize=11, va='center')
ax_text1.text(0.02, 0.3, "• -log(p) 在 p → 0⁺ 时趋于 +∞，下降较快，随后逐渐趋近于 0", color='white', fontsize=11, va='center')

# 右侧文本框
ax_text2 = fig.add_axes([0.52, 0.02, 0.44, 0.12])
ax_text2.set_facecolor('#161616')
ax_text2.set_xticks([])
ax_text2.set_yticks([])
for spine in ax_text2.spines.values():
    spine.set_color('#333333')
ax_text2.text(0.02, 0.7, "→ 在小概率区域（p 很小），-log(p) 远大于 1 - p，", color='white', fontsize=11, va='center')
ax_text2.text(0.05, 0.3, "因此 -log(p) 惩罚小概率更强", color='white', fontsize=11, va='center')


# 显示图像
plt.show()