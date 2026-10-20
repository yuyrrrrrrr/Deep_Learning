import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# 1. 设置中文字体，防止中文乱码
# Windows 系统通常使用 'SimHei'，Mac 系统通常使用 'Arial Unicode MS'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 2. 创建画布 (设置大一点，以便下方放文本框)
fig, ax = plt.subplots(figsize=(14, 8))

# ----------------- 绘制主坐标轴 -----------------
# 隐藏右侧和上方的边框
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# 将左侧和下侧边框移动到原点
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
# 添加箭头
ax.plot((1), (0), ls="", marker=">", ms=8, color="k", transform=ax.get_yaxis_transform(), clip_on=False)
ax.plot((0), (1), ls="", marker="^", ms=8, color="k", transform=ax.get_xaxis_transform(), clip_on=False)
ax.set_xlabel('\(x\)', fontsize=16, loc='right')
ax.set_ylabel('\(y\)', fontsize=16, loc='top', rotation=0)

# 设置坐标轴范围
ax.set_xlim(-4.5, 4.5)
ax.set_ylim(-1.5, 9.5)

# ----------------- 绘制曲线与切线 -----------------
x = np.linspace(-3.5, 2.8, 500)
y = x**2
ax.plot(x, y, color='#002b80', linewidth=2.5) # 绘制 y = x^2

# 绘制切线
# 1. 红色切线 at x = -3
x_tan_red = np.linspace(-3.8, -1.8, 50)
y_tan_red = -6 * x_tan_red - 9
ax.plot(x_tan_red, y_tan_red, color='#b22222', linewidth=2)

# 2. 绿色切线 at x = 0 (就是 x 轴)
ax.plot([-3.5, 3.5], [0, 0], color='#2ca02c', linewidth=2)

# 3. 绿色切线 at x = 2
x_tan_green = np.linspace(0.5, 3.5, 50)
y_tan_green = 4 * x_tan_green - 4
ax.plot(x_tan_green, y_tan_green, color='#2ca02c', linewidth=2)

# ----------------- 绘制散点和垂直虚线 -----------------
points = [(-3, 9), (0, 0), (2, 4)]
for px, py in points:
    ax.scatter(px, py, color='#104e8b', s=60, zorder=5)
    ax.vlines(px, 0, py, color='gray', linestyle='--', alpha=0.7)

# ----------------- 添加文字标注 -----------------
# 坐标点标注
ax.text(-3.2, 9.2, '(-3, 9)', fontsize=12, color='#002b80')
ax.text(0.2, -0.5, '(0, 0)', fontsize=12, color='#002b80')
ax.text(2.2, 4.2, '(2, 4)', fontsize=12, color='#002b80')

# 曲线标注
ax.text(2.5, 7.5, '\(y = x^2\)', fontsize=14, color='#002b80')

# 切线方程与斜率标注
ax.text(-4.2, 6.5, '切线: \(y = -6x - 9\)\n斜率 \(f\'(-3) = -6\)', fontsize=12, color='#b22222')
ax.text(1.5, 1.5, '切线: \(y = 4x - 4\)\n斜率 \(f\'(2) = 4\)', fontsize=12, color='#2ca02c')
ax.text(0.2, -2.2, '切线: \(y = 0\)\n斜率 \(f\'(0) = 0\)', fontsize=12, color='#2ca02c')

# 左上角公式框
box_text = '\(f(x) = x^2\)\n\(f\'(x) = 2x\)'
bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="#002b80", lw=1.5)
ax.text(-4.2, 8.5, box_text, fontsize=14, bbox=bbox_props, color='#002b80')

# 在轴上标出 -3 和 2
ax.text(-3, -0.4, '-3', fontsize=12, ha='center', color='#b22222')
ax.text(2, -0.4, '2', fontsize=12, ha='center', color='#2ca02c')

# ----------------- 绘制底部的说明框 -----------------
# 为了在图表外绘制，我们使用 fig.text
box_texts = [
    ("● 在 \(x = -3\) 处, \(f'(x) = -6 < 0\)：\n   沿 \(x\) 增大方向函数在**下降**,\n   反方向才是上升", '#fdf0ef', '#b22222'), # 红框
    ("● 在 \(x = 0\) 处, \(f'(x) = 0\)：\n   既不上也不下——\n   这就是**最低点**", '#f0f8ef', '#2ca02c'), # 绿框
    ("● 在 \(x = 2\) 处, \(f'(x) = 4 > 0\)：\n   沿 \(x\) 增大方向函数在**上升**,\n   斜率为 4", '#eff4fa', '#104e8b') # 蓝框
]

for i, (text, facecolor, edgecolor) in enumerate(box_texts):
    # 计算每个框的水平位置
    x_pos = 0.05 + i * 0.32
    # 绘制背景框
    fig.patches.append(FancyBboxPatch((x_pos, -0.05), 0.28, 0.12,
                                        boxstyle="round,pad=0.02,rounding_size=0.02",
                                        transform=fig.transFigure,
                                        facecolor=facecolor, edgecolor=edgecolor, linewidth=2,
                                        zorder=-1))
    # 添加文字 (简单的文本替换 ** 为真实字体加粗，这里用 matplotlib 不好直接渲染 markdown 的加粗，直接去掉星号)
    fig.text(x_pos + 0.02, 0.04, text.replace('**', ''), fontsize=12, color='black', va='center')

# 4. 调整布局并显示
plt.subplots_adjust(bottom=0.25) # 留出底部空间给说明框
plt.show()

# 如果你想保存图片，取消下面这行的注释：
# plt.savefig('gradient_descent_example.png', dpi=300, bbox_inches='tight')