import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. 设置中文字体，防止中文显示为方块
# Windows 用户通常使用 'SimHei'，Mac 用户使用 'Arial Unicode MS' 或 'PingFang SC'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 2. 生成正态分布数据 (均值 mu=0, 标准差 sigma=1)
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, loc=0, scale=1)

# 3. 创建画布并设置暗色主题
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#111111')  # 设置画布背景色（外部）
ax.set_facecolor('#111111')         # 设置绘图区背景色（内部）

# 4. 绘制曲线和填充区域
line_color = '#8c52ff'  # 亮紫色
fill_color = '#2a1745'  # 暗紫色
ax.plot(x, y, color=line_color, linewidth=2.5)  # 绘制正态分布曲线
ax.fill_between(x, y, color=fill_color, alpha=0.8)  # 填充曲线下方区域

# 5. 绘制中心虚线 (x=0)
# 虚线只画到曲线的顶点，所以我们获取 y 的最大值作为终点
max_y = np.max(y)
ax.vlines(x=0, ymin=0, ymax=max_y, color='#888888', linestyle='--', linewidth=1.5)

# 6. 设置 X 轴刻度和标签
ax.set_xticks([-1, 0, 1])
ax.set_xticklabels(['\(-\sigma\)', '0', '\(+\sigma\)'], color='#cccccc', fontsize=12)
# 隐藏 Y 轴刻度和标签
ax.set_yticks([])

# 7. 美化边框（坐标轴）
# 隐藏上、左、右边框
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
# 设置底部边框的颜色和粗细
ax.spines['bottom'].set_color('#888888')
ax.spines['bottom'].set_linewidth(1.5)

# 修改刻度线颜色
ax.tick_params(axis='x', colors='#888888')

# 8. 在 X 轴两端添加箭头（还原图片细节）
# 使用 annotate 在 X 轴两端画出箭头
ax.annotate('', xy=(4.3, 0), xytext=(3.8, 0),
            arrowprops=dict(arrowstyle="->", color='#8c52ff', lw=2))
ax.annotate('', xy=(-4.3, 0), xytext=(-3.8, 0),
            arrowprops=dict(arrowstyle="->", color='#8c52ff', lw=2))

# 9. 添加标题和 X 轴标签
ax.set_title('正态分布', color=line_color, fontsize=16, pad=15)
ax.set_xlabel(r'误差 \(\epsilon = y - \hat{y}\)', color='#cccccc', fontsize=13)

# 调整布局
plt.tight_layout()

# 显示图像
plt.show()