"""sigmoid函数和它的导数图象"""
import numpy as np
import matplotlib.pyplot as plt

# 1. 设置中文字体，防止中文显示为方块
# Windows 系统通常使用 'SimHei'，Mac 系统通常使用 'Arial Unicode MS'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 2. 生成数据
# 生成 x 轴数据，范围从 -6 到 6
x = np.linspace(-6, 6, 500)

# 计算 Sigmoid 函数
sigmoid = 1 / (1 + np.exp(-x))

# 计算 Sigmoid 的导数 (f(x) * (1 - f(x)))
sigmoid_derivative = sigmoid * (1 - sigmoid)

# 3. 创建画布 (1行2列，尺寸为 12x4)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# ----------------- 绘制左图：激活函数 f(x) -----------------
ax1.plot(x, sigmoid, color='indianred', linewidth=2, label='Sigmoid')
ax1.set_title('激活函数 f(x)')
ax1.grid(True, linestyle='-', alpha=0.6)
ax1.legend(loc='upper left')
# 设置坐标轴范围，让图像更贴近原图
ax1.set_xlim(-6.2, 6.2)
ax1.set_ylim(-0.05, 1.05)

# ----------------- 绘制右图：导数 f'(x) -----------------
ax2.plot(x, sigmoid_derivative, color='indianred', linewidth=2, label='Sigmoid')
ax2.set_title("导数 f'(x)")
ax2.grid(True, linestyle='-', alpha=0.6)
ax2.legend(loc='upper right')
# 设置坐标轴范围
ax2.set_xlim(-6.2, 6.2)
ax2.set_ylim(-0.01, 0.26)

# 4. 调整布局并显示
plt.tight_layout()
plt.show()

# 如果你想把图片保存下来插入到 Obsidian 中，可以取消下面这行的注释：
# plt.savefig('sigmoid_and_derivative.png', dpi=300, bbox_inches='tight')

