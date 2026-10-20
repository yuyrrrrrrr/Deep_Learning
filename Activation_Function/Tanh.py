"""sigmoid函数的保证 Tanh"""
import numpy as np
import matplotlib.pyplot as plt

# 1. 设置中文字体，防止标题中文乱码
# Windows 系统通常使用 'SimHei'，Mac 系统通常使用 'Arial Unicode MS'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 2. 生成数据
# 生成 x 轴数据，范围从 -6 到 6
x = np.linspace(-6, 6, 1000)

# --- 计算函数值 ---
# Sigmoid: 1 / (1 + e^-x)
sigmoid = 1 / (1 + np.exp(-x))
# Tanh: 直接使用 numpy 内置函数
tanh = np.tanh(x)

# --- 计算导数值 ---
# Sigmoid 导数: f(x) * (1 - f(x))
sigmoid_deriv = sigmoid * (1 - sigmoid)
# Tanh 导数: 1 - f(x)^2
tanh_deriv = 1 - tanh**2

# 3. 创建画布 (1行2列，尺寸为 12x4)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# ----------------- 绘制左图：激活函数对比 -----------------
# 设置颜色，近似原图的浅红色和浅金色
color_sigmoid = '#e67b7b'
color_tanh = '#e6c85c'

ax1.plot(x, sigmoid, color=color_sigmoid, linewidth=2, label='Sigmoid')
ax1.plot(x, tanh, color=color_tanh, linewidth=2, label='Tanh')
ax1.set_title('激活函数 f(x)')
ax1.grid(True, linestyle='-', alpha=0.4)
ax1.legend(loc='upper left', fontsize='small')
# 设置坐标轴范围和刻度 (还原原图)
ax1.set_xlim(-6.2, 6.2)
ax1.set_ylim(-1.05, 1.05)
ax1.set_xticks(np.arange(-6, 7, 2))
ax1.set_yticks(np.arange(-1.0, 1.25, 0.25))

# ----------------- 绘制右图：导数对比 -----------------
ax2.plot(x, sigmoid_deriv, color=color_sigmoid, linewidth=2, label='Sigmoid')
ax2.plot(x, tanh_deriv, color=color_tanh, linewidth=2, label='Tanh')
# 原图标题写的是 f(x) 不是 f'(x)，这里为了贴合原图写作 f(x)
ax2.set_title("导数 f(x)")
ax2.grid(True, linestyle='-', alpha=0.4)
ax2.legend(loc='upper right', fontsize='small')
# 设置坐标轴范围和刻度
ax2.set_xlim(-6.2, 6.2)
ax2.set_ylim(-0.05, 1.05)
ax2.set_xticks(np.arange(-6, 7, 2))
ax2.set_yticks(np.arange(0.0, 1.2, 0.2))

# 4. 调整布局并显示
plt.tight_layout()
plt.show()

# 如果你想保存图片到本地直接拖入 Obsidian，可以取消下面这行的注释：
# plt.savefig('sigmoid_vs_tanh.png', dpi=300, bbox_inches='tight')