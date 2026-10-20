import numpy as np
import matplotlib.pyplot as plt

# 1. 设置中文字体，防止中文乱码
# Windows 系统通常使用 'SimHei'，Mac 系统通常使用 'Arial Unicode MS'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 2. 生成数据
# 生成 x 轴数据，范围从 -6 到 6
x = np.linspace(-6, 6, 1000)

# --- 计算 ReLU 函数值 ---
# ReLU(x) = max(0, x)
relu = np.maximum(0, x)

# --- 计算 ReLU 导数值 ---
# 导数：x > 0 时为 1，x <= 0 时为 0 (在 x=0 处不可导，通常定义为其左/右极限)
relu_deriv = np.where(x > 0, 1, 0)

# 3. 创建画布 (1行2列，尺寸为 12x4)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# --- 绘制左图：ReLU 函数 ---
# 使用绿色线条
ax1.plot(x, relu, color='#2ca02c', linewidth=2.5)
ax1.set_title('激活函数 ReLU(x) = max(0, x)', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)
# 绘制坐标轴的黑线
ax1.axhline(0, color='black', linewidth=1)
ax1.axvline(0, color='black', linewidth=1)
# 设置坐标轴范围 (留出适当边距)
ax1.set_xlim(-6.2, 6.2)
ax1.set_ylim(-1.5, 6.5)
ax1.set_xticks(np.arange(-6, 7, 2))
ax1.set_yticks(np.arange(0, 7, 2))

# --- 绘制右图：ReLU 导数 ---
# 使用紫红色线条
ax2.plot(x, relu_deriv, color='#9467bd', linewidth=2.5)
ax2.set_title("导数 ReLU'(x)", fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.6)
# 绘制坐标轴的黑线
ax2.axhline(0, color='black', linewidth=1)
ax2.axvline(0, color='black', linewidth=1)
# 设置坐标轴范围 (导数只有 0 和 1，限制 y 轴范围让图像更清晰)
ax2.set_xlim(-6.2, 6.2)
ax2.set_ylim(-0.2, 1.2)
ax2.set_xticks(np.arange(-6, 7, 2))
ax2.set_yticks([0, 1])

# 4. 调整布局并显示
plt.tight_layout()
plt.show()

# 如果你想把图片保存下来插入到 Obsidian 中，可以取消下面这行的注释：
# plt.savefig('relu_and_derivative.png', dpi=300, bbox_inches='tight')