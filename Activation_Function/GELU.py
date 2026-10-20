import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. 设置中文字体，防止中文乱码
# Windows 系统通常使用 'SimHei'，Mac 系统通常使用 'Arial Unicode MS'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 2. 生成数据
x = np.linspace(-6, 6, 1000)

# --- 计算 ReLU 及其导数 ---
relu = np.maximum(0, x)
relu_deriv = np.where(x > 0, 1, 0)

# --- 计算 GELU 及其导数 ---
# GELU(x) = x * \Phi(x)，\Phi(x) 是标准正态分布的 CDF
gelu = x * norm.cdf(x)
# GELU 的导数: f'(x) = \Phi(x) + x * \phi(x)
# 其中 \phi(x) 是标准正态分布的 PDF
gelu_deriv = norm.cdf(x) + x * norm.pdf(x)

# 3. 创建画布 (1行2列，尺寸为 12x5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ================= 绘制左图：激活函数对比 =================
ax1.plot(x, relu, color='#1f77b4', linewidth=2.5, label='ReLU')
ax1.plot(x, gelu, color='#ff7f0e', linewidth=2.5, label='GELU')

ax1.set_title('激活函数对比: ReLU vs GELU', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.axhline(0, color='black', linewidth=1)
ax1.axvline(0, color='black', linewidth=1)

ax1.set_xlim(-6.2, 6.2)
ax1.set_ylim(-1.5, 6.5)
ax1.set_xticks(np.arange(-6, 7, 2))
ax1.set_yticks(np.arange(0, 7, 2))
ax1.legend(loc='upper left', fontsize=12)

# 添加标注：展示 ReLU 的硬性切断 与 GELU 的柔和软门
ax1.annotate('ReLU 硬性切断 (不可导)', xy=(0, 0), xytext=(-5.5, 2.5),
             arrowprops=dict(facecolor='#1f77b4', shrink=0.05, width=1, headwidth=6),
             fontsize=11, color='#1f77b4')
ax1.annotate('GELU 平滑过渡 (软门)', xy=(0, 0), xytext=(1.5, -1),
             arrowprops=dict(facecolor='#ff7f0e', shrink=0.05, width=1, headwidth=6),
             fontsize=11, color='#ff7f0e')


# ================= 绘制右图：导数对比 =================
ax2.plot(x, relu_deriv, color='#1f77b4', linewidth=2.5, label="ReLU'")
ax2.plot(x, gelu_deriv, color='#ff7f0e', linewidth=2.5, label="GELU'")

ax2.set_title('导数对比 (梯度行为)', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.axhline(0, color='black', linewidth=1)
ax2.axvline(0, color='black', linewidth=1)

ax2.set_xlim(-6.2, 6.2)
ax2.set_ylim(-0.2, 1.2)
ax2.set_xticks(np.arange(-6, 7, 2))
ax2.set_yticks([0, 0.5, 1])
ax2.legend(loc='upper right', fontsize=12)

# 4. 调整布局并显示
plt.tight_layout()
plt.show()

# 如果你想把图片保存下来插入到 Obsidian 中，可以取消下面这行的注释：
# plt.savefig('gelu_vs_relu.png', dpi=300, bbox_inches='tight')