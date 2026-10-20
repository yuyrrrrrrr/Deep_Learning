import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. 设置中文字体，防止中文乱码
# Windows 系统通常使用 'SimHei'，Mac 系统通常使用 'Arial Unicode MS'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 2. 生成数据
x = np.linspace(-6, 6, 1000)

# --- 计算 SiLU 及其导数 ---
sigmoid = 1 / (1 + np.exp(-x))
silu = x * sigmoid
silu_deriv = sigmoid + x * sigmoid * (1 - sigmoid)

# --- 计算 GELU 及其导数 ---
# GELU(x) = x * \Phi(x)，\Phi(x) 是标准正态分布的 CDF
gelu = x * norm.cdf(x)
# GELU 的导数: f'(x) = \Phi(x) + x * \phi(x)
gelu_deriv = norm.cdf(x) + x * norm.pdf(x)

# 3. 创建画布 (1行2列，尺寸为 12x5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ================= 绘制左图：激活函数对比 =================
ax1.plot(x, silu, color='#9467bd', linewidth=2.5, label='SiLU (Swish)')
ax1.plot(x, gelu, color='#ff7f0e', linewidth=2.5, label='GELU')

ax1.set_title('激活函数对比: SiLU vs GELU', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.axhline(0, color='black', linewidth=1)
ax1.axvline(0, color='black', linewidth=1)

ax1.set_xlim(-6.2, 6.2)
ax1.set_ylim(-1.5, 6.5)
ax1.set_xticks(np.arange(-6, 7, 2))
ax1.set_yticks(np.arange(-1, 7, 2))
ax1.legend(loc='upper left', fontsize=12)

# 添加标注：指出两者在负半轴都有微小的“下探”
ax1.annotate('SiLU 极小值点 (约 -0.28)', xy=(-1.28, -0.28), xytext=(-5.5, -1.2),
             arrowprops=dict(facecolor='#9467bd', shrink=0.05, width=1, headwidth=6),
             fontsize=11, color='#9467bd')
ax1.annotate('GELU 极小值点 (约 -0.17)', xy=(-0.75, -0.17), xytext=(1.5, -1.2),
             arrowprops=dict(facecolor='#ff7f0e', shrink=0.05, width=1, headwidth=6),
             fontsize=11, color='#ff7f0e')

# ================= 绘制右图：导数对比 =================
ax2.plot(x, silu_deriv, color='#9467bd', linewidth=2.5, label="SiLU'")
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

# 添加标注：指出两者在 x=0 处的截距不同
ax2.scatter(0, 0.5, color='#9467bd', s=50, zorder=5)
ax2.scatter(0, 0.5, color='#ff7f0e', s=50, zorder=5)
ax2.annotate('在 x=0 处导数均为 0.5', xy=(0, 0.5), xytext=(1.5, 0.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             fontsize=11, color='black')

# 4. 调整布局并显示
plt.tight_layout()
plt.show()

# 如果你想保存图片到本地直接拖入 Obsidian，可以取消下面这行的注释：
# plt.savefig('silu_vs_gelu.png', dpi=300, bbox_inches='tight')