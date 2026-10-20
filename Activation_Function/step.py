import numpy as np
import matplotlib.pyplot as plt

# 1. 设置中文字体，防止中文乱码
# Windows 系统通常使用 'SimHei'，Mac 系统通常使用 'Arial Unicode MS'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 2. 生成数据
x = np.linspace(-6, 6, 1000)

# --- 阶跃函数 ---
# 当 x >= 0 时输出 1，否则输出 0
step = np.where(x >= 0, 1, 0)

# 3. 创建画布 (1行2列，尺寸为 12x4)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# --- 绘制左图：阶跃函数 ---
ax1.plot(x, step, color='#1f77b4', linewidth=2.5)
ax1.set_title('阶跃函数 Step(x)')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.axhline(0, color='black', linewidth=1)
ax1.axvline(0, color='black', linewidth=1)

# 绘制跳跃点：在 x=0 处，y=0 画空心圆，y=1 画实心圆
ax1.scatter(0, 0, facecolors='white', edgecolors='#1f77b4', s=80, zorder=5, linewidth=2)
ax1.scatter(0, 1, color='#1f77b4', s=80, zorder=5)

# 设置坐标轴范围
ax1.set_xlim(-6.2, 6.2)
ax1.set_ylim(-0.2, 1.2)
ax1.set_xticks(np.arange(-6, 7, 2))
ax1.set_yticks([0, 1])

# --- 绘制右图：导数 (冲激函数) ---
# 在 x != 0 的地方，导数全部为 0
y_deriv = np.zeros_like(x)
ax2.plot(x, y_deriv, color='#d62728', linewidth=2.5, label="恒为 0")

# 在 x=0 处画一条竖线表示无穷大的脉冲 (Dirac delta)
ax2.vlines(0, 0, 1, color='#d62728', linewidth=2.5, linestyles='--')
ax2.scatter(0, 1, color='#d62728', marker='^', s=80, zorder=5) # 用向上的三角形表示冲激

ax2.set_title("导数 Step'(x) (狄拉克 δ 函数)")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.axhline(0, color='black', linewidth=1)
ax2.axvline(0, color='black', linewidth=1)

# 设置坐标轴范围
ax2.set_xlim(-6.2, 6.2)
ax2.set_ylim(-0.2, 1.2)
ax2.set_xticks(np.arange(-6, 7, 2))
ax2.set_yticks([0, 1])

# 添加文本标注
ax2.text(0.5, 0.9, "在 x=0 处趋于无穷", color='#d62728', fontsize=11)
ax2.text(-5, 0.1, "x ≠ 0 时均为 0", color='#d62728', fontsize=11)

# 4. 调整布局并显示
plt.tight_layout()
plt.show()

# 如果你想保存图片到本地直接拖入 Obsidian，可以取消下面这行的注释：
# plt.savefig('step_and_derivative.png', dpi=300, bbox_inches='tight')