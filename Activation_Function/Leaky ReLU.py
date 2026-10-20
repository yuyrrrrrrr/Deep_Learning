import numpy as np
import matplotlib.pyplot as plt

# 1. 设置中文字体，防止中文乱码
# Windows 系统通常使用 'SimHei'，Mac 系统通常使用 'Arial Unicode MS'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 2. 生成数据
x = np.linspace(-6, 6, 1000)
alpha = 0.1  # 设定 Leaky ReLU 负半轴的斜率

# --- 计算 ReLU 及其导数 ---
relu = np.maximum(0, x)
relu_deriv = np.where(x > 0, 1, 0)

# --- 计算 Leaky ReLU 及其导数 ---
leaky_relu = np.where(x > 0, x, alpha * x)
leaky_relu_deriv = np.where(x > 0, 1, alpha)

# 3. 创建画布 (1行2列，尺寸为 12x5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ================= 绘制左图：激活函数对比 =================
# ReLU (蓝色实线)
ax1.plot(x, relu, color='#1f77b4', linewidth=2.5, label='ReLU')
# Leaky ReLU (绿色实线)
ax1.plot(x, leaky_relu, color='#2ca02c', linewidth=2.5, label=f'Leaky ReLU (α={alpha})')

ax1.set_title('激活函数对比: ReLU vs Leaky ReLU', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.axhline(0, color='black', linewidth=1)
ax1.axvline(0, color='black', linewidth=1)

ax1.set_xlim(-6.2, 6.2)
ax1.set_ylim(-1.5, 6.5)
ax1.set_xticks(np.arange(-6, 7, 2))
ax1.set_yticks(np.arange(0, 7, 2))
ax1.legend(loc='upper left', fontsize=12)

# 添加标注，指出负半轴的区别
ax1.annotate('负半轴为 0 (死区)', xy=(-3, 0), xytext=(-5, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             fontsize=11, color='#1f77b4')
ax1.annotate(f'负半轴斜率为 α', xy=(-3, -0.3), xytext=(-5.5, -1.2),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
             fontsize=11, color='#2ca02c')


# ================= 绘制右图：导数对比 =================
# ReLU 导数 (蓝色实线)
ax2.plot(x, relu_deriv, color='#1f77b4', linewidth=2.5, label="ReLU'")
# Leaky ReLU 导数 (绿色实线)
ax2.plot(x, leaky_relu_deriv, color='#2ca02c', linewidth=2.5, label="Leaky ReLU'")

ax2.set_title('导数对比', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.axhline(0, color='black', linewidth=1)
ax2.axvline(0, color='black', linewidth=1)

ax2.set_xlim(-6.2, 6.2)
ax2.set_ylim(-0.2, 1.2)
ax2.set_xticks(np.arange(-6, 7, 2))
# 自定义 y 轴刻度，标出 α 的位置
ax2.set_yticks([0, alpha, 1])
ax2.set_yticklabels(['0', f'α={alpha}', '1'])
ax2.legend(loc='upper right', fontsize=12)

# 4. 调整布局并显示
plt.tight_layout()
plt.show()

# 如果你想保存图片到本地直接拖入 Obsidian，可以取消下面这行的注释：
# plt.savefig('relu_vs_leakyrelu.png', dpi=300, bbox_inches='tight')