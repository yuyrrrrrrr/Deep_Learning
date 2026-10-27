import matplotlib

# 强制使用非交互式后端
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

# 设置随机种子，确保每次生成的数据形状一致
np.random.seed(42)

# ==========================================
# 1. 模拟生成数据
# ==========================================
x = np.linspace(0, 6.5, 300)

# --- 左图: Training Error ---
# 20-layer (黄绿色)：平滑下降，中间突降
y_train_20 = 10 + 10 * np.exp(-x / 1.5) + np.random.normal(0, 0.3, size=len(x))
# 模拟在 3.2 处的断崖式下降
y_train_20[x > 3.2] -= 6
y_train_20 = np.clip(y_train_20, 1.5, 20)

# 56-layer (深红色)：从 iter=2 开始，前期剧烈震荡，中间突降
y_train_56 = np.full_like(x, np.nan)
mask_56 = x >= 2.0
# 基础下降趋势 + 剧烈震荡
base_56 = 14 + 6 * np.exp(-(x[mask_56] - 2) / 1.5)
noise_56 = np.random.normal(0, 0.8, size=len(x[mask_56]))
y_train_56[mask_56] = base_56 + noise_56
# 模拟在 3.2 处的断崖式下降
y_train_56[x > 3.2] -= 3.5
y_train_56 = np.clip(y_train_56, 5, 20)

# --- 右图: Test Error ---
# 20-layer：平滑下降，没有明显的断崖，最终 ~9%
y_test_20 = 9 + 11 * np.exp(-x / 1.8) + np.random.normal(0, 0.2, size=len(x))
y_test_20 = np.clip(y_test_20, 9, 20)

# 56-layer：震荡，且在 3.2 处有一个明显的台阶下降，最终平稳在 ~12%
y_test_56 = np.full_like(x, np.nan)
base_test_56 = 15 + 5 * np.exp(-(x[mask_56] - 2) / 1.5)
noise_test_56 = np.random.normal(0, 0.6, size=len(x[mask_56]))
y_test_56[mask_56] = base_test_56 + noise_test_56
y_test_56[x > 3.2] -= 2.5  # 台阶下降
y_test_56 = np.clip(y_test_56, 12, 20)

# ==========================================
# 2. 绘图设置
# ==========================================
# 颜色提取
color_56 = '#b31b1b'  # 深红色
color_20 = '#a3b52c'  # 黄绿色

# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))


# 定义通用的坐标轴格式化函数
def format_axis(ax, ylabel):
    # 设置坐标轴范围
    ax.set_xlim(0, 6.5)
    ax.set_ylim(0, 20.5)
    # 设置刻度
    ax.set_xticks(np.arange(0, 7, 1))
    ax.set_yticks([0, 10, 20])
    # 设置标签字体大小
    ax.set_xlabel('iter. (1e4)', fontsize=16, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', labelsize=13)

    # 隐藏上方和右侧的边框线
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # 加粗左侧和下方边框
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')


# ==========================================
# 3. 绘制左图 (Training Error)
# ==========================================
ax1.plot(x, y_train_20, color=color_20, linewidth=2.5)
ax1.plot(x, y_train_56, color=color_56, linewidth=2.5)

format_axis(ax1, 'training error (%)')
# 添加曲线标注
ax1.text(5.2, 5.5, '56-layer', fontsize=15, fontweight='bold', color='black')
ax1.text(5.2, 2.0, '20-layer', fontsize=15, fontweight='bold', color='black')

# ==========================================
# 4. 绘制右图 (Test Error)
# ==========================================
ax2.plot(x, y_test_20, color=color_20, linewidth=2.5)
ax2.plot(x, y_test_56, color=color_56, linewidth=2.5)

format_axis(ax2, 'test error (%)')
# 添加曲线标注
ax2.text(5.2, 13.0, '56-layer', fontsize=15, fontweight='bold', color='black')
ax2.text(5.2, 9.5, '20-layer', fontsize=15, fontweight='bold', color='black')

# ==========================================
# 5. 保存并展示
# ==========================================
plt.tight_layout()
save_path = 'resnet_degradation_plot.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 图像已成功生成并保存在当前目录：{save_path}")