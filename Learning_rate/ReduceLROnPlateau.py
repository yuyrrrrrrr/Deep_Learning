import matplotlib

# 强制使用非交互式后端，确保在任何环境（如IDE、终端）都能成功渲染并保存
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FixedFormatter

# 1. 准备数据
# 为了画出垂直下降的阶梯效果，手动构造 X 和 Y 的坐标点
x_points = [0, 20, 20, 45, 45, 70, 70, 100]
y_points = [0.1, 0.1, 0.01, 0.01, 0.001, 0.001, 0.0001, 0.0001]

# 2. 设置画布
fig, ax = plt.subplots(figsize=(8, 5))

# 3. 绘制折线图 (使用特定的蓝色)
line_color = '#3a68b8'
ax.plot(x_points, y_points, color=line_color, linewidth=2)

# 4. 设置 Y 轴为对数坐标系
ax.set_yscale('log')

# 5. 设置坐标轴范围
ax.set_xlim(0, 102)
ax.set_ylim(5e-5, 0.12)

# 6. 设置自定义的 Y 轴刻度 (按照原图格式: 0.100, 0.010, 0.001, 0.0001)
yticks = [1e-1, 1e-2, 1e-3, 1e-4]
yticklabels = ['0.100', '0.010', '0.001', '0.0001']
ax.yaxis.set_major_locator(FixedLocator(yticks))
ax.yaxis.set_major_formatter(FixedFormatter(yticklabels))

# 设置 X 轴刻度
ax.set_xticks(np.arange(0, 101, 20))
ax.tick_params(axis='both', which='major', labelsize=11)

# 7. 设置标题和标签
ax.set_title('ReduceLROnPlateau', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Learning Rate', fontsize=13)

# 8. 添加网格线 (仅保留水平方向的虚线)
ax.grid(True, which='major', axis='y', linestyle='--', alpha=0.4, color='#a0a0a0')
ax.grid(True, which='major', axis='x', linestyle='--', alpha=0.2, color='#d0d0d0')

# 9. 添加箭头和文字标注 "metric plateau"
# 下降点分别在 x=20, 45, 70
drop_points = [20, 45, 70]
# 对应下降前的高度 (设置箭头指向这里)
y_tops = [0.1, 0.01, 0.001]

for x, y in zip(drop_points, y_tops):
    # 添加向下箭头 (xytext 是箭头的尾部，xy 是箭头指向的头部)
    ax.annotate('', xy=(x, y), xytext=(x, y * 2.8),
                arrowprops=dict(facecolor='black', arrowstyle='-|>', lw=1.2))

    # 添加文字说明 (放置在箭头上方)
    ax.text(x, y * 3.2, 'metric\nplateau', ha='center', va='bottom',
            fontsize=9, fontweight='bold', color='#333333')

# 10. 调整边框粗细
for spine in ax.spines.values():
    spine.set_linewidth(1.2)
    spine.set_color('#333333')

# 11. 调整布局并保存
plt.tight_layout()
save_path = 'reduce_lr_on_plateau.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"✅ 图像已成功生成并保存在当前目录：{save_path}")