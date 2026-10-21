import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.patheffects as path_effects

# 1. 设置中文字体，防止乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ================= 2. 构造曲面与轨迹数据 =================
# 构造一个形似原图的复杂曲面：左侧高峰 -> 鞍点 -> 右侧平坦低谷
def f(x, y):
    # 主峰 (左上方)
    peak = 4.0 * np.exp(-((x + 2.5) ** 2 / 2.5 + (y - 1.5) ** 2 / 2.5))
    # 基础谷底倾斜面
    base = -0.3 * x + 0.1 * y
    # 右侧微凸起以形成缓坡
    bump = 0.8 * np.exp(-((x - 2.0) ** 2 / 3.0 + (y + 0.5) ** 2 / 2.0))
    return peak + base + bump


# 生成网格数据
x_val = np.linspace(-5, 5, 100)
y_val = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_val, y_val)
Z = f(X, Y)

# 模拟梯度下降轨迹（精确匹配原图的走向）
trace_x = np.array([-3.2, -2.8, -2.3, -1.8, -1.2, -0.5, 0.2, 1.0, 1.8, 2.5, 3.2, 3.8])
trace_y = np.array([1.8, 1.4, 0.9, 0.4, 0.0, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1])
trace_z = f(trace_x, trace_y)

# ================= 3. 创建画布 =================
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#ffffff')

# ================= 4. 绘制上方 3D 曲面 =================
# 使用绝对定位放置 3D 图 (左, 下, 宽, 高)
ax3d = fig.add_axes([0.05, 0.4, 0.9, 0.55], projection='3d')

# 绘制曲面，使用与原图相似的色带
surf = ax3d.plot_surface(X, Y, Z, cmap='turbo', rstride=2, cstride=2, alpha=0.9, linewidth=0.1, edgecolors='gray')

# 绘制 3D 轨迹线
ax3d.plot(trace_x, trace_y, trace_z, color='black', marker='o', markersize=6, linestyle='-', linewidth=2, zorder=10)

# 设置 3D 视角 (仰角, 方位角)，完美还原原图视角
ax3d.view_init(elev=25, azim=-60)

# 隐藏默认的坐标轴背景，自定义坐标轴
ax3d.set_axis_off()
# 手动绘制坐标轴箭头
ax3d.quiver(-5, -3, -1, 0, 6, 0, color='k', arrow_length_ratio=0.1, linewidth=1.5)  # y轴
ax3d.quiver(-5, -3, -1, 10, 0, 0, color='k', arrow_length_ratio=0.1, linewidth=1.5)  # x轴 (斜向)
ax3d.quiver(-5, -3, -1, 0, 0, 5, color='k', arrow_length_ratio=0.1, linewidth=1.5)  # z轴 (向上)

# 添加 3D 文字标注
ax3d.text(-5, -3, 4.5, '\(f(x, y)\)', fontsize=16, fontweight='bold')
ax3d.text(-1, -3.5, -1.5, '\(x\)', fontsize=14)
ax3d.text(3, -1, -1.5, '\(y\)', fontsize=14)

# 添加原图中的文字标注和指向线
ax3d.text(-0.5, 1.5, 3.5, '鞍点 / 梯度很小区域', fontsize=12)
ax3d.plot([0, 0], [0.5, 0], [3.4, 1.8], color='black', linestyle='-', linewidth=1)  # 指向线

ax3d.text(2.5, 1.5, 3.5, '容易停在平坦区域\n(梯度很小)', fontsize=12)
ax3d.plot([2.5, 3.5], [0.5, 0], [3.4, 0.5], color='black', linestyle='-', linewidth=1)  # 指向线

# ================= 5. 绘制下方 3 个 2D 等高线图 =================
# 定义三个子图的位置 [左, 下, 宽, 高]
rects = [
    [0.08, 0.05, 0.22, 0.25],  # 阶段 1
    [0.38, 0.05, 0.22, 0.25],  # 阶段 2
    [0.68, 0.05, 0.22, 0.25]  # 阶段 3
]
titles = ['阶段 1：靠近鞍点', '阶段 2：进入鞍点/梯度很小区域', '阶段 3：停在平坦区域']

# 对应的轨迹数据切片 (只截取轨迹点，体现阶段变化)
trace_slices = [
    slice(0, 5),  # 陡坡下降
    slice(5, 8),  # 平缓进入鞍点
    slice(8, 12)  # 在平坦区域反复震荡
]

# 定义不同阶段展示的 X 轴和 Y 轴范围 (实现放大的效果)
lims = [
    ([-5, 0], [-2, 3]),  # 阶段 1 范围
    ([-2, 2], [-1.5, 1.5]),  # 阶段 2 范围
    ([1, 5], [-1.5, 1.5])  # 阶段 3 范围
]

axs_2d = []
for i in range(3):
    ax = fig.add_axes(rects[i])
    axs_2d.append(ax)

    # 绘制等高线填充图
    contour = ax.contourf(X, Y, Z, levels=20, cmap='turbo')

    # 获取当前阶段的轨迹点并绘制
    s = trace_slices[i]
    ax.plot(trace_x[s], trace_y[s], color='black', marker='o', markersize=4, linestyle='-', linewidth=1.5, zorder=10)

    # 设置标题和坐标轴范围
    ax.set_title(titles[i], fontsize=12, pad=10)
    ax.set_xlim(lims[i][0])
    ax.set_ylim(lims[i][1])

    # 隐藏刻度，保留边框
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 手动添加带箭头的坐标轴
    # x 轴
    ax.annotate('', xy=(lims[i][0][1], lims[i][1][0]), xytext=(lims[i][0][0], lims[i][1][0]),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    # y 轴
    ax.annotate('', xy=(lims[i][0][0], lims[i][1][1]), xytext=(lims[i][0][0], lims[i][1][0]),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # 添加 x, y 标签
    ax.text(lims[i][0][1], lims[i][1][0] - 0.3, '\(x\)', fontsize=14, ha='right')
    ax.text(lims[i][0][0] - 0.3, lims[i][1][1], '\(y\)', fontsize=14, va='top')

# ================= 6. 添加连接线、虚线框和颜色条 =================
# 6.1 3D 图中间虚线框 (用 2D 坐标近似画在 3D 投影上很复杂，这里通过添加一个虚线框在中间的图上方模拟)
# 实际上原图是一个虚线框连接 3D 图的某块区域。这里用一条竖线连接 3D 和中间的 2D 图
# 添加从 3D 图到阶段 2 的垂直虚线
fig.add_artist(FancyArrowPatch((0.49, 0.38), (0.49, 0.31),
                               arrowstyle='->', mutation_scale=20, linestyle='--', color='black', linewidth=1.5))

# 6.2 2D 图之间的箭头
# 从阶段 1 到阶段 2
fig.add_artist(FancyArrowPatch((0.31, 0.175), (0.37, 0.175),
                               arrowstyle='->', mutation_scale=20, color='black', linewidth=2))
# 从阶段 2 到阶段 3
fig.add_artist(FancyArrowPatch((0.61, 0.175), (0.67, 0.175),
                               arrowstyle='->', mutation_scale=20, color='black', linewidth=2))

# 6.3 底部颜色条 (Colorbar)
cax = fig.add_axes([0.4, 0.01, 0.2, 0.02])  # [左, 下, 宽, 高]
cbar = fig.colorbar(contour, cax=cax, orientation='horizontal')
cbar.set_ticks([])  # 隐藏刻度
cbar.set_label('\(f(x, y)\)', fontsize=14, labelpad=10)
# 添加“低”和“高”文字
cbar.ax.text(-0.05, 0.5, '低', va='center', ha='right', transform=cbar.ax.transAxes, fontsize=12)
cbar.ax.text(1.05, 0.5, '高', va='center', ha='left', transform=cbar.ax.transAxes, fontsize=12)

plt.show()
# 如需保存，取消注释下一行
# plt.savefig('optimization_landscape_perfect.png', dpi=300, bbox_inches='tight')