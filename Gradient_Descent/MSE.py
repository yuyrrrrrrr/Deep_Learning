# import matplotlib.pyplot as plt
# import numpy as np
#
# # 1. 设置中文字体，防止中文乱码
# # Windows 系统通常使用 'SimHei'，Mac 系统通常使用 'Arial Unicode MS'
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 2. 设置画布大小
# fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
#
# # 3. 定义数据 (根据原图视觉位置提取)
# # 真实值点 (x, y)
# x_vals = [50, 70, 90, 110, 130]
# y_true = [108, 178, 196, 240, 315]
#
# # 误差标签文字
# error_labels = ['e=9.0', 'e=15.0', 'e=196.0', 'e=289.0', 'e=225.0']
#
# # 4. 定义模型预测线 y = mx + c
# # 通过视觉推断，斜率为 2.5，截距为 -20 左右最贴合原图
# m = 2.5
# c = -20
# x_line = np.linspace(40, 140, 100)
# y_line = m * x_line + c
# y_pred = [m * x + c for x in x_vals]
#
# # 5. 绘制图形
# # 绘制散点 (真实值)
# ax.scatter(x_vals, y_true, color='#0000CD', s=50, zorder=5, label='真实值')
#
# # 绘制回归线 (模型预测线)
# ax.plot(x_line, y_line, color='#C0392B', linewidth=2, zorder=3, label='模型预测线')
#
# # 6. 绘制残差虚线和误差标签
# for i in range(len(x_vals)):
#     # 绘制垂直虚线 (误差)
#     if i == 0:
#         # 为了避免图例中重复出现，只在第一次循环时添加 label
#         ax.plot([x_vals[i], x_vals[i]], [y_true[i], y_pred[i]],
#                 color='#8E44AD', linestyle='--', linewidth=1.5, zorder=4, label='误差(残差)')
#     else:
#         ax.plot([x_vals[i], x_vals[i]], [y_true[i], y_pred[i]],
#                 color='#8E44AD', linestyle='--', linewidth=1.5, zorder=4)
#
#     # 添加误差文字标签 (偏移一点位置避免遮挡)
#     # 根据点在线之上还是线之下，调整文字的上下位置
#     y_offset = 12 if y_true[i] > y_pred[i] else -12
#     # 前两个点和最后一个点在线上方，中间两个点在下方
#     if i in [0, 1, 4]:
#         y_text_pos = max(y_true[i], y_pred[i]) + 10
#     else:
#         y_text_pos = max(y_true[i], y_pred[i]) - 20
#
#     ax.text(x_vals[i] + 3, y_text_pos, error_labels[i],
#             fontsize=9, color='#8E44AD', va='center')
#
# # 7. 设置坐标轴标签和范围
# ax.set_xlabel('面积 ( \(m^2\) )', fontsize=11)
# ax.set_ylabel('总价 (万)', fontsize=11)
# ax.set_xlim(35, 145)
# ax.set_ylim(70, 340)
#
# # 8. 设置刻度
# ax.set_xticks(np.arange(40, 141, 20))
# ax.set_yticks(np.arange(100, 301, 50))
#
# # 9. 添加网格线
# ax.grid(True, linestyle='--', alpha=0.6)
#
# # 10. 添加图例
# # 将图例放在左上角，设置边框和背景
# legend = ax.legend(loc='upper left', frameon=True, framealpha=1, edgecolor='lightgray')
# # 调整图例中文字的大小
# for text in legend.get_texts():
#     text.set_fontsize(10)
#
# # 11. 调整布局并显示/保存
# plt.tight_layout()
# plt.show()
#
# # 如果你想把图片保存下来插入到 Obsidian 中，可以取消下面这行的注释：
# # plt.savefig('regression_residuals.png', dpi=300, bbox_inches='tight')

import matplotlib.pyplot as plt
import numpy as np

# 1. 设置画布与背景颜色 (暗黑主题)
fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
fig.patch.set_facecolor('#0f0f0f')  # 设置整个图形背景色
ax.set_facecolor('#0f0f0f')         # 设置坐标轴背景色

# 2. 隐藏默认的边框和刻度
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# 设置坐标轴范围
ax.set_xlim(-5, 5)
ax.set_ylim(-2, 2.5)

# 3. 绘制中心十字坐标轴 (灰色)
ax.axhline(0, color='#777777', linewidth=1.5, zorder=1) # x轴
# y轴使用 annotate 添加箭头
ax.annotate('', xy=(0, 2.2), xytext=(0, -2),
            arrowprops=dict(arrowstyle="->", color='#777777', lw=1.5))

# 4. 绘制 MSE 曲线 (蓝紫色斜线，y = x)
# 实际原图斜率可能略小于1，这里为了标准起见设为 y = 0.8x
x_mse = np.linspace(-4.5, 4.5, 100)
y_mse = x_mse * 0.9
ax.plot(x_mse, y_mse, color='#8a5cf5', linewidth=2.5, zorder=3)

# 5. 绘制 MAE 曲线 (红色阶梯线，断开)
# 左半段
ax.plot([-4.5, 0], [-1, -1], color='#f25c5c', linewidth=2.5, zorder=2)
# 右半段
ax.plot([0, 4.5], [1, 1], color='#f25c5c', linewidth=2.5, zorder=2)

# 在 x=0 处画两个红点（使用空心圆点样式更贴合原图，或者实心，这里用实心）
ax.scatter([0, 0], [1, -1], color='#f25c5c', s=40, zorder=4)

# 6. 添加文字标签
# y轴上方标签 ∂L/∂ŷ
ax.text(0.1, 2.2, r'\(\partial L / \partial \hat{y}\)', color='#aaaaaa', fontsize=14, va='bottom')

# MSE 与 MAE 标签 (带旋转)
ax.text(2.8, 1.5, 'MSE', color='#8a5cf5', fontsize=12, rotation=15, fontweight='bold')
ax.text(2.8, 1.25, 'MAE', color='#f25c5c', fontsize=12, fontweight='bold')
ax.text(3.3, 1.3, '比例', color='#d166ff', fontsize=11, rotation=15)

# 7. 调整布局并显示
plt.tight_layout()
plt.show()

# 如果你想保存图片到本地直接拖入 Obsidian，可以取消下面这行的注释：
# plt.savefig('mse_vs_mae_gradient.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())