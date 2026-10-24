import torch
import matplotlib.pyplot as plt

ELEMENT_NUMBER = 30
# 1. 实际平均温度
# def tet01():
#     # 固定随机数种子
#     torch.manual_seed(0)
#     # 产生30天的随机温度
#     temperature = torch.randn(size=[ELEMENT_NUMBER,]) * 10
#     print(temperature)
#     # 绘制平均温度
#     days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
#     plt.plot(days, temperature, color='r')
#     plt.scatter(days, temperature)
#     plt.show()
#
# # 2. 指数加权平均温度
# def tet02(beta=0.9):
#     torch.manual_seed(0) # 固定随机数种子
#     temperature = torch.randn(size=[ELEMENT_NUMBER,]) * 10 # 产生30天的随机温度
#     exp_weight_avg = []
#     for idx, temp in enumerate(temperature, 1):  # 从下标1开始
#         # 第一个元素的的 EWA 值等于自身
#         if idx == 1:
#             exp_weight_avg.append(temp)
#             continue
#         # 第二个元素的 EWA 值等于上一个 EWA 乘以 β + 当前气温乘以 (1-β)
#         new_temp = exp_weight_avg[idx - 2] * beta + (1 - beta) * temp
#         exp_weight_avg.append(new_temp)
#     days = torch.arange(1, ELEMENT_NUMBER + 1, 1)
#     plt.plot(days, exp_weight_avg, color='r')
#     plt.scatter(days, temperature)
#     plt.show()


def txt01():
    # 1 初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    loss = ((w ** 2) / 2.0).sum()

    # 2 实例化优化方法：SGD 指定参数beta=0.9
    optimizer = torch.optim.SGD([w], lr=0.01, momentum=0.9)

    # 3 第1次更新 计算梯度，并对参数进行更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第1次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))

    # 4 第2次更新 计算梯度，并对参数进行更新
    # 使用更新后的参数机选输出结果
    loss = ((w ** 2) / 2.0).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('第2次: 梯度w.grad: %f, 更新后的权重:%f' % (w.grad.numpy(), w.detach().numpy()))


if __name__ == '__main__':
    txt01()