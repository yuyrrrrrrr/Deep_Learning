import torch
import torch.nn as nn


def dm01_pool():
    """
    展示池化操作
    :return:
    """
    # 1. 准备数据
    # 1x3x3 维数据，单通道
    data = torch.tensor([
        [[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]]
    ], dtype=torch.float) # 转浮点型，后续要计算平均

    # # 多通道
    # # 3x3x3维度
    # data = torch.tensor([
    #     [[0, 1., 2],
    #     [3, 4, 5],
    #     [6, 7, 8]],
    #
    #     [[0, 1, 2],
    #      [3, 4, 5],
    #      [6, 7, 8]],
    #
    #     [[0, 1, 2],
    #      [3, 4, 5],
    #      [6, 7, 8]]
    # ], dtype=torch.float) # 转浮点型

    print("data--->\n", data)
    print("data.shape--->", data.shape)

    # 2. 最大池化
    pool_max = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
    # 2.1 把数据送给池化层
    data1= pool_max(data)
    # 2.2 打印结果
    print("data1--->\n", data1)
    print("data1.shape--->", data1.shape)

    # 3. 平均池化
    # 注意：数据要改成浮点型，否则结果不对
    pool_avg = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
    data2 = pool_avg(data)
    print("data2--->\n", data2)
    print("data2.shape--->", data2.shape)



if __name__ == '__main__':
    dm01_pool()

    print("End!")