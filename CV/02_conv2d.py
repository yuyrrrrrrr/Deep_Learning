import matplotlib.pyplot as plt
import torch
import torch.nn as nn



def dm01_conv2d():
    """
    CNN在Pytorch中的实现API
    演示特征图计算
    :return:
    """
    # 1. 准备并读取图像数据
    data = plt.imread('../data/img.jpg')
    # print("data--->", data)
    print("data.shape--->", data.shape)  # (640, 640, 3)
    print("type(data)--->", type(data))
    print("data.dtype--->", data.dtype)
    # plt.imshow(data)
    # plt.show()

    # 2. 创建卷积层
    # in_channels:输入通道数，输入图片数据通道数为3，这里也必须设置为3
    # out_channels: 输出通道数量，也就是卷积核的数量，还可以是特征图的通道数量
    # kernel_size：卷积核大小
    # stride: 步长， 默认1
    # padding: 填充，默认0
    conv = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1)

    # 3. 调整维度顺序并转浮点型
    # 输入图像数据为的维度：HWC
    # Pytorch框架要求的格式：CHW
    # permute实现逻辑: 按照转换后的顺序，填写原来的索引
    data = torch.tensor(data).permute(2, 0, 1)

    # pytorch中浮点数默认是32位，需要将数据转成float32的
    data = data.to(torch.float)
    print("data.shape--->", data.shape) # [3, 640, 640]

    # 4. 升维度，
    # 为什么升维？体现样本的数量
    # [C, H, W] ---> [N, C, H, W]
    data = data.unsqueeze(0)
    print("data.shape--->", data.shape) # [1, 3, 640, 640]

    # 5. 把数据给到卷积层
    feature_map = conv(data)
    # 6. 打印结果
    # print("feature_map--->", feature_map)
    # N = (W - F + 2P)/S + 1
    # N = (640 - 3 + 0)/1 + 1 = 638
    # N = (640 - 3 + 0)/2 + 1 = 319
    print("feature_map.shape--->", feature_map.shape) # torch.Size([1, 3, 638, 638])


if __name__ == '__main__':
    dm01_conv2d()
    print("End!")