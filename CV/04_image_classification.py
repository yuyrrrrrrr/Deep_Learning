# tqdm 进度条 pip install tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor
import time
import warnings
warnings.filterwarnings("ignore")

# ***需要在当前路径创建data文件夹，将cifar10数据拷贝进去***
# 需要提升精度可以修改epochs等参数

# 1. 加载数据
def load_data():
    """
    加载cifar10数据
    :return: Dataset类型的数据集
    """
    # root:数据目录，如果不存在，会自动下载（很慢）
    # train: 是否是训练集
    # transform: 对数据进行处理，这里是把数据处理成张量
    train = CIFAR10(root="./data", train=True, transform=Compose([ToTensor()]))
    test = CIFAR10(root="./data", train=False, transform=Compose([ToTensor()]))
    return train, test


def test_load_data():
    # 调用数据加载函数
    train, test = load_data()
    # 打印数据
    print("train--->", train)
    print("test--->", test)

    # 打印数据集的类别
    print("train class--->", train.class_to_idx) # 所有标签的类别索引，编码

    # print("tain--->", train.data)# 所有数据的结果，很大
    print("train.data.shape--->", train.data.shape)# 所有数据的结果，很大
    print("train.data[0].shape--->", train.data[0].shape)# 一个样本（一张图片）的shape

    print("len(tain.targets)--->", len(train.targets)) # 标签数量 targets是列表
    print("train.targets[0]--->", train.targets[0]) # 标签样例

    # 可视化图片
    plt.figure(figsize=(2, 2))
    plt.imshow(train.data[0]) # 青蛙
    plt.show()


# 2. 模型构建
# 2.1 创建一个类，继承自nn.Module
class ImageModel(nn.Module):
    def __init__(self):
        # super(ImageModel, self).__init__() # 旧时写法，python2的写法
        super().__init__() # 新式写法，python3
        # 创建隐藏层
        # 创建第一个卷积层和池化层
        # in_channels = 3，因为输入图片是三通道的；out_channels数字随便定义的（下同）
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(3,3))
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 创建第二个卷积层和池化层
        # in_channels=6一定要等于上个卷积层的输出数量（下同）
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3,3))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # 创建输出层
        # in_features=576 是根据特征图的计算公式得出的
        self.linear1 = nn.Linear(in_features=576, out_features=120)
        self.linear2 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10) # out_features=10是因为输出10个类别


   # 2.2 创建init和forward方法
    def forward(self, x):
        # 让数据经过第一个卷积层和池化层
        x = self.conv1(x)
        # print("x.shape--->", x.shape) # 打印数据维度
        x = self.pool1(x)
        # print("x.shape--->", x.shape)
        # # 再经过第二个卷积层和池化层
        x = self.conv2(x)
        # print("x.shape--->", x.shape)
        x = self.pool2(x) # [1, 16, 6, 6]
        # print("x.shape--->", x.shape)

        # ------------测试下数据维度-----------
        #数据维度转换  [1, 16, 6, 6] ---> [1, 576]
        x = x.reshape(x.shape[0], -1) # [1, 576]
        # print("x.shape--->", x.shape)
        # 也可以使用的方式
        # x = torch.reshape(x, shape=(x.shape[0], -1))

        # 线性层

        # 让数据经过第一线性层和激活层
        x = self.linear1(x)
        # print("x.shape--->", x.shape)
        x = torch.relu(x)
        # 再数据经过第二线性层和激活层
        x = self.linear2(x)
        # print("x.shape--->", x.shape)
        x = torch.relu(x)
        # 最后数据经过第三线性层
        out = self.out(x)
        return out

def test_ImageModel():
    """
    测试模型和数据
    :return:
    """
    # 准备数据
    train, test = load_data()
    # 构建DastaLoader
    dataloader = DataLoader(dataset=train, batch_size=1, shuffle=True, drop_last=True)
    #实例化模型
    model = ImageModel()
    # 遍历数据
    for x, y in dataloader: # dataset 中设置了两个变量
        # print("x--->", x)
        print("x.shape--->", x.shape) # [1, 3, 32, 32]
        # print("y--->", y)
        print("y.shape--->", y.shape) # [1]
        # 把数据喂给模型
        # model(x) # 暂时没有返回值，不需要接收
        output = model(x)
        print("output.shape--->", output.shape) # [1, 10]
        break

# 3. 模型训练
def train_model():
    """
    构建模型训练流程
    :return:
    """
    # 3.1 准备数据dataset和dataloader
    train, test = load_data()
    dataloader = DataLoader(dataset=train, batch_size=4, shuffle=True, drop_last=True)
    # 3.2 创建模型
    model = ImageModel()
    # 3.3 创建损失函数
    criterion = nn.CrossEntropyLoss()
    # 3.4 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 3.5 外层遍历轮次
    # 参数
    epochs = 2 # 可修改该参数，用于提升性能
    total_num = 0
    total_loss = 0
    start_time = time.time()
    for epoch in range(epochs):
        # 3.6 内层遍历批次
        for x, y in tqdm(dataloader):
            # 3.7 前向传播
            y_pred = model(x)
            # 3.8 损失计算
            loss = criterion(y_pred, y)
            # 3.9 梯度清零
            optimizer.zero_grad()
            # 3.10 反向传播
            loss.backward()
            # 3.11 梯度更新
            optimizer.step()

            # 记录日志
            total_num += len(y)
            total_loss += loss.item()
        # 3.12 打印日志
        print("epoch--->", epoch, "loss--->", total_loss / total_num, "time--->", time.time() - start_time)

    # 3.13 保存模型
    torch.save(model.state_dict(), "./model.pth")

# 4. 模型预测（评估）
def eval_model():
    """
    评估模型性能
    :return:
    """
    # 4.1 加载数据
    train, test = load_data()
    # 转换位dataloader
    dataloader = DataLoader(dataset=test, batch_size=1, shuffle=False)
    # 4.2 加载模型和参数
    model = ImageModel()
    model.load_state_dict(torch.load("./model.pth"))
    # 4.3 预测结果
    # 参数
    total_correct = 0
    total_num = 0
    for x, y in tqdm(dataloader):
        # 前向传播，得到预测结果
        y_pred = model(x)
        # 从预测结果中得到预测标签
        y_pred = torch.argmax(y_pred, dim=-1)
        # 判断预测结果与真实结果是否相等
        total_correct += (y_pred == y).sum()
        total_num += len(y)
    # 4.4 计算正确率
    print("accuracy--->", total_correct.item() / total_num)

if __name__ == '__main__':

    # test_load_data()
    # test_ImageModel()
    train_model()
    eval_model()
    print("end")
