import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchinfo import summary          # 需要 pip install torchinfo
from sklearn.model_selection import train_test_split


# ==================== 1. 数据集构建 ====================
def create_dataset():
    # 使用 pandas 读取数据
    data = pd.read_csv('data/手机价格预测.csv')

    # 特征值和目标值
    x, y = data.iloc[:, :-1], data.iloc[:, -1]

    # 类型转换
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    # 数据集划分
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, train_size=0.8, random_state=88
    )

    # 构建数据集
    train_dataset = TensorDataset(
        torch.from_numpy(x_train.values),
        torch.tensor(y_train.values),
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(x_valid.values),
        torch.tensor(y_valid.values),
    )

    return train_dataset, valid_dataset, x_train.shape[1], len(np.unique(y))


# ==================== 2. 网络模型 ====================
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PhonePriceModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.linear3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        output = self.linear3(x)
        return output


# ==================== 3. 模型训练 ====================
def train(train_dataset, input_dim, class_num):
    # 固定随机数种子
    torch.manual_seed(0)

    # 初始化模型
    model = PhonePriceModel(input_dim, class_num)

    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化方法
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    # 训练轮数
    num_epoch = 50

    # 数据加载器（放到 epoch 循环外面即可，无需每轮重建）
    dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)

    # 切换到训练模式
    model.train()

    # 遍历每个轮次
    for epoch_idx in range(num_epoch):
        start = time.time()
        total_loss = 0.0
        total_num = 0

        # 遍历每个 batch
        for x, y in dataloader:
            output = model(x)                 # 前向传播
            loss = criterion(output, y)       # 计算损失

            optimizer.zero_grad()             # 梯度清零
            loss.backward()                   # 反向传播
            optimizer.step()                  # 参数更新

            total_loss += loss.item()
            total_num += 1

        print('epoch: %s loss: %.2f, time: %.2f' % (
            epoch_idx + 1, total_loss / total_num, time.time() - start
        ))

    # 模型保存（先确保目录存在）
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), 'model/phone.pth')
    print('模型已保存到 model/phone.pth')


# ==================== 4. 模型测试 ====================
def tst(valid_dataset, input_dim, class_num):
    # 加载模型和训练好的网络参数
    model = PhonePriceModel(input_dim, class_num)
    model.load_state_dict(torch.load('model/phone.pth'))

    # 切换到评估模式
    model.eval()

    # 构建加载器
    dataloader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

    correct = 0
    # 评估时不需要计算梯度
    with torch.no_grad():
        for x, y in dataloader:
            output = model(x)
            y_pred = torch.argmax(output, dim=1)
            correct += (y_pred == y).sum().item()

    print('Acc: %.5f' % (correct / len(valid_dataset)))


# ==================== 5. 入口 ====================
if __name__ == '__main__':
    # 获取数据
    train_dataset, valid_dataset, input_dim, class_num = create_dataset()
    print("输入特征数：", input_dim)
    print("分类个数：", class_num)

    # 查看模型结构
    model = PhonePriceModel(input_dim, class_num)
    summary(model, input_size=(16, input_dim))

    # 训练
    train(train_dataset, input_dim, class_num)

    # 测试
    tst(valid_dataset, input_dim, class_num)