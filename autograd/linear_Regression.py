# 导入相关模块
import torch
from torch.utils.data import TensorDataset  # 构造数据集对象
from torch.utils.data import DataLoader  # 数据加载器
from torch import nn  # nn模块中有平方损失函数和假设函数
from torch import optim  # optim模块中有优化器函数
from sklearn.datasets import make_regression  # 创建线性回归模型数据集
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def create_dataset():
    x, y, coef = make_regression(n_samples=100,
                                 n_features=1,
                                 noise=10,
                                 coef=True,
                                 bias=1.5,
                                 random_state=0)

    # 将构建数据转换为张量类型
    x = torch.tensor(x)
    y = torch.tensor(y)

    return x, y, coef


# 构造数据集
x, y, coef = create_dataset()
# 构造数据集对象
dataset = TensorDataset(x, y)
# 构造数据加载器
# dataset=:数据集对象
# batch_size=:批量训练样本数据
# shuffle=:样本数据是否进行乱序
dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
# 构造模型
# in_features指的是输入张量的大小size
# out_features指的是输出张量的大小size
model = nn.Linear(in_features=1, out_features=1)

# 构造平方损失函数
criterion = nn.MSELoss()
# 构造优化函数
optimizer = optim.SGD(params=model.parameters(), lr=1e-2)
epochs = 100
# 损失的变化
loss_epoch = []
total_loss = 0.0
train_sample = 0.0
for _ in range(epochs):
    for train_x, train_y in dataloader:
        # 将一个batch的训练数据送入模型
        y_pred = model(train_x.type(torch.float32))
        # 计算损失值
        loss = criterion(y_pred, train_y.reshape(-1, 1).type(torch.float32))
        total_loss += loss.item()
        train_sample += len(train_y)
        # 梯度清零
        optimizer.zero_grad()
        # 自动微分(反向传播)
        loss.backward()
        # 更新参数
        optimizer.step()
        # 获取每个batch的损失
        loss_epoch.append(total_loss/train_sample)


if __name__ == "__main__":
    # 生成的数据
    x, y, coef = create_dataset()
    # 绘制损失变化曲线
    plt.plot(loss_epoch)
    plt.title('损失变化曲线')
    plt.grid()
    plt.show()
    # 绘制拟合直线
    plt.scatter(x, y)
    x = torch.linspace(x.min(), x.max(), 1000)
    y1 = torch.tensor([v * model.weight + model.bias for v in x])
    y2 = torch.tensor([v * coef + 1.5 for v in x])
    plt.plot(x, y1, label='训练')
    plt.plot(x, y2, label='真实')
    plt.grid()
    plt.legend()
    plt.show()



