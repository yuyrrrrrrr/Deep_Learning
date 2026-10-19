import torch


# def dm01():
#     a = torch.tensor([1., 2., 3.])
#
#     a = a.add(1)
#     print(a)
#     a = a.sub(1)
#     print(a)
#     a = a.mul(2)
#     print(a)
#     a = a.div(2)
#     print(a)
#     b = torch.tensor([4., 5., 6.])
#     # 逐元素a ** 2;
#     print(a + 1)
#     print(a + b)
#     print(a - b)
#     print(a * b)
#     print(a / b)
#     torch.pow(a, 2)  # 幂
#     print(torch.pow(a, 2))
#     torch.sqrt(a)  # 平方根
#     print(torch.sqrt(a))
#     torch.exp(a)   # 指数e的a次方
#     print(torch.exp(a))
#     torch.log(a)  # 自然对数
#     print(torch.log(a))
#     torch.abs(a)  # 绝对值
#     print(torch.abs(a))
#     torch.neg(a)  # 取负
#     print(torch.neg(a))
#     # 矩阵乘
#     m1 = torch.randn(2, 3)
#     print(m1)
#     m2 = torch.randn(3, 4)
#     print(m2)
#     m1 @ m2  # 推荐写法
#     print(m1 @ m2)
#     torch.matmul(m1, m2)
#     print(torch.matmul(m1, m2))
#     torch.mm(m1, m2)  # 只能二维
#     print(torch.mm(m1, m2))
#     # 批量矩阵乘
#     batch1 = torch.randn(10, 2, 3)
#     batch2 = torch.randn(10, 3, 4)
#     torch.bmm(batch1, batch2)  # [10,2,4]
#     print(torch.bmm(batch1, batch2))

#
# def dm2():
#     x = torch.arange(12).reshape(3, 4)
#     a = x[0]  # 第 0 行
#     b = x[:, 1]  # 第 1 列
#     c = x[1, 2]  # 第 1 行第 2 列，标量张量
#     d = x[0:2, 1:3]  # 切片 第1行到第2行, 第2列到第3列 区间右开
#     e = x[x > 5]  # 布尔索引，返回一维
#     f = x[[0, 2]]  # 花式索引，取第 0、2 行
#     g = x[..., 0]  # ... 表示省略前面所有维度
#     print(x)
#     print(a)
#     print(b)
#     print(c)
#     print(d)
#     print(e)
#     print(f)
#     print(g)
#     y = x[0:1]
#     print(y)
#     y[0, 0] = 999  # 会改到 x！
#     print(y)
#     torch.manual_seed(10)
#     data = torch.randint(0, 10, (3, 4, 3))
#     print(data)
#
#     data1 = data[0, :, :]
#     data2 = data[:, 0, :]
#     data3 = data[:, :, 0]
#
#     data4 = data[1, :, :]
#     data5 = data[:, 1, :]
#     data6 = data[:, :, 1]
#
#     print(data)
#     print(data1)
#     print(data2)
#     print(data3)
#     print(data4)
#     print(data5)
#     print(data6)

# cat 拼已有维度, stack 新增维度
# def dm03():
#     a = torch.randn(2, 3)
#     b = torch.randn(2, 3)
#     d = torch.cat([a, b], dim=0)  # [4,3] 沿已有维度拼
#     e = torch.cat([a, b], dim=1)  # [2,6]
#     f = torch.stack([a, b], dim=0)  # [2,2,3] 新建一个维度
#     c = torch.randn(6, 3)
#     g = torch.split(c, 2, dim=0)  # 按大小 2 切 → 3 个 [2,3]
#     h = torch.chunk(c, 3, dim=0)  # 按份数 3 切 → 3 个 [2,3]
#     print(a)
#     print(b)
#     print(c)
#     print(d)
#     print(e)
#     print(f)
#     print(g)
#     print(h)

# def dm04():
#     x = torch.randn(3, 4)
#     print(x)
#     print(x.sum())  # 全部求和，标量
#     print(x.sum(dim=0))  # 沿 dim0 求和 → [4]
#     print(x.sum(dim=1, keepdim=True))  # → [3,1]，保留维度方便广播
#     print(x.mean())
#     print(x.median())  # 注意：整型不能 mean，会报错
#     print(x.max())
#     print(x.min())
#     print(x.max(dim=1))  # 返回 (values, indices) 命名元组
#     print(x.argmax(dim=1))  # 只返回索引
#     print(x.std())
#     print(x.var())
#     print(x.cumsum(dim=0))  # 累加
#     print(torch.sort(x, dim=1))  # 排序，返回 values 和 indices

def dm05():
    x = torch.randn(3, requires_grad=True)
    y = (x ** 2).sum()
    y.backward()
    print(x.grad)  # dy/dx
    x_detached = x.detach()  # 脱离计算图，共享内存
    print(x_detached.requires_grad)  # False
    print(x_detached.data_ptr() == x.data_ptr())  # True，共享内存

    x_detached[0] = 100.0
    print(x)  # x[0] 也变成 100.0，因为共享内存
    with torch.no_grad():  # 上下文内不记录梯度，推理时用
        z = x * 2
    x.requires_grad_(False)  # 就地关闭


def dm01_squeeze_unsqueeze():
    # 为了演示前半段代码，假设 data 是之前定义好的 (这里给个默认值防止报错)
    data = torch.randn(2, 3)

    data1 = data.unsqueeze(dim=1)
    print("data1--->", data1)
    print("data1.shape--->", data1.shape)
    print("*" * 40)

    # 3. 降维操作
    data2 = data1.squeeze()
    print("data2--->", data2)
    print("data2.shape--->", data2.shape)
    print("*" * 40)

    # todo 部分实现
    print("--- todo 部分 ---")

    # 1. 创建张量：维度为 [1, 2, 3, 4, 5]
    # 这里使用 torch.randn 生成随机数据，形状为 1x2x3x4x5
    data_todo = torch.randn(1, 2, 3, 4, 5)
    print("原始张量形状:", data_todo.shape)

    # 2. 降维到 [2, 3, 4, 5]
    # squeeze() 默认会去掉所有大小为 1 的维度
    # 也可以指定维度：data_todo.squeeze(dim=0)
    data_squeezed = data_todo.squeeze()
    print("降维后形状:", data_squeezed.shape)

    # 3. 数据维度还原
    # 在第 0 维增加一个大小为 1 的维度
    data_restored = data_squeezed.unsqueeze(dim=0)
    print("还原后形状:", data_restored.shape)


if __name__ == '__main__':
    # dm01()
    # dm2()
    # dm03()
    # dm04()
    # dm05()
    dm01_squeeze_unsqueeze()
    print("end")
