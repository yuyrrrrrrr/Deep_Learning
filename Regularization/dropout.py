import torch
import torch.nn as nn

def tst():
    # 初始化随机失活层
    dropout = nn.Dropout(p=0.4)
    # 初始化输入数据:表示某一层的weight信息
    inputs = torch.randint(0, 10, size=[1, 4]).float()
    layer = nn.Linear(4,5)
    y = layer(inputs)
    print("未失活FC层的输出结果：\n", y)

    y =  dropout(y)
    print("失活后FC层的输出结果：\n", y)


tst()