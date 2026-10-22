import torch


def dm01_softmax():
    """
    实现softmax, 将输出结果转成概率
    """
    # 定义模型输出logits (z)
    logits = torch.tensor([0.2, 0.02, 0.15, 0.15, 1.3, 0.5, 0.06, 1.1, 0.05, 3.75])
    # y_pred = torch.softmax(logits, dim=0)
    y_pred = torch.softmax(logits, dim=-1) # 统一使用-1
    print("y_pred--->", y_pred)
    print("y_pred.shape--->", y_pred.shape)


if __name__ == '__main__':
    dm01_softmax()
    print("end")