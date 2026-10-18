import torch
import numpy as  np

# def dm_tensor():
#     data = torch.tensor(10)
#     print(data)
#     print(data.shape)
#     print("*" * 40)
#
#     data1 = torch.tensor([3])
#     print(data1)
#     print(data1.shape)
#     print("*" * 40)
#
#     data2 = torch.tensor([[3, 4], [5, 6]])
#     print(data2)
#     print(data2.shape)
#     print("*" * 40)
#
#     data3 = torch.tensor([[[1, 2], [3, 4]],
#                       [[5, 6], [7, 8]]])
#     print(data3)
#     print(data3.shape)
#     print("*" * 40)
#
#     array_list = [[[1, 2, 3], [4, 5, 6], [6, 5, 4], [7, 8, 9]]]
#     data3_other = torch.tensor(array_list)
#     print(data3_other)
#     print(data3_other.shape)
#     print(array_list)
#     print("*" * 40)
#
#     array = np.random.randn()


# def dm02_tensor():
#     data = torch.Tensor(2, 3)
#     print(data)
#     print(data.shape)
#     print("*" * 40)
#
#     data1 = torch.Tensor([20])
#     print(data1)
#     print(data1.shape)
#     print("*" * 40)
#
#     data2 = torch.Tensor([20, 30])
#     print(data2)
#     print(data2.shape)
#     print("*" * 40)
#
#     data3 = torch.Tensor([[20, 30]])
#     print(data3)
#     print(data3.shape)
#     print("*" * 40)
#
#     data4 = torch.Tensor([[20, 30], [40, 50]])
#     print(data4)
#     print(data4.shape)
#     print("*" * 40)
#
#     data4_other = torch.full((2, 3, 4), 7)
#     print(data4_other)
#     print(data4_other.shape)
#     print("*" * 40)

# def dm1_arrange():
#     data = torch.arange(start=2, end=10, step=2)
#     print(data)
#     print(data.shape)
#     print("*" * 40)
#
#     data1 = torch.arange(start=0, end=11, step=1)
#     data2 = torch.arange(11)
#     data3 = torch.tensor(list(range(11)))  # Python list
#     data4 = torch.from_numpy(np.arange(11))  # NumPy
#     data5 = torch.linspace(0, 10, 11, dtype=torch.int64)  # 注意 dtype
#     data6 = torch.tensor(list(range(11)), dtype=torch.int32)  # int32
#     print(data1)
#     print(data1.shape)
#     print("*" * 40)
#     print(data2)
#     print(data3)
#     print(data4)
#     print(data5)
#     print(data6)
#     print(data2.shape)
#     print("*" * 40)


# def dm03():
#     data5 = torch.linspace(0, 10, 11, dtype=torch.int64)  # 注意 dtype
#     print(data5)
#     print(data5.shape)

# def dm04():
#     data = torch.randn(2, 3)
#     print(data)
#     print(data.shape)
#     print("*" * 40)
#     print(torch.random.initial_seed())
#
#     torch.manual_seed(10)
#     data = torch.randn(2, 3)
#     print(data)
#     print(data.shape)

# def dm05():
#     data = torch.ones(2, 3)
#     print(data)
#     print(data.shape)
#     print("*" * 40)
#
#     data1 = torch.randn(2, 3)
#     print(data1)
#     print(data1.shape)
#     print("*" * 40)
#
#     data1 = torch.ones_like(data)
#     print(data1)
#     print(data1.shape)
#     print("*" * 40)
#
#     data2 = torch.full(size=(2, 3), fill_value=3.141592657)
#     print(data2)
#     print(data2.shape)
#     print("*" * 40)
#
#     data3 = torch.full_like(data2, fill_value=3.141592657)
#     print(data3)
#     print(data3.shape)
#     print("*" * 40)
#
#     data = torch.randint(low=2, high=10, size=(1, 2, 3))
#
#     print(data)
#     print(data.shape)
#     data = data.int()
#     print(data)
#     print(data.shape)
#     data = data.float()
#     print(data)
#     print(data.shape)
#     data = data.int()
#     print(data)
#     print(data.shape)

def dm06():
    data = torch.tensor([2, 3, 4])
    print(data)
    print(data.shape)
    print(type(data))

    data_np = data.numpy()
    print(data_np)
    print(data_np.shape)
    print(type(data_np))

    data_np_cp = data.numpy().copy()
    data_np_cp[0] = 100
    print(data_np_cp)
    print(data_np.shape)

    data_a = np.array([2, 3, 4])
    print(data_a)
    print(data_a.shape)
    print(type(data_a))

    data_tensor = torch.from_numpy(data)
    data_tensor[0] = 100
    print(data_tensor)
    print(data_tensor.shape)
    print(type(data_tensor))

    data_tensor = torch.tensor(data)
    print(data_tensor)
    print(data)


if __name__ == '__main__':
    # dm_tensor()
    # dm1_arrange()
    # dm02_tensor()
    # dm03()
    # dm04
    # dm05()
    dm06()
    print('end')