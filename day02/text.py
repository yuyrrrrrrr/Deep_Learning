import torch


def print_sep(title):
    print(f"\n{'='*15} {title} {'='*15}")

# ==========================================
# 基础数据准备 (2行3列的 2D 张量)
# ==========================================

data = torch.tensor([[1, 2, 3],
                     [4, 5, 6]])
print("【基础原始数据 data】:\n", data, "\n形状:", data.shape)


# ==========================================
# 1. reshape / view (改变形状)
# ==========================================
print_sep("reshape / view")
print("原始数据:\n", data)
reshaped_data = data.reshape(3, 2)
print("-> data.reshape(3, 2) 结果:\n", reshaped_data)
# 输出:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6]])

viewed_data = data.view(6)
print("-> data.view(6) 结果:\n", viewed_data)
# 输出: tensor([1, 2, 3, 4, 5, 6])


# ==========================================
# 2. unsqueeze (增加一个维度)
# ==========================================
print_sep("unsqueeze")
print("原始数据:\n", data)
unsqueezed_data = data.unsqueeze(0) # 在第0维增加一个维度
print("-> data.unsqueeze(0) 结果:\n", unsqueezed_data)
print("形状变化:", data.shape, "->", unsqueezed_data.shape)
# 输出:
# tensor([[[1, 2, 3],
#          [4, 5, 6]]])
# 形状变化: torch.Size([2, 3]) -> torch.Size([1, 2, 3])


# ==========================================
# 3. squeeze (去掉大小为1的维度)
# ==========================================
print_sep("squeeze")
print("原始数据 (使用上面的 unsqueezed_data):\n", unsqueezed_data)
squeezed_data = unsqueezed_data.squeeze(0) # 去掉第0维
print("-> unsqueezed_data.squeeze(0) 结果:\n", squeezed_data)
print("形状变化:", unsqueezed_data.shape, "->", squeezed_data.shape)
# 输出:
# tensor([[1, 2, 3],
#         [4, 5, 6]])
# 形状变化: torch.Size([1, 2, 3]) -> torch.Size([2, 3])


# ==========================================
# 4. transpose (交换两个维度)
# ==========================================
print_sep("transpose")
print("原始数据:\n", data)
transposed_data = data.transpose(0, 1) # 交换第0维和第1维
print("-> data.transpose(0, 1) 结果:\n", transposed_data)
# 输出:
# tensor([[1, 4],
#         [2, 5],
#         [3, 6]])


# ==========================================
# 5. permute (重新排列所有维度)
# ==========================================
print_sep("permute")
# 创建一个 3D 张量 (2, 2, 2)
data_3d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print("原始 3D 数据:\n", data_3d)
permuted_data = data_3d.permute(2, 1, 0) # 将维度顺序从 (0,1,2) 变成 (2,1,0)
print("-> data_3d.permute(2, 1, 0) 结果:\n", permuted_data)
# 输出:
# tensor([[[1, 5],
#          [3, 7]],
#
#         [[2, 6],
#          [4, 8]]])


# ==========================================
# 6. cat (在已有维度上拼接)
# ==========================================
print_sep("cat (拼接)")
A = torch.tensor([[1, 2], [3, 4]])
B = torch.tensor([[5, 6], [7, 8]])
print("张量 A:\n", A)
print("张量 B:\n", B)

cat_dim0 = torch.cat([A, B], dim=0)
print("-> torch.cat([A, B], dim=0) (按行拼):\n", cat_dim0)
# 输出:
# tensor([[1, 2],
#         [3, 4],
#         [5, 6],
#         [7, 8]])

cat_dim1 = torch.cat([A, B], dim=1)
print("-> torch.cat([A, B], dim=1) (按列拼):\n", cat_dim1)
# 输出:
# tensor([[1, 2, 5, 6],
#         [3, 4, 7, 8]])


# ==========================================
# 7. stack (在新维度上堆叠)
# ==========================================
print_sep("stack (堆叠)")
print("张量 A:\n", A)
print("张量 B:\n", B)

stack_dim0 = torch.stack([A, B], dim=0)
print("-> torch.stack([A, B], dim=0) (在最外层加一维):\n", stack_dim0)
# 输出:
# tensor([[[1, 2],
#          [3, 4]],
#
#         [[5, 6],
#          [7, 8]]])

stack_dim1 = torch.stack([A, B], dim=1)
print("-> torch.stack([A, B], dim=1) (在中间加一维):\n", stack_dim1)
# 输出:
# tensor([[[1, 2],
#          [5, 6]],
#
#         [[3, 4],
#          [7, 8]]])


# ==========================================
# 8. 核心避坑：contiguous() 与 view() 的恩怨
# ==========================================
print_sep("transpose 后的内存连续性")
print("原始数据 data:\n", data)
data_T = data.transpose(0, 1)
print("转置后的 data_T:\n", data_T)

# 尝试 1：直接 view 会报错！
try:
    print("尝试直接 data_T.view(6)...")
    print(data_T.view(6))
except RuntimeError as e:
    print("\n❌ 报错了！错误信息如下：")
    print(e)

# 尝试 2：正确做法
print("\n✅ 正确做法：先调用 contiguous()")
view_data = data_T.contiguous().view(6)
print("-> data_T.contiguous().view(6) 结果:\n", view_data)
# 输出: tensor([1, 4, 2, 5, 3, 6])
