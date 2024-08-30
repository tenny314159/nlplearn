# 张量操作

import torch

A = torch.arange(12.0).reshape(3, 4)
print(A)
# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.]])


A = A.resize_(2, 6)  # 形状修改 跟reshape()类似
print(A)
# tensor([[ 0.,  1.,  2.,  3.,  4.,  5.],
#         [ 6.,  7.,  8.,  9., 10., 11.]])


B = A.resize_as(A)  # 形状修改 跟reshape()类似
print(B)
# tensor([[ 0.,  1.,  2.,  3.,  4.,  5.],
#         [ 6.,  7.,  8.,  9., 10., 11.]])


x = torch.tensor([[1], [2], [3]])
print("原始张量：")
print(x)
print("原始张量的形状：", x.shape)
# 原始张量：
# tensor([[1],
#         [2],
#         [3]])
# 原始张量的形状： torch.Size([3, 1])

# 使用squeeze()函数去除维度为1的维度
y = torch.squeeze(x)
print("\n去除维度为1后的张量：")
print(y)
print("去除维度为1后的张量形状：", y.shape)
# 去除维度为1后的张量：
# tensor([1, 2, 3])
# 去除维度为1后的张量形状： torch.Size([3])


C = torch.arange(24).reshape(2, 3, 4)
C = C.reshape(1, 1, 3, 4, 2, 1, 1)
print(C)
print(C.shape)
# torch.Size([1, 1, 3, 4, 2, 1, 1])
C = torch.squeeze(C)
print(C)
print(C.shape)
# torch.Size([3, 4, 2])   #从这个实验可以看出 squeeze()是把维度数量为1 的全部都降维掉了


A = torch.arange(3)
B = A.expand(3, -1)
print(B)
# tensor([[0, 1, 2],
#         [0, 1, 2],
#         [0, 1, 2]])


D = B.repeat(1, 2, 2)  # 重复填充
print(D)
print(D.shape)
# tensor([[[0, 1, 2, 0, 1, 2],
#          [0, 1, 2, 0, 1, 2],
#          [0, 1, 2, 0, 1, 2],
#          [0, 1, 2, 0, 1, 2],
#          [0, 1, 2, 0, 1, 2],
#          [0, 1, 2, 0, 1, 2]]])


A = torch.arange(12).reshape(1, 3, 4)
B = -A
print(torch.where(A > 5, A, B)) # torch.where() 当A >5 为真时返回x 对应位置值 ， 假时返回y的值


print(A[A > 5])
# tensor([ 6,  7,  8,  9, 10, 11])


print(A)
# tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]]])


print(torch.tril(A,diagonal=0)) # diagonal控制要考虑的对角线  下三角
# tensor([[[ 0,  0,  0,  0],
#          [ 4,  5,  0,  0],
#          [ 8,  9, 10,  0]]])


print(torch.triu(A,diagonal=1))  # 上三角
# tensor([[[ 0,  1,  2,  3],
#          [ 0,  0,  6,  7],
#          [ 0,  0,  0, 11]]])


A = A.reshape(3,4)
print(A)
# tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]]])
print(torch.diag(A , diagonal=0))
# tensor([ 0,  5, 10])

print(torch.diag(A,diagonal=1))
# tensor([ 1,  6, 11])

print(torch.diag(A,diagonal=-1))
# tensor([4, 9])

