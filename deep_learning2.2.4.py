# 张量计算
import torch

A = torch.tensor([10.0])
B = torch.tensor([10.1])

A = torch.tensor(float('nan'))
print(torch.allclose(A, A, equal_nan=True))  # equal_nan 为真时 ，判断缺失值nan为接近
print(torch.allclose(A, A, equal_nan=False))
# True
# False


A = torch.tensor([1, 2, 3, 4, 5, 6])
B = torch.arange(1, 7)
C = torch.unsqueeze(B, dim=0)
print(C)
# tensor([[1, 2, 3, 4, 5, 6]])


print(torch.ge(A, B))  # 判断大于等于
# tensor([True, True, True, True, True, True])

print(torch.ge(A, C))
# tensor([[True, True, True, True, True, True]])


print(torch.gt(A, B))  # 判断大于
# tensor([False, False, False, False, False, False])

print(torch.gt(A, C))
# tensor([[False, False, False, False, False, False]])


A = torch.arange(6.0).reshape(2, 3)
print(A)
# tensor([[0., 1., 2.],
#         [3., 4., 5.]])
print(torch.rsqrt(A))  # reciprocal square root 与下一行代码操作结果相同
# tensor([[   inf, 1.0000, 0.7071],
#         [0.5774, 0.5000, 0.4472]])
print(1 / (A ** 0.5))
# tensor([[   inf, 1.0000, 0.7071],
#         [0.5774, 0.5000, 0.4472]])


A = torch.arange(12.0).reshape(2, 2, 3)
B = torch.arange(12.0).reshape(2, 3, 2)
print(A)
# tensor([[[ 0.,  1.,  2.],
#          [ 3.,  4.,  5.]],
#
#         [[ 6.,  7.,  8.],
#          [ 9., 10., 11.]]])
print(B)
# tensor([[[ 0.,  1.],
#          [ 2.,  3.],
#          [ 4.,  5.]],
#
#         [[ 6.,  7.],
#          [ 8.,  9.],
#          [10., 11.]]])


print(A[0])
# tensor([[0., 1., 2.],
#         [3., 4., 5.]])
print(B[0])
# tensor([[0., 1.],
#         [2., 3.],
#         [4., 5.]])

