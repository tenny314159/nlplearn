# 基本tensor创建

import torch

A = torch.tensor([[1, 2], [3, 4]])
print(A.numel())  # numel() 方法返回张量中的元素数量 number of elements
# 4


B = torch.tensor((1, 2, 3), dtype=torch.float32, requires_grad=True)  # 注意只有浮点类型可以计算梯度
print(B)
# tensor([1., 2., 3.], requires_grad=True)

y = B.pow(2).sum()
print(y)
# tensor(14., grad_fn=<SumBackward0>)

y.backward()
print(B.grad)
# tensor([2., 4., 6.])


# 除了 torch.tensor() 外还有  torch.Tensor() 可构造张量
D = torch.Tensor(2, 3)  # 特定尺寸
print(D)
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])
print(torch.ones_like(D))  # 全1
# tensor([[1., 1., 1.],
#         [1., 1., 1.]])
print(torch.zeros_like(D))  # 全0
# tensor([[0., 0., 0.],
#         [0., 0., 0.]])
print(torch.rand_like(D)) # 随机数
# tensor([[0.1927, 0.3265, 0.9107],
#         [0.6054, 0.7528, 0.0278]])


E = [[1,2],[3,4]]
print(E)
E = D.new_tensor(E)
print(E)
print(D) #D 和 E各自是什么

# tensor([[1., 2.],
#         [3., 4.]])
# tensor([[-8.8013e+12,  1.1799e-42,  0.0000e+00],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00]])

print(E.dtype)
print(D.dtype)  # tensor.new_tensor()函数的具体作用，数据类型相同

# torch.float32
# torch.float32

E = D.new_full((3,3),fill_value=2)  # 3*3 使用2 填充的张量
print(E)

# tensor([[2., 2., 2.],
#         [2., 2., 2.],
#         [2., 2., 2.]])


import numpy as np
F = np.ones((3,3))
print(F)
# [[1. 1. 1.]
#  [1. 1. 1.]
#  [1. 1. 1.]]


Ftensor = torch.as_tensor(F) # numpy 与 pytorch张量的互换
print(Ftensor)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64)


Ftensor = torch.from_numpy(F) # numpy 与 pytorch张量的互换
print(Ftensor)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]], dtype=torch.float64)


torch.manual_seed(123)
A = torch.normal(mean=0.0, std=torch.tensor(1.0))
print(A)
# tensor(-0.1115)


A = torch.normal(mean=torch.arange(1 , 5.0), std=torch.arange(1.0 , 5))  #mean随机数的均值，std随机数的标准差
print(A)
# tensor([ 1.1204,  1.2607,  2.2787, -0.7877])


B = torch.rand(3,4) # 在【0，1】的均匀分布
print(B)
# tensor([[0.0756, 0.1966, 0.3164, 0.4017],
#         [0.1186, 0.8274, 0.3821, 0.6605],
#         [0.8536, 0.5932, 0.6367, 0.9826]])

C = torch.randperm(10) # 随机排序后输出
print(C)
# tensor([9, 1, 7, 6, 3, 4, 5, 8, 0, 2])

D = torch.logspace(start=0.1, end=1.0, steps=5) # 以对数为间隔的张量
print(D)
# tensor([ 1.2589,  2.1135,  3.5481,  5.9566, 10.0000])

