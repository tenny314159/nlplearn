# 自动微分

import torch
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = torch.sum(x ** 2 + x * 2 + 1)

print(x.requires_grad)
print(y.requires_grad)
# True
# True
print(x)
# tensor([[1., 2.],
#         [3., 4.]], requires_grad=True)

print(y)
# tensor(54., grad_fn=<SumBackward0>)

y.backward()
print(x.grad)  # 导数结果是 2*x + 2 ， 即对应梯度
# tensor([[ 4.,  6.],
#         [ 8., 10.]])


