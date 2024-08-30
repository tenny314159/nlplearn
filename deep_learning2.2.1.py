# 学习代码是 pytorch 深度学习入门与实战 孙玉林 余本国 中国水利水电出版社


# 本地编译器环境
# python 3.10
# torch                     2.2.1+cu118              pypi_0    pypi
# torchaudio                2.2.1+cu118              pypi_0    pypi
# torchvision               0.17.1+cu118             pypi_0    pypi


# torch的数据结构

import torch

print(torch.tensor([1.2, 2.4]).dtype)
# torch.float32

torch.set_default_tensor_type(torch.DoubleTensor)
print(torch.tensor([1.2, 2.4]).dtype)  # CPU tensor中的DoubleTensor对应dtype中的torch.float64
# torch.float64


a = torch.tensor([1.2, 2.4])  # 强制类型转换
print('a.dtype', a.dtype)
# a.dtype torch.float64
print(a.long().dtype)
# torch.int64
print(a.int().dtype)
#  torch.int32
print(a.float().dtype)
#   torch.float32
print(a.double().dtype)
#    torch.float64


torch.set_default_tensor_type(torch.FloatTensor)  # 转回去
print(torch.tensor([1.2, 2.4]).dtype)
# torch.float32


print('get_default_dtype', torch.get_default_dtype())  # 获取默认的数据类型
# get_default_dtype torch.float32
