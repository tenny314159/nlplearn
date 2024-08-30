# nn模块 做卷积

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

myim = Image.open("./123.jpg")
myimgray = np.array(myim.convert("L"), dtype=np.float32)  # 转为灰度图像，将结果转换为float32
plt.figure(figsize=(6, 6))
plt.imshow(myimgray, cmap=plt.cm.gray)
plt.axis("off")
plt.show()

imh, imw = myimgray.shape
myimgray_t = torch.from_numpy(myimgray.reshape((1, 1, imh, imw)))
print(myimgray_t.shape)
# torch.Size([1, 1, 512, 512]) batch, channel , height, width 转换格式才能继续做卷积操作


kersize = 5
ker = torch.ones(kersize, kersize, dtype=torch.float32) * -1  # 注意创建张量的代码写法，这里元素全部为-1
ker[2, 2] = 24  # 5 * 5 的矩阵的中心元素[2,2]为24
ker = ker.reshape((1, 1, kersize, kersize))  # 转换格式才能继续做卷积操作
conv2d = nn.Conv2d(1, 2, (kersize, kersize), bias=False)  # 定义卷积 ，channel为2
conv2d.weight.data[0] = ker  # 将定义的卷积层 conv2d 的第一个卷积核的权重设置为之前创建的  ker  张量。在卷积神经网络中，卷积层的权重参数就是卷积核，它用来提取输入数据的特征
imconv2dout = conv2d(myimgray_t)  # 卷积操作
imconv2dout_im = imconv2dout.data.squeeze()  # 压缩
print(imconv2dout_im.shape)
# torch.Size([2, 508, 508])


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(imconv2dout_im[0], cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(imconv2dout_im[1], cmap=plt.cm.gray)
plt.axis("off")
plt.show()

maxpool2 = nn.MaxPool2d(2, stride=2)  # 最大池化定义pool of square window of size=2
pool2_out = maxpool2(imconv2dout)
pool2_out_im = pool2_out.squeeze()
print(pool2_out.shape)
# torch.Size([1, 2, 254, 254])


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(pool2_out_im[0].data, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(pool2_out_im[1].data, cmap=plt.cm.gray)
plt.axis("off")
plt.show()

avgpool2 = nn.AvgPool2d(2, stride=2)  # 平均值池化
pool2_out = avgpool2(imconv2dout)
pool2_out_im = pool2_out.squeeze()
print(pool2_out.shape)
# torch.Size([1, 2, 254, 254])


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(pool2_out_im[0].data, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(pool2_out_im[1].data, cmap=plt.cm.gray)
plt.axis("off")
plt.show()

Adaavgpool2 = nn.AdaptiveAvgPool2d((100, 200))  # 自适应平均值池化
pool2_out = Adaavgpool2(imconv2dout)
pool2_out_im = pool2_out.squeeze()
print(pool2_out.shape)
# torch.Size([1, 2, 100, 200])


plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(pool2_out_im[0].data, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(pool2_out_im[1].data, cmap=plt.cm.gray)
plt.axis("off")
plt.show()


# 激活函数

x = torch.linspace(-6, 6, 100)
sigmoid = nn.Sigmoid()  # 定义Sigmoid激活函数
ysigmoid = sigmoid(x)
tanh = nn.Tanh()  # 定义Tanh激活函数
ytanh = tanh(x)
relu = nn.ReLU()  # 定义ReLU激活函数
yrelu = relu(x)
softplus = nn.Softplus()  # 定义Softplus激活函数
ysoftplus = softplus(x)


plt.figure(figsize=(14, 3))
plt.subplot(1, 4, 1)
plt.plot(x.data.numpy(), ysigmoid.data.numpy(), "r-")
plt.title("Sigmoid")
plt.grid()
plt.subplot(1, 4, 2)
plt.plot(x.data.numpy(), ytanh.data.numpy(), "r-")
plt.title("Tanh")
plt.grid()
plt.subplot(1, 4, 3)
plt.plot(x.data.numpy(), yrelu.data.numpy(), "r-")
plt.title("Relu")
plt.grid()
plt.subplot(1, 4, 4)
plt.plot(x.data.numpy(), ysoftplus.data.numpy(), "r-")
plt.title("Softplus")
plt.grid()
plt.show()
