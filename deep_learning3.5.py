# 网络参数初始化

import torch
import torch.nn as nn

conv1 = torch.nn.Conv2d(3, 16, 3)

torch.manual_seed(12)

torch.nn.init.normal(conv1.weight, mean=0, std=1)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.hist(conv1.weight.data.numpy().reshape((-1, 1)), bins=30)
plt.show()

print(torch.nn.init.constant(conv1.bias, val=0.1))


# Parameter containing:
# tensor([0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
#         0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
#        requires_grad=True)


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        # 3: 表示输入通道数。在这个例子中，意味着该卷积层期望接收的输入数据有3个通道，这通常对应于RGB彩色图像，每个通道代表图像的一种颜色信息（红色、绿色、蓝色）。
        # 16: 表示输出通道数，也就是卷积操作后产生的特征图（Feature
        # Maps）的数量。这意味着该卷积层会通过学习，从输入图像中提取16种不同的特征。
        # 3: 指定卷积核（Kernel）的大小，这里是指一个3x3的卷积核。卷积核是在输入数据上滑动的小窗口，用于执行局部区域的加权和操作，从而提取特征。3
        # x3的卷积核是比较常用的，因为它可以在保持较高分辨率的同时，捕获局部特征。
        self.hidden = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
        )
        self.cla = nn.Linear(50, 10)
        self.cla = nn.Linear(50, 10)
        # 创建了一个从50维输入映射到10维输出的全连接层，并将其赋值给类或模块的成员变量self.cla。这个全连接层在神经网络模型的尾部经常用来进行分类预测，将提取到的特征转换为类别概率或直接的类别预测。

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(x.shape[0], -1)
        x = self.hidden(x)
        output = self.cla(x)
        return output


# Conv2D和Conv1D是深度学习中两种类型的卷积层，它们在处理不同类型的数据和应用场景上有所区别：
# Conv1D（一维卷积）
# 主要用途：一维卷积主要用于处理序列数据，如时间序列分析、自然语言处理（NLP）中的文本数据或音频信号。这类数据在结构上是一维的，即数据沿单一时间轴或序列轴排列。
# 输入结构：输入数据通常具有形状(batch_size, channels, sequence_length)，其中channels对应于特征的数量，sequence_length是序列的长度。
# 卷积核：一维卷积核也是线性的，形状为(out_channels, in_channels, kernel_size)，其中out_channels是输出特征图的数量，in_channels是输入特征的数量，kernel_size表示卷积核覆盖的序列上的元素数量。
# 应用实例：情感分析、文本分类、语音识别等。
# Conv2D（二维卷积）
# 主要用途：二维卷积主要应用于图像处理、视频帧分析以及其他类型的二维数据。它能够捕获空间结构信息，如图像中的纹理、边缘和形状。
# 输入结构：输入数据的形状通常是(batch_size, channels, height, width)，其中channels代表颜色通道（如RGB图像中的红、绿、蓝通道），height和width分别代表图像的高度和宽度。
# 卷积核：二维卷积核的形状为(out_channels, in_channels, kernel_height, kernel_width)，它在图像的宽度和高度上滑动，提取特征。
# 应用实例：图像分类、物体检测、图像分割、视频分析等。
# 主要区别
# 数据维度：最本质的区别在于处理数据的维度。Conv1D处理一维序列数据，而Conv2D处理二维图像或网格数据。
# 卷积核形状：相应地，卷积核的维度也不同，Conv1D的卷积核是一维的，而Conv2D的卷积核是二维的。
# 应用场景：由于数据维度的不同，它们的应用场景也有所侧重。Conv1D适用于处理序列数据相关的任务，而Conv2D则更适合图像处理和相关视觉任务。
# 特征提取：Conv1D更倾向于捕捉序列中的时间或顺序相关特征，而Conv2D则擅长于提取空间结构特征。
# 总的来说，选择Conv1D还是Conv2D取决于数据的性质和任务的需求。在某些特定情况下，通过适当的调整，Conv2D也可以处理看似一维的数据，但通常情况下，针对数据的维度选择合适的卷积类型会更高效和合适。


testnet = TestNet()
print(testnet)


# TestNet(
#   (conv1): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
#   (hidden): Sequential(
#     (0): Linear(in_features=100, out_features=100, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=100, out_features=50, bias=True)
#     (3): ReLU()
#   )
#   (cla): Linear(in_features=50, out_features=10, bias=True)
# )


def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal(m.weight, mean=0, std=0.5)
    if type(m) == nn.Linear:
        torch.nn.init.uniform(m.weight, a=-0.1, b=0.1)
        m.bias.data.fill_(0.01)


torch.manual_seed(13)
testnet.apply(init_weights)  # apply 方法，对所有可训练参数应用初始化权重函数

