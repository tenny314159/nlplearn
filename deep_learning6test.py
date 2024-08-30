import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import MNIST

train_data = MNIST(
    root="/deep Learning",
    train="True",
    transform=transforms.ToTensor(),
    download=True
)

train_loader = Data.DataLoader(
    dataset=train_data,  # 使用FashionMNIST数据集
    batch_size=64,  # 批处理样本大小
    shuffle=False,  # 每次迭代不打乱顺序，有利于我们切分为训练集和验证集
    num_workers=0,
)

test_data = MNIST(
    root="deep Learning",
    train=False,
    download=True
)

test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x, dim=1)  # 在指定位置 dim 插入一个大小为1的维度
test_data_y = test_data.targets
print("test_data_x.shape:", test_data_x.shape)
print("test_data_y.shape:", test_data_y.shape)


# 这里我们重新定义一个空洞卷积神经网络
class myconvdilanet(nn.Module):
    def __init__(self):
        super(myconvdilanet, self).__init__()
        # 定义第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, dilation=2),
            # 1：输入通道数（input channels），表示输入特征图的深度或数量。
            # 16：输出通道数（output channels），表示卷积后生成的特征图的深度或数量。
            # 3：卷积核的大小（kernel size），这里是 3x3 的卷积核。
            # 1：步幅（stride），表示卷积核在输入特征图上移动的步长。
            # 1：填充（padding），表示在输入特征图的边缘添加的像素，通常用于保持特征图的尺寸。
            # dilation=2：膨胀率（dilation），表示卷积核的膨胀系数，用于增加卷积核的感受野，同时增加计算量。
            nn.ReLU(),  # 激活函数
            nn.AvgPool2d(2, 2),
        )
        # 定义第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, dilation=2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.AvgPool2d(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


Myconvdilanet = myconvdilanet()


def train_model(model, traindataloader, train_rate, criterion, optimizer, num_epochs):
    batch_num = len(traindataloader)
    train_batch_num = round(train_rate * batch_num)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        val_loss = 0
        val_corrects = 0
        val_num = 0
        for step, (b_x, b_y) in enumerate(traindataloader):
            if step < train_batch_num:
                model.train()
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * b_x.size(0)
                train_corrects += torch.sum(pre_lab == b_y.data)
                train_num += b_x.size(0)
            else:
                model.eval()
                output = model(b_x)
                pre_lab = torch.argmax(output, 1)
                loss = criterion(output, b_y)
                val_loss += loss.item() * b_x.size(0)
                val_corrects += torch.sum(pre_lab == b_y.data)
                val_num += b_x.size(0)
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_corrects.double().item() / val_num)
        print('{}  Train Loss: {:.4f}, Train Acc: {:.4f},'.format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print('{}  Val Loss: {:.4f}  ,   Val Acc: {:.4f}'.format(epoch, val_loss_all[-1], val_acc_all[-1]))
        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())
            print('save model')
        time_use = time.time() - since
        print('Train and val complete in {:.0f}m {:.0f}s'.format(time_use // 60, time_use % 60))

    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame(data={'epoch': range(num_epochs),
                                       'train_loss': train_loss_all, 'train_acc': train_acc_all,
                                       'val_loss': val_loss_all, 'val_acc': val_acc_all})
    return model, train_process


optimizer = torch.optim.Adam(Myconvdilanet.parameters(), lr=0.0003)
criterion = nn.CrossEntropyLoss()
Myconvdilanet, train_process = train_model(
    Myconvdilanet, train_loader, 0.8,
    criterion, optimizer, num_epochs=25
)
