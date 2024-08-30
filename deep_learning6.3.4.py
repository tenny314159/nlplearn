# 6.3.4 空洞卷积神经网络的搭建


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as Data
from torchvision import transforms
import hiddenlayer as hl
from torchvision.datasets import FashionMNIST
from sklearn.metrics import accuracy_score

train_data = FashionMNIST(
    root="./deep Learning/FashionMNIST",  # 数据的路径
    train=True,  # 只使用训练数据集
    # 将数据转化为torch使用的张量,取汁范围为［0，1］
    transform=transforms.ToTensor(),
    download=True  # 因为数据已经下载过，所以这里不再下载
)

test_data = FashionMNIST(root="./deep Learning/FashionMNIST", train=False)
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x, dim=1)
test_data_y = test_data.targets


class_label = train_data.classes
class_label[0] = 'T-shirt'



train_loader = Data.DataLoader(
    dataset=train_data,  # 使用的数据集
    batch_size=64,  # 批处理样本大小
    shuffle=True,  # 每次迭代前打乱数据
    num_workers=0,  # 使用两个进程
)


class MyConvdilaNet(nn.Module):
    def __init__(self):
        super(MyConvdilaNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,

                kernel_size=3,
                stride=1,
                padding=1,
                dilation=2
            ),
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, dilation=2),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 256),
            # nn.Linear(32*7*7,128),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)  # 展开卷积层
        output = self.classifier(x)
        return output


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


myconvdilanet = MyConvdilaNet()
print(myconvdilanet)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(myconvdilanet.parameters(), lr=0.0003)
myconvdilanet, train_process = train_model(myconvdilanet, train_loader, 0.8, criterion, optimizer, num_epochs=25)


plt.figure(figsize=(12, 4))
# 绘制损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_process['epoch'], train_process['train_loss'], 'ro-', label='Train loss')
plt.plot(train_process['epoch'], train_process['val_loss'], 'bs-', label='Val loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 绘制准确率曲线
plt.subplot(1, 2, 2)
plt.plot(train_process['epoch'], train_process['train_acc'], 'ro-', label='Train accuracy')
plt.plot(train_process['epoch'], train_process['val_acc'], 'bs-', label='Val accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 显示图表
plt.tight_layout()  # 可选，用于调整子图布局
plt.show()


myconvdilanet.eval()
output = myconvdilanet(test_data_x)
pre_lab = torch.argmax(output, 1)
acc = accuracy_score(test_data_y, pre_lab)
print('在测试集的精度', acc)


conf_mat = confusion_matrix(test_data_y, pre_lab)
df_cm = pd.DataFrame(conf_mat, index=class_label, columns=class_label)
heatmap = sns.heatmap(df_cm, annot=True, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right')
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# Epoch 24/24
# ----------
# 24  Train Loss: 0.2729, Train Acc: 0.8980,
# 24  Val Loss: 0.2767  ,   Val Acc: 0.8967
# save model
# Train and val complete in 21m 49s
# 在测试集的精度 0.8826


