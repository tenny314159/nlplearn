# 6.3 卷积神经网络识别fashion-MNIST

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as Data
from torchvision import transforms
import hiddenlayer as hl
from torchvision.datasets import FashionMNIST
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

train_data = FashionMNIST(
    root="./deep Learning/FashionMNIST",  # 数据的路径
    train=True,  # 只使用训练数据集
    # 将数据转化为torch使用的张量,取汁范围为［0，1］
    transform=transforms.ToTensor(),
    download=True  # 因为数据已经下载过，所以这里不再下载
)

# 定义一个数据加载器
train_loader = Data.DataLoader(
    dataset=train_data,  # 使用的数据集
    batch_size=64,  # 批处理样本大小
    shuffle=True,  # 每次迭代前打乱数据
    num_workers=0,  # 使用两个进程
)
print('train_loader的batch数量为：', len(train_loader))
# train_loader的batch数量为： 938


#  可视化训练数据集的一个batch的样本来查看图像内容
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break

# 输出训练图像的尺寸和标签的尺寸，都是torch格式的数据
print(b_x.shape)
print(b_y.shape)
# torch.Size([64, 1, 28, 28])
# torch.Size([64])

# 显示一个batch的图像
batch_x = b_x.squeeze().numpy()
batch_y = b_y.numpy()
class_label = train_data.classes
class_label[0] = 'T-shirt'
plt.figure(figsize=(12, 5))
for ii in np.arange(len(batch_y)):
    plt.subplot(4, 16, ii + 1)
    plt.imshow(batch_x[ii], cmap='gray')
    plt.title(class_label[batch_y[ii]], fontsize=9)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.05)

plt.show()  # 课本缺了一行show

test_data = FashionMNIST(root="./deep Learning/FashionMNIST", train=False)
test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x, dim=1)
test_data_y = test_data.targets
print(test_data_x.shape)
print(test_data_y.shape)
# torch.Size([10000, 1, 28, 28])
# torch.Size([10000])

class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,

                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2,
            ),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0),
            nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 6 * 6, 256),
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


myconvnet = MyConvNet()
print(myconvnet)


# MyConvNet(
#   (conv1): Sequential(
#     (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU()
#     (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
#   )
#   (conv2): Sequential(
#     (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1))
#     (1): ReLU()
#     (2): AvgPool2d(kernel_size=2, stride=2, padding=0)
#   )
#   (classifier): Sequential(
#     (0): Linear(in_features=1152, out_features=256, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=256, out_features=128, bias=True)
#     (3): ReLU()
#     (4): Linear(in_features=128, out_features=10, bias=True)
#   )
# )

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


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(myconvnet.parameters(), lr=0.0003)
myconvnet, train_process = train_model(myconvnet, train_loader, 0.8, criterion, optimizer, num_epochs=25)


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


# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(train_process.epoch, train_process.train_loss_all, 'ro-', label='Train loss')
# plt.plot(train_process.epoch, train_process.val_loss_all, 'bs-', label='Val loss')
# plt.legend()
# plt.xlabel('epoch')
# plt.ylabel('Loss')
# plt.subplot(1, 2, 2)
# plt.plot(train_process.epoch, train_process.train_acc_all, 'ro-', label='Train acc')
# plt.plot(train_process.epoch, train_process.val_acc_all, 'bs-', label='Val acc')
# plt.xlabel('epoch')
# plt.ylabel('acc')
# plt.legend()
# plt.show()


myconvnet.eval()
output = myconvnet(test_data_x)
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
# 24  Train Loss: 0.1940, Train Acc: 0.9273,
# 24  Val Loss: 0.1891  ,   Val Acc: 0.9288
# save model
# Train and val complete in 28m 33s
# 在测试集的精度 0.9051



# Traceback (most recent call last):
#   File "D:\pythoncode\learn\a\deep_learning6.3.py", line 208, in <module>
#     plt.plot(train_process.epoch, train_process.train_loss_all, 'ro-', label='train_loss')
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\pandas\core\generic.py", line 5141, in __getattr__
#     return object.__getattribute__(self, name)
# AttributeError: 'DataFrame' object has no attribute 'train_loss_all'


# plt.plot(train_process.epoch, train_process.train_loss_all, 'ro-', label='Train loss')
# plt.plot(train_process.epoch, train_process.val_loss_all, 'bs-', label='Val loss')
# 把上面的代码换成下面的这种，不知道课本当时为什么这么写。。。
# plt.plot(train_process['epoch'], train_process['train_acc'], 'ro-', label='Train accuracy')
# plt.plot(train_process['epoch'], train_process['val_acc'], 'bs-', label='Val accuracy')

