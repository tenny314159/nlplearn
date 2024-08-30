# 4.2 训练过程的可视化


import torch.nn as nn

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        self.fc = nn.Sequential(nn.Linear(in_features=32 * 7 * 7, out_features=128), nn.ReLU(), nn.Linear(128, 64),
                                nn.ReLU())
        self.out = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output = self.out(x)
        return output


MyConvnet = ConvNet()


from tensorboardX import SummaryWriter
import torch

SumWriter = SummaryWriter(log_dir='deep Learning/log')
optimizer = torch.optim.Adam(MyConvnet.parameters(), lr=0.0003)



import hiddenlayer as hl
import time
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data
import torchvision.transforms as transforms
import numpy as np

train_data = FashionMNIST(root="./deep Learning/",
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=128,
                               shuffle=True,
                               )

print(len(train_loader))
# 469

test_data = FashionMNIST(root='./deep Learning/',
                         train=False,
                         download=False)
print(len(test_data))
# 10000

test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
test_data_x = torch.unsqueeze(test_data_x, dim=1)
test_data_y = test_data.targets
print(test_data_x.shape)
# torch.Size([10000, 1, 28, 28])
print(test_data_y.shape)


# torch.Size([10000])

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, 2))
        # self.fc = nn.Sequential(nn.Linear(in_features=32 * 7 * 7, out_features=128), nn.ReLU(), nn.Linear(128, 64),
        #                         nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(in_features=32 * 6 * 6, out_features=128), nn.ReLU(), nn.Linear(128, 64),
                                nn.ReLU())
        self.out = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        output = self.out(x)
        return output


MyConvnet = ConvNet()
optimizer = torch.optim.Adam(MyConvnet.parameters(), lr=0.0003)
loss_func = nn.CrossEntropyLoss()
history1 = hl.History()
canvas1 = hl.Canvas()
print_step = 100

print(MyConvnet.fc[2].weight)
print(MyConvnet.fc[2].weight.shape)
# Parameter containing:
# tensor([[ 0.0166,  0.0044,  0.0586,  ..., -0.0650, -0.0125, -0.0475],
#         [ 0.0823,  0.0646, -0.0560,  ...,  0.0881,  0.0576, -0.0146],
#         [-0.0416,  0.0772, -0.0110,  ...,  0.0261,  0.0081,  0.0283],
#         ...,
#         [-0.0356,  0.0712, -0.0448,  ..., -0.0462, -0.0544,  0.0026],
#         [-0.0192,  0.0113, -0.0848,  ..., -0.0070,  0.0719,  0.0529],
#         [-0.0513, -0.0832, -0.0089,  ...,  0.0436, -0.0239, -0.0772]],
#        requires_grad=True)
# torch.Size([64, 128])

for epoch in range(5):

    for step, (b_x, b_y) in enumerate(train_loader):
        output = MyConvnet(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % print_step == 0:
            # 记录训练集上的精度
            output = MyConvnet(test_data_x)
            _, pre_lab = torch.max(output, 1)
            acc = accuracy_score(test_data_y, pre_lab)

            history1.log((epoch, step),
                         train_loss=loss,
                         test__acc=acc,
                         hidden_weight=MyConvnet.fc[2].weight
                         #hidden_weight=MyConvnet.fc[2].weight.squeeze(dim=0)
                         )

            with canvas1:
                canvas1.draw_plot(history1['train_loss'])
                canvas1.draw_plot(history1['test_acc'])
                #canvas1.draw_plot(history1['hidden_weight'])




# 尝试解决成功，换一个高版本的python
# protobuf requires Python '>=3.7' but the running Python is 3.6.13
