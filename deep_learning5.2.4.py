# 5.2.4 使用预处理后的数据训练模型


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.utils.data as Data

import matplotlib.pyplot as plt
import seaborn as sns
import hiddenlayer as hl
from torchviz import make_dot

spam = pd.read_csv("D:\pythoncode\learn/a\deep Learning\spambase/spambase.csv")
X = spam.iloc[:, 0:57].values
# y = spam.label.values
y = spam.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123
)

scales = MinMaxScaler(feature_range=(0, 1))
X_train_s = scales.fit_transform(X_train)
X_test_s = scales.transform(X_test)

X_train_t = torch.from_numpy(X_train_s.astype(np.float32))
y_train_t = torch.from_numpy(y_train.astype(np.int64))
X_test_t = torch.from_numpy(X_test_s.astype(np.float32))
y_test_t = torch.from_numpy(y_test.astype(np.int64))

# 将训练集转化为张量后,使用TensorDataset将X和Y整理到一起
train_data = Data.TensorDataset(X_train_t, y_train_t)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0,  # 不使用进程
)


class MLPclassifica(nn.Module):
    def __init__(self):
        super(MLPclassifica, self).__init__()
        # 定义第一个隐藏层
        self.hidden1 = nn.Sequential(
            nn.Linear(
                in_features=57,  # 第一个隐藏层的输入，数据的特征数
                out_features=30,  # 第一个隐藏层的输出，神经元的数量
                bias=True,  # 默认会有偏置
            ),
            nn.ReLU()
        )
        # 定义第二个隐藏层
        self.hidden2 = nn.Sequential(
            nn.Linear(30, 10),
            nn.ReLU()
        )
        # 分类层
        self.classifica = nn.Sequential(
            nn.Linear(10, 2),
            nn.Sigmoid()
        )

    # 定义网络的向前传播路径
    def forward(self, x):
        fc1 = self.hidden1(x)
        fc2 = self.hidden2(fc1)
        output = self.classifica(fc2)
        # 输出为两个隐藏层和输出层的输出
        return fc1, fc2, output


mlpc = MLPclassifica()

optimizer = torch.optim.Adam(mlpc.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()  # 二分类损失函数
# 记录训练过程的指标
history1 = hl.History()
# 使用Canvas进行可视化
canvas1 = hl.Canvas()
print_step = 25
# 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(15):
    # 对训练数据的迭代器进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):

        _, _, output = mlpc(b_x)  # MLP在训练batch上的输出
        train_loss = loss_func(output, b_y)  # 二分类交叉熵损失函数
        optimizer.zero_grad()  # 每个迭代步的梯度初始化为0
        train_loss.backward()  # 损失的后向传播，计算梯度
        optimizer.step()  # 使用梯度进行优化
        niter = epoch * len(train_loader) + step + 1

        # 计算每经过print_step次迭代后的输出
        if niter % print_step == 0:
            _, _, output = mlpc(X_test_t)
            _, pre_lab = torch.max(output, 1)
            test_accuracy = accuracy_score(y_test_t, pre_lab)
            # 为history添加epoch，损失和精度
            history1.log(niter, train_loss=train_loss,
                         test_accuracy=test_accuracy)
            # 使用两个图像可视化损失函数和精度
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["test_accuracy"])

_, _, output = mlpc(X_test_t)
_, pre_lab = torch.max(output, 1)
test_accuracy = accuracy_score(y_test_t, pre_lab)
print("test_accuracy:", test_accuracy)
print(classification_report(y_test_t, pre_lab))
print(confusion_matrix(y_test_t, pre_lab))
# test_accuracy: 0.9347826086956522
#               precision    recall  f1-score   support
#
#            0       0.93      0.96      0.95       696
#            1       0.94      0.90      0.92       454
#
#     accuracy                           0.93      1150
#    macro avg       0.93      0.93      0.93      1150
# weighted avg       0.93      0.93      0.93      1150
#
# [[668  28]
#  [ 47 407]]


# 获取中间层的输出并可视化
_, test_fc2, _ = mlpc(X_test_t)
print("test_fc2.shape:", test_fc2.shape)
# test_fc2.shape: torch.Size([1150, 10])


# 使用散点图进行可视化
# 对输出进行降维并可视化
test_fc2_tsne = TSNE(n_components=2).fit_transform(test_fc2.data.numpy())

# 将特征进行可视化
plt.figure(figsize=(8, 6))
# 可视化前设置坐标系的取值范围
plt.xlim([min(test_fc2_tsne[:, 0] - 1), max(test_fc2_tsne[:, 0]) + 1])
plt.ylim([min(test_fc2_tsne[:, 1] - 1), max(test_fc2_tsne[:, 1]) + 1])
plt.plot(test_fc2_tsne[y_test == 0, 0], test_fc2_tsne[y_test == 0, 1],
         "bo", label="0")
plt.plot(test_fc2_tsne[y_test == 1, 0], test_fc2_tsne[y_test == 1, 1],
         "rd", label="1")
plt.legend()
plt.title("test_fc2_tsne")
plt.show()

activation = {}  # 保存不同层的输出


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


mlpc.classifica.register_forward_hook(get_activation("classifica"))
_, _, _ = mlpc(X_test_t)
classifica = activation["classifica"].data.numpy()
print("classifica.shape:", classifica.shape)
# classifica.shape: (1150, 2)


plt.figure(figsize=(8, 6))
# 可视化前设置坐标系的取值范围
plt.plot(classifica[y_test == 0, 0], classifica[y_test == 0, 1],
         "bo", label="0")
plt.plot(classifica[y_test == 1, 0], classifica[y_test == 1, 1],
         "rd", label="1")
plt.legend()
plt.title("classifica")
plt.show()

