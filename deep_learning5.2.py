# 5.2 MLP分类模型
# 使用未预处理的数据训练模型

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
print(spam.head())
#       0  0.64  0.64.1  0.1  0.32   0.2  ...   0.43   0.44  3.756   61   278  1
# 0  0.21  0.28    0.50  0.0  0.14  0.28  ...  0.180  0.048  5.114  101  1028  1
# 1  0.06  0.00    0.71  0.0  1.23  0.19  ...  0.184  0.010  9.821  485  2259  1
# 2  0.00  0.00    0.00  0.0  0.63  0.00  ...  0.000  0.000  3.537   40   191  1
# 3  0.00  0.00    0.00  0.0  0.63  0.00  ...  0.000  0.000  3.537   40   191  1
# 4  0.00  0.00    0.00  0.0  1.85  0.00  ...  0.000  0.000  3.000   15    54  1
#
# [5 rows x 58 columns]


df = pd.DataFrame(spam)

# 获取DataFrame的最后一列
last_column = df.iloc[:, -1]

# 统计最后一列中值为1和值为0的数量
count_1 = last_column.eq(1).sum()
count_0 = last_column.eq(0).sum()

print(f"值为1的数量：{count_1}")
print(f"值为0的数量：{count_0}")
# 值为1的数量：1812
# 值为0的数量：2788


X = spam.iloc[:, 0:57].values
# y = spam.label.values
y = spam.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123
)

scales = MinMaxScaler(feature_range=(0, 1))
X_train_s = scales.fit_transform(X_train)
X_test_s = scales.transform(X_test)

colname = spam.columns.values[:-1]
plt.figure(figsize=(20, 14))
for ii in range(len(colname)):
    plt.subplot(7, 9, ii + 1)
    sns.boxplot(x=y_train, y=X_train_s[:, ii])

    plt.title(colname[ii])
plt.subplots_adjust(hspace=0.4)
plt.show()


# 全连接网络
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
print(mlpc)
# MLPclassifica(
#   (hidden1): Sequential(
#     (0): Linear(in_features=57, out_features=30, bias=True)
#     (1): ReLU()
#   )
#   (hidden2): Sequential(
#     (0): Linear(in_features=30, out_features=10, bias=True)
#     (1): ReLU()
#   )
#   (classifica): Sequential(
#     (0): Linear(in_features=10, out_features=2, bias=True)
#     (1): Sigmoid()
#   )
# )

x = torch.randn(1, 57).requires_grad_(True)
y = mlpc(x)
Mymlpcvis = make_dot(y, params=dict(list(mlpc.named_parameters()) + [('x', x)]))

Mymlpcvis.format = "png"  # 形式转化为png,默认pdf
# 指定文件保存位置
Mymlpcvis.directory = "D:\pythoncode\learn/a\deep Learning\spambase/"
Mymlpcvis.view()  # 会自动生成文件

# 使用未处理的数据训练模型
X_train_nots = torch.from_numpy(X_train.astype(np.float32))  # 带s是scale过后的，不带s是没有经过标准化处理的
y_train_t = torch.from_numpy(y_train.astype(np.int64))
X_test_nots = torch.from_numpy(X_test.astype(np.float32))
y_test_t = torch.from_numpy(y_test.astype(np.int64))

# 将训练集转化为张量后,使用TensorDataset将X和Y整理到一起
train_data_nots = Data.TensorDataset(X_train_nots, y_train_t)

train_loader = Data.DataLoader(
    dataset=train_data_nots,
    batch_size=64,
    shuffle=True,
    num_workers=0,  # 不使用进程
)
print(len(train_loader))
# 54


optimizer = torch.optim.Adam(mlpc.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()  # 二分类损失函数
# 记录训练过程的指标
history1 = hl.History()
# 使用Canvas进行可视化
canvas1 = hl.Canvas()
print_step = 25
# 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(15):
    ## 对训练数据的迭代器进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):
        ## 计算每个batch的
        _, _, output = mlpc(b_x)  # MLP在训练batch上的输出
        train_loss = loss_func(output, b_y)  # 二分类交叉熵损失函数
        optimizer.zero_grad()  # 每个迭代步的梯度初始化为0
        train_loss.backward()  # 损失的后向传播，计算梯度
        optimizer.step()  # 使用梯度进行优化
        niter = epoch * len(train_loader) + step + 1

        ## 计算每经过print_step次迭代后的输出
        if niter % print_step == 0:
            _, _, output = mlpc(X_test_nots)
            _, pre_lab = torch.max(output, 1)
            test_accuracy = accuracy_score(y_test_t, pre_lab)
            # 为history添加epoch，损失和精度
            history1.log(niter, train_loss=train_loss,
                         test_accuracy=test_accuracy)
            # 使用两个图像可视化损失函数和精度
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["test_accuracy"])

