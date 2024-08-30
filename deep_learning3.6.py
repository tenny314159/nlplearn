# PYTORCH中定义网络的方式

from sklearn.preprocessing import StandardScaler
import torch
import numpy as np
import torch.utils.data as Data
import pandas as pd
# from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import SGD

# 针对回归，换成另外一个回归数据集
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

X = diabetes.data  # 特征数据
Y = diabetes.target  # 目标数据

# 查看特征数据和目标数据的形状
print("特征数据形状:", X.shape)
print("目标数据形状:", Y.shape)
# 特征数据形状: (442, 10)
# 目标数据形状: (442,)

plt.figure()
plt.hist(Y, bins=20)
plt.show()

ss = StandardScaler(with_mean=True, with_std=True)

diabetes_Xs = ss.fit_transform(X)
diabetes_Ys = Y

train_xt = torch.from_numpy(diabetes_Xs.astype(np.float32))
train_yt = torch.from_numpy(diabetes_Ys.astype(np.float32))

train_data = Data.TensorDataset(train_xt, train_yt)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=128,
    shuffle=True,
    # num_workers=1  不用多线程
)


class MLPmodel(nn.Module):
    def __init__(self):
        super(MLPmodel, self).__init__()
        self.hidden1 = nn.Linear(
            in_features=10,
            # in_features=13,
            out_features=10,
            bias=True
        )
        self.active1 = nn.ReLU()
        self.hidden2 = nn.Linear(10, 10)
        self.active2 = nn.ReLU()
        self.regression = nn.Linear(10, 1)

    def forward(self, x):
        x = self.hidden1(x)
        x = self.active1(x)
        x = self.hidden2(x)
        x = self.active2(x)
        output = self.regression(x)
        return output


mlp1 = MLPmodel()
print(mlp1)
# MLPmodel(
#   (hidden1): Linear(in_features=13, out_features=10, bias=True)
#   (active1): ReLU()
#   (hidden2): Linear(in_features=10, out_features=10, bias=True)
#   (active2): ReLU()
#   (regression): Linear(in_features=10, out_features=1, bias=True)
# )

optimizer = SGD(mlp1.parameters(), lr=0.001)
loss_func = nn.MSELoss()
train_loss_all = []

for epoch in range(30):

    for step, (b_x, b_y) in enumerate(train_loader):
        output = mlp1(b_x).flatten()
        train_loss = loss_func(output, b_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_loss_all.append(train_loss.item())

plt.figure()
plt.plot(train_loss_all, 'r-')
plt.title('train loss')
plt.show()


class MLPmodedl2(nn.Module):
    def __init__(self):
        super(MLPmodedl2, self).__init__()
        self.hidden = nn.Sequential(
            # nn.Linear(13,10)
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        )
        self.regression = nn.Linear(10, 1)

    def forward(self, x):
        x = self.hidden(x)
        output = self.regression(x)
        return output


mlp2 = MLPmodedl2()
print(mlp2)

optimizer = SGD(mlp2.parameters(), lr=0.001)
loss_func = nn.MSELoss()
train_loss_all = []

for epoch in range(30):

    for step, (b_x, b_y) in enumerate(train_loader):
        output = mlp2(b_x).flatten()
        train_loss = loss_func(output, b_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_loss_all.append(train_loss.item())

plt.figure()
plt.plot(train_loss_all, 'r-')
plt.title('train loss')
plt.show()

#模型的保存
torch.save(mlp2,'./deep Learning/mlp2.pkl')
mlp2load = torch.load('./deep Learning/mlp2.pkl')
print(mlp2load)
# MLPmodedl2(
#   (hidden): Sequential(
#     (0): Linear(in_features=10, out_features=10, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=10, out_features=10, bias=True)
#     (3): ReLU()
#   )
#   (regression): Linear(in_features=10, out_features=1, bias=True)
# )


# 只保存模型的参数
torch.save(mlp2.state_dict(),'./deep Learning/mlp2.pkl')
mlp2param = torch.load('./deep Learning/mlp2.pkl')
print(mlp2param)
# OrderedDict([('hidden.0.weight', tensor([[ -0.9232,  -1.2095,  -1.7874,  -2.4585,  -2.9473,  -2.6885,   1.8891,
#           -3.7561,  -3.1867,  -3.1027],
#         [ -0.2042,  -0.0691,  -0.4388,  -0.3634,  -0.3157,  -0.4386,   0.1197,
#           -0.3744,  -0.4468,  -0.2533],
#         [ -2.7624,  -4.0176,  -4.5567,  -5.7148,  -7.7913,  -7.2320,   5.5492,
#          -10.1651,  -8.1563,  -7.5583],
#         [  0.0754,  -0.1642,  -0.2635,  -0.1018,  -0.1627,  -0.2434,   0.0623,
#            0.0514,  -0.1754,  -0.2588],
#         [ -0.7921,  -2.4244,  -2.5805,  -2.9983,  -4.5993,  -4.1026,   3.2293,
#           -5.5808,  -4.6215,  -4.0749],
#         [ -0.9643,  -1.4268,  -1.9347,  -1.8638,  -2.4141,  -2.5897,   2.0014,
#           -3.9987,  -2.6508,  -2.5059],
#         [ -0.3963,  -0.2835,  -0.4565,  -0.8690,  -0.8343,  -0.4606,   0.6018,
#           -0.6620,  -0.6662,  -0.4508],
#         [ -0.1957,  -0.0131,  -0.1863,  -0.3652,  -0.1023,  -0.3196,   0.1042,
#           -0.2538,  -0.3476,  -0.2444],
#         [  0.1132,   0.0381,  -0.1355,   0.0587,  -0.1994,   0.2194,   0.0137,
#           -0.1489,  -0.1327,  -0.1354],
#         [ -1.8380,  -2.7347,  -3.7258,  -4.0203,  -6.0127,  -5.8283,   3.2918,
#           -7.1877,  -5.3882,  -4.5944]])), ('hidden.0.bias', tensor([-3.4781, -0.3689, -6.9135, -0.1291, -3.7820, -0.8013, -0.1081, -0.1459,
#         -0.1595, -5.0331])), ('hidden.2.weight', tensor([[-1.8987e-01, -2.1553e-01, -2.4452e-01, -1.2826e-01, -4.4600e-01,
#          -3.9230e-02, -1.3994e-01,  2.4188e-01, -1.8353e-01, -3.7245e-02],
#         [-1.2330e+00,  1.4935e-01, -2.1006e+00,  2.2815e-01, -1.3141e+00,
#          -4.9419e-01, -2.0292e-01, -3.7244e-02, -1.8308e-02, -1.7430e+00],
#         [-2.9740e+00,  2.6738e-01, -6.7490e+00,  3.9170e-01, -3.8174e+00,
#          -6.6887e-01,  8.7731e-02, -1.1104e-01, -2.6860e-01, -4.4536e+00],
#         [-3.8733e-01,  1.7222e-01, -6.1735e-01,  1.8737e-01, -4.5800e-01,
#          -1.2687e-01, -2.8769e-01,  3.4088e-02,  1.3322e-01, -5.5480e-01],
#         [-2.8074e-01,  1.0812e-01,  7.2275e-02, -1.6865e-01, -2.5029e-01,
#          -2.1132e-02,  1.5517e-01,  9.2169e-02,  9.1193e-02, -4.0287e-02],
#         [-1.1228e+00, -9.4796e-02, -1.7891e+00, -1.0016e-01, -8.4967e-01,
#          -4.5255e-01, -2.8438e-01,  6.7924e-02, -4.4027e-02, -1.0355e+00],
#         [-7.0236e+00,  2.0964e-01, -1.5519e+01,  7.8860e-03, -7.8425e+00,
#          -1.5617e+00,  4.7506e-01,  3.0627e-01,  4.2218e-02, -1.0231e+01],
#         [ 1.0062e-01,  1.9500e-01,  1.1216e-02,  2.4669e-01, -2.5071e-02,
#           2.0878e-01, -3.1347e-01, -8.7177e-02,  8.6670e-02, -2.5709e-01],
#         [-4.4382e+00,  5.1471e-01, -9.3459e+00, -6.2396e-02, -4.8647e+00,
#          -8.5413e-01,  8.7904e-02,  2.1781e-01, -1.4455e-01, -6.1295e+00],
#         [-1.5022e+00,  8.5380e-02, -3.0917e+00,  2.0259e-01, -2.0209e+00,
#          -9.6872e-02, -1.2980e-01, -5.7904e-02,  9.4876e-02, -2.2930e+00]])), ('hidden.2.bias', tensor([-0.2126, -0.2515, -1.5094, -0.2702, -0.2527, -0.0605, -2.9485, -0.1605,
#         -1.4649, -0.6159])), ('regression.weight', tensor([[ -0.0986,  -2.6625,  -8.5537,   0.1091,  -0.1195,  -2.5662, -18.3474,
#            0.0404, -10.5618,  -3.9268]])), ('regression.bias', tensor([31.6779]))])

