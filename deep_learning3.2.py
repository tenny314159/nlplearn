# pytorch 中的优化器

import torch
import torch.nn as nn


class TestNet(nn.Module):
    def __init__(self):
        super(TestNet, self).__init__()
        self.hidden = nn.Sequential(nn.Linear(10, 10), nn.ReLU())
        self.regression = nn.Linear(10, 1)

    def forward(self, x):
        x = self.hidden(x)
        output = self.regression(x)
        return output


testnet = TestNet()
print(testnet)
# TestNet(
#   (hidden): Sequential(
#     (0): Linear(in_features=13, out_features=10, bias=True)
#     (1): ReLU()
#   )
#   (regression): Linear(in_features=10, out_features=1, bias=True)
# )

from torch.optim import Adam

optimizer = Adam(testnet.parameters(), lr=0.001)

# 书本代码示例 ， 没有结合具体场景 ， 下面结合具体场景
# for input, target in dataset:
#     optimizer.zero_grad()
#     output = testnet(input)
#     loss = nn.MSELoss()(output,target)
#     loss.backward()
#     optimizer.step()


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset  # 假设数据集已经转换为TensorDataset形式
from torch.optim import Adam
import numpy as np
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

X = diabetes.data  # 特征数据
Y = diabetes.target  # 目标数据

train_xt = torch.from_numpy(X.astype(np.float32))
train_yt = torch.from_numpy(Y.astype(np.float32))

# 假设这是你的数据集，包含输入(input)和目标(target)的张量对
dataset = TensorDataset(train_xt, train_yt)

# 实例化网络、损失函数和优化器
criterion = nn.MSELoss()  # 回归使用均方误差损失函数

# 假设已经有了一个数据加载器，用于批量处理数据
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练循环
num_epochs = 10  # 训练轮数
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in dataloader:  # 使用DataLoader进行迭代
        optimizer.zero_grad()  # 清零梯度
        # 前向传播
        outputs = testnet(inputs)
        targets = targets.unsqueeze(1)
        # 计算损失
        loss = criterion(outputs, targets)
        # 反向传播和优化
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新权重
        # 记录损失
        running_loss += loss.item()

    # 输出每轮训练的平均损失
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}')

print("Training finished.")

# Epoch [1/10], Loss: 29159.409319196428
# Epoch [2/10], Loss: 29114.459123883928
# Epoch [3/10], Loss: 29081.398297991072
# Epoch [4/10], Loss: 29000.051897321428
# Epoch [5/10], Loss: 29023.331612723214
# Epoch [6/10], Loss: 28958.711774553572
# Epoch [7/10], Loss: 28973.793108258928
# Epoch [8/10], Loss: 28933.306222098214
# Epoch [9/10], Loss: 29034.154157366072
# Epoch [10/10], Loss: 29012.052873883928
# Training finished.
