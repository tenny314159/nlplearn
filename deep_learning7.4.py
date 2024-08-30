# 7.4 GRU网络进行情感分类


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time
import copy
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchtext import data
from torchtext.vocab import Vectors
from tqdm import tqdm

mytokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=mytokenize,  # torchtext.data.field.Field  dtype:torch.int64
                  include_lengths=True, use_vocab=True,
                  batch_first=True, fix_length=200)

LABEL = data.Field(sequential=False, use_vocab=False,  # torchtext.data.field.Field  dtype:torch.int64
                   pad_token=None, unk_token=None)
# 对所要读取的数据集的列进行处理
train_test_fields = [
    ("text", TEXT),
    ("label", LABEL)
]  # list 里面有两个元素，都是tuple

# 读取数据
traindata, testdata = data.TabularDataset.splits(  # traindata是一个torchtext.data.dataset.tabulardataset
    path="./", format="csv",
    train="imdb_train_preprocessed.csv", fields=train_test_fields,  #
    test="imdb_test_preprocessed.csv", skip_header=True
)

print(len(traindata), len(testdata))
# 25000 25000
vec = Vectors("glove.6B.100d.txt", './deep Learning')  # torchtext.vocab.vectors
# 使用训练集构建单词表，导入预先训练的词嵌入
TEXT.build_vocab(traindata, max_size=20000, vectors=vec)
LABEL.build_vocab(traindata)

# 训练集、验证集和测试集定义为迭代器
BATCH_SIZE = 32
train_iter = data.BucketIterator(traindata, batch_size=BATCH_SIZE)
test_iter = data.BucketIterator(testdata, batch_size=BATCH_SIZE)


class GRUNet(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim  # GRU神经元个数
        self.layer_dim = layer_dim  # GRU的层数
        # 对文本进行词项量处理
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM ＋ 全连接层
        self.gru = nn.GRU(embedding_dim, hidden_dim, layer_dim,
                          batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        embeds = self.embedding(x)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        r_out, h_n = self.gru(embeds, None)  # None 表示初始的 hidden state 为0
        # 选取最后一个时间点的out输出
        out = self.fc1(r_out[:, -1, :])
        return out


vocab_size = len(TEXT.vocab)
embedding_dim = vec.dim  # 词向量的维度
# embedding_dim = 128 #  词向量的维度
hidden_dim = 128
layer_dim = 1
output_dim = 2
grumodel = GRUNet(vocab_size, embedding_dim, hidden_dim, layer_dim, output_dim)
print(grumodel)
# GRUNet(
#   (embedding): Embedding(4, 100)
#   (gru): GRU(100, 128, batch_first=True)
#   (fc1): Sequential(
#     (0): Linear(in_features=128, out_features=128, bias=True)
#     (1): Dropout(p=0.5, inplace=False)
#     (2): ReLU()
#     (3): Linear(in_features=128, out_features=2, bias=True)
#   )
# )


grumodel.embedding.weight.data.copy_(TEXT.vocab.vectors)
# 将无法识别的词'<unk>', '<pad>'的向量初始化为0
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
grumodel.embedding.weight.data[UNK_IDX] = torch.zeros(vec.dim)
grumodel.embedding.weight.data[PAD_IDX] = torch.zeros(vec.dim)


def train_model(model, traindataloader, testdataloader, criterion,
                optimizer, num_epochs):
    train_loss_all = []
    train_acc_all = []
    test_loss_all = []
    test_acc_all = []
    learn_rate = []
    since = time.time()
    # 设置等间隔调整学习率,每隔step_size个epoch,学习率缩小10倍
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    for epoch in range(num_epochs):
        learn_rate.append(scheduler.get_lr()[0])
        print('-' * 10)
        print('Epoch {}/{},Lr:{}'.format(epoch, num_epochs - 1, learn_rate[-1]))
        # 每个epoch有两个阶段,训练阶段和验证阶段
        train_loss = 0.0
        train_corrects = 0
        train_num = 0
        test_loss = 0.0
        test_corrects = 0
        test_num = 0
        model.train()  # 设置模型为训练模式
        for step, batch in enumerate(tqdm(traindataloader)):
            textdata, target = batch.text[0], batch.label
            out = model(textdata)
            pre_lab = torch.argmax(out, 1)  # 预测的标签
            loss = criterion(out, target)  # 计算损失函数值
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(target)
            train_corrects += torch.sum(pre_lab == target.data)
            train_num += len(target)
        # 计算一个epoch在训练集上的损失和精度
        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_corrects.double().item() / train_num)
        print('{} Train Loss: {:.4f}  Train Acc: {:.4f}'.format(
            epoch, train_loss_all[-1], train_acc_all[-1]))
        scheduler.step()  # 更新学习率
        # 计算一个epoch的训练后在验证集上的损失和精度

        model.eval()  # 设置模型为训练模式评估模式
        for step, batch in enumerate(tqdm(testdataloader)):
            textdata, target = batch.text[0], batch.label
            out = model(textdata)
            pre_lab = torch.argmax(out, 1)
            loss = criterion(out, target)
            test_loss += loss.item() * len(target)
            test_corrects += torch.sum(pre_lab == target.data)
            test_num += len(target)
        # 计算一个epoch在训练集上的损失和精度
        test_loss_all.append(test_loss / test_num)
        test_acc_all.append(test_corrects.double().item() / test_num)
        print('{} Test Loss: {:.4f}  Test Acc: {:.4f}'.format(
            epoch, test_loss_all[-1], test_acc_all[-1]))

    train_process = pd.DataFrame(
        data={"epoch": range(num_epochs),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "test_loss_all": test_loss_all,
              "test_acc_all": test_acc_all,
              "learn_rate": learn_rate})
    return model, train_process


optimizer = optim.RMSprop(grumodel.parameters(), lr=0.003)
loss_func = nn.CrossEntropyLoss()  # 交叉熵作为损失函数
# 对模型进行迭代训练,对所有的数据训练EPOCH轮
grumodel, train_process = train_model(grumodel, train_iter, test_iter, loss_func, optimizer, num_epochs=8)


# 可视化模型训练过程中
plt.figure(figsize=(18, 6))
plt.subplot(1, 2, 1)
plt.plot(train_process.epoch, train_process.train_loss_all,
         "r.-", label="Train loss")
plt.plot(train_process.epoch, train_process.test_loss_all,
         "bs-", label="Test loss")
plt.legend()
plt.xlabel("Epoch number", size=13)
plt.ylabel("Loss value", size=13)
plt.subplot(1, 2, 2)
plt.plot(train_process.epoch, train_process.train_acc_all,
         "r.-", label="Train acc")
plt.plot(train_process.epoch, train_process.test_acc_all,
         "bs-", label="Test acc")
plt.xlabel("Epoch number", size=13)
plt.ylabel("Acc", size=13)
plt.legend()
plt.show()


grumodel.eval()  # 设置模型为训练模式评估模式
test_y_all = torch.LongTensor()
pre_lab_all = torch.LongTensor()
for step, batch in enumerate(test_iter):
    textdata, target = batch.text[0], batch.label.view(-1)
    out = grumodel(textdata)
    pre_lab = torch.argmax(out, 1)
    test_y_all = torch.cat((test_y_all, target))  # 测试集的标签
    pre_lab_all = torch.cat((pre_lab_all, pre_lab))  # 测试集的预测标签

acc = accuracy_score(test_y_all, pre_lab_all)
print("在测试集上的预测精度为:", acc)
#  在测试集上的预测精度为: 0.84936