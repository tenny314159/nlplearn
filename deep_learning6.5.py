# 6.5  卷积神经网络进行情感分类


# 导入本章所需要的模块
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import string
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torchtext import data
from torchtext.vocab import Vectors
import re
import time
import copy


def load_text_data(path):
    text_data = []
    label = []
    for dset in ['pos', 'neg']:
        path_dset = os.path.join(path, dset)  # dset : dataset
        path_list = os.listdir(path_dset)
        for fname in path_list:  # fname : filename
            if fname.endswith('.txt'):
                filename = os.path.join(path_dset, fname)
                with open(filename) as f:
                    text_data.append(f.read())
            if dset == 'pos ':
                label.append(1)
            else:
                label.append(0)
    return np.array(text_data), np.array(label)


df1 = pd.read_csv('train_imdb.csv')
df2 = pd.read_csv('test_imdb.csv')
# 提取"text"字段的内容并转换为Numpy数组
text_data = df1['text'].values
train_text = np.array(text_data)

train_label = df1['label'].values

text_data = df2['text'].values
test_text = np.array(text_data)

test_label = df2['label'].values

print(len(train_text), len(train_label))
print(len(test_text), len(test_label))
# 25000 25000
# 25000 25000


def text_preprocess(text_data):
    text_pre = []
    for text1 in text_data:
        text1 = re.sub('<br /><br />', ' ', text1)
        text1 = text1.lower()
        text1 = re.sub('\d+', '', text1)
        text1 = text1.translate(str.maketrans("", "", string.punctuation.replace("'", "")))
        text1 = text1.strip()
        text_pre.append(text1)
    return np.array(text_pre)


train_text_pre = text_preprocess(train_text)
test_text_pre = text_preprocess(test_text)

from nltk.tokenize import word_tokenize
import numpy as np
import re


# nltk.download('punkt')  # 下载punkt tokenizer

def stop_stem_word(datalist, stop_words):
    datalist_pre = []
    for text in datalist:
        text_words = word_tokenize(text)
        text_words = [word for word in text_words if word not in stop_words]
        text_words = [word for word in text_words if len(re.findall("'", word)) == 0]
        datalist_pre.append(text_words)
    return np.array(datalist_pre)


from nltk.corpus import stopwords

# nltk.download('stopwords')

stop_words = stopwords.words('english')
stop_words = set(stop_words)
train_text_pre2 = stop_stem_word(train_text_pre, stop_words)
test_text_pre2 = stop_stem_word(test_text_pre, stop_words)
print(train_text_pre[1])
# 篇章级语料
# i am curious yellow is a risible and pretentious steaming pile it doesn't matter what
print("=" * 10)
print(train_text_pre2[1])
# 去除停用词，按照空格切分成一个个的单词
# ['curious', 'yellow', 'risible', 'pretentious', 'steaming', 'pile',  'matter', 'one', 'political'


# 装上NLTK的stopwords
# Traceback (most recent call last):
#   File "D:\pythoncode\learn\a\deep_learning6.5.py", line 94, in <module>
#     stop_words = stopwords.words('english')
# NameError: name 'stopwords' is not defined


texts = [" ".join(words) for words in train_text_pre2]
traindatasave = pd.DataFrame({"text": texts, "label": train_label})
texts = [" ".join(words) for words in test_text_pre2]
testdatasave = pd.DataFrame({"text": texts, "label": test_label})
traindatasave.to_csv("imdb_train.csv", index=False)
testdatasave.to_csv("imdb_test.csv", index=False)


traindata = pd.DataFrame({"train_text": train_text, "train_word": train_text_pre2, "train_label": train_label})
train_word_num = [len(text) for text in train_text_pre2]
traindata["train_word_num"] = train_word_num
plt.figure(figsize=(8, 5))
_ = plt.hist(train_word_num, bins=100)
plt.xlabel("word number")
plt.ylabel("Freq")
plt.show()

from wordcloud import WordCloud

plt.figure(figsize=(16, 10))
for ii in np.unique(train_label):
    text = np.array(traindata.train_word[traindata.train_label == ii])
    text = ' '.join(np.concatenate(text))
    plt.subplot(1, 2, ii + 1)
    wordcod = WordCloud(font_path='simhei.ttf', margin=5, width=1800, height=1000, max_words=500, min_font_size=5,
                        background_color='white',
                        max_font_size=250)
    wordcod.generate_from_text(text)
    plt.imshow(wordcod)
    plt.axis("off")
    if ii == 1:
        plt.title("Positive")
    else:
        plt.title("Negative")
    plt.subplots_adjust(wspace=0.05)

plt.show()

# 课本代码导入模块没有给出代码导致，导入包即可
# Traceback (most recent call last):
#   File "D:\pythoncode\learn\a\deep_learning6.5.py", line 140, in <module>
#     wordcod = WordCloud(margin=5, width=1800, height=1000, max_words=500, min_font_size=5, max_font_size=250)
# NameError: name 'WordCloud' is not defined


# 字体报错 TrueType字体 TrueTypefonts .ttf文件  经过测试，字体路径是找到的，不然会报cannot open resource，
# wordcloud装了最新版1.9.2，降版本pip install wordcloud == 1.8.0 成功解决
# Traceback (most recent call last):
#   File "D:\pythoncode\learn\a\deep_learning6.5.py", line 142, in <module>
#     wordcod.generate_from_text(text)
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\wordcloud\wordcloud.py", line 621, in generate_from_text
#     self.generate_from_frequencies(words)
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\wordcloud\wordcloud.py", line 508, in generate_from_frequencies
#     box_size = draw.textbbox((0, 0), word, font=transposed_font, anchor="lt")
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\PIL\ImageDraw.py", line 651, in textbbox
#     raise ValueError("Only supported for TrueType fonts")
# ValueError: Only supported for TrueType fonts
#
# 进程已结束,退出代码1


mytokenize = lambda x: x.split()
# tokenize='basic_english'
TEXT = data.Field(sequential=True, tokenize=mytokenize,
                  include_lengths=True, use_vocab=True,
                  batch_first=True, fix_length=200)

LABEL = data.Field(sequential=False, use_vocab=False,
                   pad_token=None, unk_token=None)

# 对所要读取的数据集的列进行处理
train_test_fields = [  # 踩坑，注意这里是按照顺序索引的，比如，它不会按照列名找到”label“，而是按照第一个就是TEXT
    ("text", TEXT),
    ("label", LABEL)
]

traindata, testdata = data.TabularDataset.splits(
    path='./',
    format='csv',
    train='imdb_train.csv', fields=train_test_fields,
    test='imdb_test.csv', skip_header=True
)

# 打印数据集大小
print(len(traindata), len(testdata))
# 25000 25000


# ex0 = traindata.examples[0]
# print('ex0.label', ex0.label)
# print('ex0.text', ex0.text)


train_data, val_data = traindata.split(split_ratio=0.7)
print(len(train_data), len(val_data))
# 17500 7500


vec = Vectors("glove.6B.100d.txt")
#  使用训练集构建单词表，导入预先训练的词嵌入
TEXT.build_vocab(train_data, max_size=20000, vectors=vec)
LABEL.build_vocab(train_data)

# 训练集、验证集和测试集定义为迭代器
print(TEXT.vocab.freqs.most_common(n=10))
# [('movie', 30061), ('film', 27361), ('one', 18244), ('like', 13721), ('good', 10390), ('would', 9348), ('even', 8852), ('time', 8475), ('really', 8274), ('story', 8239)]

print("词典的词数:", len(TEXT.vocab.itos))
# 词典的词数: 20002

print("前10个单词：", TEXT.vocab.itos[0:10])
# 前10个单词： ['<unk>', '<pad>', 'movie', 'film', 'one', 'like', 'good', 'would', 'even', 'time']

print("类别标签情况：", LABEL.vocab.freqs)
# 类别标签情况： Counter({'0': 8795, '1': 8705})

BATCH_SIZE = 32
train_iter = data.BucketIterator(train_data, batch_size=BATCH_SIZE)
val_iter = data.BucketIterator(val_data, batch_size=BATCH_SIZE)
test_iter = data.BucketIterator(testdata, batch_size=BATCH_SIZE)


for step, batch in enumerate(train_iter):
    if step > 0:
        break
print("数据的类别标签", batch.label)
print("数据的尺寸", batch.text[0].shape)
print("数据样本数", batch.text[1])

# 数据的类别标签 tensor([1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
#         1, 0, 1, 1, 1, 0, 0, 0])
# 数据的尺寸 torch.Size([32, 200])
# 数据样本数 tensor([ 55,  30,  96,  68, 129, 100,  33, 200, 156,  49,  39, 200, 200,  21,
#         165, 152,  62, 101,  33, 110,  68, 105,  31, 111, 107,  58, 109,  76,
#         101, 200,  68,  64])

class CNN_Text(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters,
                      kernel_size=(fs, embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 100
N_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN_Text(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)
print(model)

pretrained_embeddings = TEXT.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()
#
def train_epoch(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    train_corrects = 0
    train_num = 0
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        pre = model(batch.text[0]).squeeze(1)
        loss = criterion(pre, batch.label.type(torch.FloatTensor))
        pre_lab = torch.round(torch.sigmoid(pre))
        train_corrects += torch.sum(torch.tensor(pre_lab.long() == batch.label))  # 括号内转为torch.tensor
        train_num += len(batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss = epoch_loss / train_num
    epoch_acc = train_corrects.double().item() / train_num
    return epoch_loss, epoch_acc


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    train_corrects = 0
    train_num = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            pre = model(batch.text[0]).squeeze(1)
            loss = criterion(pre, batch.label.type(torch.FloatTensor))
            pre_lab = torch.round(torch.sigmoid(pre))
            train_corrects += torch.sum(torch.tensor(pre_lab.long() == batch.label))
            train_num += len(batch.label)
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / train_num
        epoch_acc = train_corrects.double().item() / train_num
    return epoch_loss, epoch_acc


EPOCHS = 10
best_val_loss = float('inf')
best_acc = float(0)

for epoch in range(EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_epoch(model, train_iter, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_iter, criterion)
    end_time = time.time()
    print("Epoch:", epoch + 1, "|", "Epoch time", end_time - start_time, "s")
    print("Train Loss:", train_loss, "|", "Train Acc:", train_acc)
    print("Val. Loss:", val_loss, "|", "Val. Acc:", val_acc)

    if (val_loss < best_val_loss) & (val_acc > best_acc):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_val_loss = val_loss
        best_acc = val_acc

        model.load_state_dict(best_model_wts)

test_loss, test_acc = evaluate(model, test_iter, criterion)
print("在测试集上的预测精度", test_acc)


# Epoch: 1 | Epoch time 62.18106269836426 s
# Train Loss: 0.014305785978691919 | Train Acc: 0.7802857142857142
# Val. Loss: 0.010499856820702553 | Val. Acc: 0.8602666666666666
# Epoch: 2 | Epoch time 64.33356475830078 s
# Train Loss: 0.008702067593591553 | Train Acc: 0.8852
# Val. Loss: 0.0092372254550457 | Val. Acc: 0.878
# Epoch: 3 | Epoch time 63.778724908828735 s
# Train Loss: 0.005293139319973332 | Train Acc: 0.9350857142857143
# Val. Loss: 0.009738493733604749 | Val. Acc: 0.8790666666666667
# Epoch: 4 | Epoch time 64.7396559715271 s
# Train Loss: 0.0028063043298731955 | Train Acc: 0.9686857142857143
# Val. Loss: 0.011253245748331149 | Val. Acc: 0.8785333333333334
# Epoch: 5 | Epoch time 66.96816277503967 s
# Train Loss: 0.0014695384308296654 | Train Acc: 0.9862857142857143
# Val. Loss: 0.013079128769785165 | Val. Acc: 0.8745333333333334
# Epoch: 6 | Epoch time 66.8894910812378 s
# Train Loss: 0.0007409520095946001 | Train Acc: 0.9937142857142857
# Val. Loss: 0.015568103588372469 | Val. Acc: 0.8732
# Epoch: 7 | Epoch time 64.34104418754578 s
# Train Loss: 0.0005200898583279923 | Train Acc: 0.9951428571428571
# Val. Loss: 0.017476340961021682 | Val. Acc: 0.8736
# Epoch: 8 | Epoch time 67.11497616767883 s
# Train Loss: 0.0003756954844608637 | Train Acc: 0.9958857142857143
# Val. Loss: 0.02083487620341281 | Val. Acc: 0.8597333333333333
# Epoch: 9 | Epoch time 65.55319833755493 s
# Train Loss: 0.00022720735681110195 | Train Acc: 0.9985142857142857
# Val. Loss: 0.021132646035403012 | Val. Acc: 0.8742666666666666
# Epoch: 10 | Epoch time 78.16548037528992 s
# Train Loss: 0.00020813046585618786 | Train Acc: 0.9982857142857143
# Val. Loss: 0.023343770688772202 | Val. Acc: 0.8709333333333333
# 在测试集上的预测精度 0.85108
#
# 进程已结束,退出代码0
