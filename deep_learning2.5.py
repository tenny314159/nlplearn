# pytorch 中数据操作和预处理

# 书中的版本有两个类型，一种是CPU,PYTORCH 1.3 , PYTHON 3.6 ,MacOS 系统，
# 另一个是 GPU ,PYTORCH 1.0 CUDA 8.0 , PYTHON 3.6 CentOS Linux 7 (Core) 系统


import torch
import numpy as np
import torch.utils.data as Data
import pandas as pd
from sklearn.datasets import load_iris
# from sklearn.datasets import load_boston


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

print("X:", type(X), X.dtype)
print("Y:", type(Y), Y.dtype)
# X: <class 'numpy.ndarray'> float64
# Y: <class 'numpy.ndarray'> float64

# 先变32 再变成 tensor
train_xt = torch.from_numpy(X.astype(np.float32))
train_yt = torch.from_numpy(Y.astype(np.float32))

print("train_xt:", type(train_xt), train_xt.dtype)
print("train_yt:", type(train_yt), train_yt.dtype)
# train_xt: <class 'torch.Tensor'> torch.float32
# train_yt: <class 'torch.Tensor'> torch.float32

train_data = Data.TensorDataset(train_xt, train_yt)
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0  # 多线程容易报错
                               )

for step, (d_x, d_y) in enumerate(train_loader):
    if step > 0:
        break

print('d_x.shape:', d_x.shape)
print('d_y.shape:', d_y.shape)
print('d_x.dtype', d_x.dtype)
print('d_y.dtype', d_y.dtype)

# d_x.shape: torch.Size([64, 10])
# d_y.shape: torch.Size([64])
# d_x.dtype torch.float32
# d_y.dtype torch.float32


''' # `load_boston` has been removed from scikit-learn since version 1.2
报错 ，解决思路：上网下载后，本地加载 ；
换一个版本的sklearn ; 
换一个数据集 。这里是换了一个数据集

D:\anaconda3\envs\portfolio\python.exe D:\pythoncode\learn\a\deep_learning2.5.py 
Traceback (most recent call last):
  File "D:\pythoncode\learn\a\deep_learning2.5.py", line 7, in <module>
    from sklearn.datasets import load_iris , load_boston
  File "D:\anaconda3\envs\portfolio\lib\site-packages\sklearn\datasets\__init__.py", line 157, in __getattr__
    raise ImportError(msg)
ImportError: 
`load_boston` has been removed from scikit-learn since version 1.2.

The Boston housing prices dataset has an ethical problem: as
investigated in [1], the authors of this dataset engineered a
non-invertible variable "B" assuming that racial self-segregation had a
positive impact on house prices [2]. Furthermore the goal of the
research that led to the creation of this dataset was to study the
impact of air quality but it did not give adequate demonstration of the
validity of this assumption.

The scikit-learn maintainers therefore strongly discourage the use of
this dataset unless the purpose of the code is to study and educate
about ethical issues in data science and machine learning.

In this special case, you can fetch the dataset from the original
source::

    import pandas as pd
    import numpy as np

    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

Alternative datasets include the California housing dataset and the
Ames housing dataset. You can load the datasets as follows::

    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()

for the California housing dataset and::

    from sklearn.datasets import fetch_openml
    housing = fetch_openml(name="house_prices", as_frame=True)

for the Ames housing dataset.

[1] M Carlisle.
"Racist data destruction?"
<https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8>

[2] Harrison Jr, David, and Daniel L. Rubinfeld.
"Hedonic housing prices and the demand for clean air."
Journal of environmental economics and management 5.1 (1978): 81-102.
<https://www.researchgate.net/publication/4974606_Hedonic_housing_prices_and_the_demand_for_clean_air>
'''

iris_x, iris_y = load_iris(return_X_y=True)

print('iris_x.dtype:', iris_x.dtype)
print('iris_y.dtype:', iris_y.dtype)
# iris_x.dtype: float64
# iris_y.dtype: int32

print('iris_x.shape:', iris_x.shape)
print('iris_y.shape:', iris_y.shape)
# iris_x.shape: (150, 4)
# iris_y.shape: (150,)


train_xt = torch.from_numpy(iris_x.astype(np.float32))
train_yt = torch.from_numpy(iris_y.astype(np.int64))

print('train_xt.dtype:', train_xt.dtype)
print('train_yt.dtype:', train_yt.dtype)
# train_xt.dtype: torch.float32
# train_yt.dtype: torch.int64


train_data = Data.TensorDataset(train_xt, train_yt)
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=10, shuffle=True,
                               # num_workers = 1, # 使用两个线程可能会报错
                               num_workers=0,
                               )

for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break

print('b_x.shape', b_x.shape)
print('b_y.shape', b_y.shape)
# b_x.shape torch.Size([10, 4])
# b_y.shape torch.Size([10])
print('b_x.dtype:', b_x.dtype)
print('b_y.dtype:', b_y.dtype)
# b_x.dtype: torch.float32
# b_y.dtype: torch.int64


# File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\utils\data\dataloader.py", line 1133, in _try_get_data
#     data = self._data_queue.get(timeout=timeout)
#   File "D:\anaconda3\envs\portfolio\lib\multiprocessing\queues.py", line 114, in get
#     raise Empty
# _queue.Empty


# import torch
# from torchvision.datasets import FashionMNIST
# import torchvision.transforms as transforms
# from torchvision.datasets import ImageFolder
#
# train_data = FashionMNIST(root='./deep Learning/FashionMNIST',
#                           train=True,
#                           transform=transforms.ToTensor(),
#                           download=True)
#
# train_loader = Data.DataLoader(dataset=train_data,
#                                batch_size=64,
#                                shuffle=True,
#                                num_workers=0,
#                                )
#
# print(len(train_loader))
# # 938
#
# test_data = FashionMNIST(root='./deep Learning/FashionMNIST',
#                          train=False,
#                          download=False)
#
# test_data_x = test_data.data.type(torch.FloatTensor) / 255.0
# test_data_x = torch.unsqueeze(test_data_x, dim=1)
# test_data_y = test_data.targets
# print(test_data_x.shape)
# print(test_data_y.shape)
# # torch.Size([10000, 1, 28, 28])
# # torch.Size([10000])
#
#
# train_data_transforms = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
# # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# # 是一个数据标准化（Normalization）操作，
# # 用于对图像数据进行标准化处理。其中的六个数字分别代表均值和标准差。
#
#
# train_data_dir = './deep Learning/'
# train_data = ImageFolder(train_data_dir,
#                          transform=train_data_transforms,
#                          )
#
# train_data_loader = Data.DataLoader(
#     train_data,
#     batch_size=4,
#     shuffle=True
# )
#
# print(train_data.targets)
# # [0]
#
# for step, (b_x, b_y) in enumerate(train_data_loader):
#     if step > 0:
#         break
#
# print(b_x.shape)
# print(b_y.shape)
# print(b_x.min(), b_x.max())
# # torch.Size([1, 3, 224, 224])
# # torch.Size([1])
# # tensor(-1.4329) tensor(2.6400)


from torchtext import data
import torchtext

mytokenize = lambda x: x.split()

TEXT = torchtext.data.Field(sequential=True,
                            tokenize=mytokenize,
                            use_vocab=True,
                            batch_first=True,
                            fix_length=200)

LABEL = data.Field(sequential=False, use_vocab=False,
                   pad_token=None,
                   unk_token=None
                   )

text_data_fields = [('label', LABEL), ('text', TEXT)]

traindata, testdata = data.TabularDataset.splits(
    path='./deep Learning/',
    format='csv',
    train='train.csv',
    fields=text_data_fields,
    test='test.csv',
    skip_header=True)
print(traindata, testdata)
# <torchtext.data.dataset.TabularDataset object at 0x000001739709DC60> <torchtext.data.dataset.TabularDataset object at 0x000001739709D6C0>
print(len(traindata), len(testdata))
# 1 1

TEXT.build_vocab(traindata, max_size=1000, vectors=None)
train_iter = data.BucketIterator(traindata, batch_size=4)
test_iter = data.BucketIterator(testdata, batch_size=4)
for step, batch in enumerate(train_iter):
    if step > 0:
        break

    print(batch.label)
    print(batch.text.shape)
    # tensor([1])
    # torch.Size([1, 200])


# D:\anaconda3\envs\portfolio\lib\site-packages\torchtext\data\__init__.py:4: UserWarning:
# /!\ IMPORTANT WARNING ABOUT TORCHTEXT STATUS /!\
# Torchtext is deprecated and the last released version will be 0.18 (this one). You can silence this warning by calling the following at the beginnign of your scripts: `import torchtext; torchtext.disable_torchtext_deprecation_warning()`
#   warnings.warn(torchtext._TORCHTEXT_DEPRECATION_MSG)
# Traceback (most recent call last):
#   File "D:\pythoncode\learn\a\deep_learning2.5.py", line 239, in <module>
#     TEXT = data.Field(sequential=True,
# AttributeError: module 'torchtext.data' has no attribute 'Field'


# 官方的releases里面看到了，
# 官方在0.9.0版本中将Field 等函数放进了legacy中，
# 在最新版的0.12.0中移除了这个文件夹。
# pip 0.9.0没有找到，最终pip install torchtext==0.6.0 成功解决

