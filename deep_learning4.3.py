# 4.3 使用Visdom进行可视化

import torch
from visdom import Visdom
from sklearn.datasets import load_iris
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
import torch.utils.data as Data

iris_x, iris_y = load_iris(return_X_y=True)

print(iris_x.shape)
print(iris_y.shape)
# (150, 4)
# (150,)


vis = Visdom()
vis.scatter(iris_x[:, 0:2], Y=iris_y + 1, win='windows1', env='main')
vis.scatter(iris_x[:, 0:3], Y=iris_y + 1, win='3D 散点图', env='main',
            opts=dict(marksize=4, xlabel='特征1', ylabel='特征2'))

vis = Visdom()
x = torch.linspace(-6, 6, 100).view((-1, 1))
sigmoid = torch.nn.Sigmoid()
sigmoidy = sigmoid(x)
tanh = torch.nn.Tanh()
tanhy = tanh(x)
relu = torch.nn.ReLU()
reluy = relu(x)
ploty = torch.cat((sigmoidy, tanhy, reluy), dim=1)
plotx = torch.cat((x, x, x), dim=1)
vis.line(Y=ploty, X=plotx, win='line plot', env='main',
         opts=dict(dash=np.array(['solid', 'dash', 'dashdot']), legend=['sigmoid', 'tanh', 'relu']))

x = torch.linspace(-6, 6, 100).view((-1, 1))
y1 = torch.sin(x)
y2 = torch.cos(x)
plotx = torch.cat((x, x), dim=1)
ploty = torch.cat((y1, y2), dim=1)
vis.stem(X=plotx, Y=ploty, win='stem plot', env='main',
         opts=dict(legend=['sin(x)', 'cos(x)'],
                   title='茎叶图'))

iris_corr = torch.from_numpy(np.corrcoef(iris_x, rowvar=False))
vis.heatmap(iris_corr, win='heatmap', env='main',
            opts=dict(rownames=['x1', 'x2', 'x3', 'x4'],
                      columnnames=['x1', 'x2', 'x3', 'x4'],
                      title='热力图'
                      )
            )

train_data = FashionMNIST(root="./deep Learning/",
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=128,
                               shuffle=True,
                               )

for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break

    print(b_x.shape)
    print(b_y.shape)
    # torch.Size([128, 1, 28, 28])
    # torch.Size([128])

vis.image(b_x[0, :, :, :], win='one image', env='MyimagePlot',
          opts=dict(title='一张图像')
          )
vis.images(b_x, win='my batch image ', env='MyimagePlot',
           nrow=16,
           opts=dict(title='一个batch的图像')
           )

texts = 'A Flexible tool for creating, organizing, and sharing visualizations of live, rich data. Supports Torch and Numpy.'
vis.text(texts, win='text plot', env='My image plot', opts=dict(title='可视化文本'))


# urllib3.exceptions.MaxRetryError: HTTPConnectionPool(host='localhost', port=8097): Max retries exceeded with url: /events (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x0000027D7F91E080>: Failed to establish a new connection: [WinError 10061] 由于目标计算机积极拒绝，无法连接。',))
#
# During handling of the above exception, another exception occurred:
#
# Traceback (most recent call last):
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\visdom\__init__.py", line 760, in _send
#     data=json.dumps(msg),
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\visdom\__init__.py", line 720, in _handle_post
#     r = self.session.post(url, data=data)
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\requests\sessions.py", line 577, in post
#     return self.request('POST', url, data=data, json=json, **kwargs)
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\requests\sessions.py", line 529, in request
#     resp = self.send(prep, **send_kwargs)
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\requests\sessions.py", line 645, in send
#     r = adapter.send(request, **kwargs)
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\requests\adapters.py", line 519, in send
#     raise ConnectionError(e, request=request)
# requests.exceptions.ConnectionError: HTTPConnectionPool(host='localhost', port=8097): Max retries exceeded with url: /events (Caused by NewConnectionError('<urllib3.connection.HTTPConnection object at 0x0000027D7F91E080>: Failed to establish a new connection: [WinError 10061] 由于目标计算机积极拒绝，无法连接。',))
#
# 进程已结束,退出代码0


# 解决方法，先在命令行运行，然后运行py文件
# (deeplearning) C:\Users\黄天佑>python -m visdom.server -p 8091
# Checking for scripts.
# Downloading scripts, this may take a little while
# ERROR:root:Error [WinError 10060] 由于连接方在一段时间后没有正确答复或连接的主机没有反应，连接尝试失败。 while downloading https://cdn.plot.ly/plotly-2.11.1.min.js
# It's Alive!
# INFO:root:Application Started
# INFO:root:Working directory: C:\Users\黄天佑\.visdom
# You can navigate to http://localhost:8091

