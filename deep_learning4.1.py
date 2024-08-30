# 基于pytorch的相关可视化工具 网络结构的可视化

import torch
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
from torch.optim import SGD
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
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
print(MyConvnet)
# ConvNet(
#   (conv2): Sequential(
#     (0): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#     (1): ReLU()
#     (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
#   )
#   (fc): Sequential(
#     (0): Linear(in_features=1568, out_features=128, bias=True)
#     (1): ReLU()
#     (2): Linear(in_features=128, out_features=64, bias=True)
#     (3): ReLU()
#   )
#   (out): Linear(in_features=64, out_features=10, bias=True)
# )

import hiddenlayer as hl

# hl_graph = hl.build_graph(MyConvnet, torch.zeros([1, 1, 32, 32]))
hl_graph = hl.build_graph(MyConvnet, torch.zeros([1, 1, 30, 30]))

hl_graph.theme = hl.graph.THEMES['blue'].copy()
# hl_graph.save("deep Learning/MyConvnet_hl.png", format='png')


from torchviz import make_dot

x = torch.randn(1, 1, 30, 30).requires_grad_(True)
y = MyConvnet(x)
MyConvnetvis = make_dot(y, params=dict(list(MyConvnet.named_parameters()) + [('x', x)]))
MyConvnetvis.format = 'png'
MyConvnetvis.render("deep Learning/MyConvnet_vis")
MyConvnetvis.view()


# 输入的张量维度有问题，修改
# 修改前 课本的 hl_graph = hl.build_graph(MyConvnet, torch.zeros([1, 1, 28, 28]))
# 修改后编译通过  hl_graph = hl.build_graph(MyConvnet, torch.zeros([1, 1, 32, 32]))

# Traceback (most recent call last):
#   File "D:\pythoncode\learn\a\deep_learning4.1.py", line 85, in <module>
#     hl_graph = hl.build_graph(MyConvnet, torch.zeros([1,1,28,28]))
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\hiddenlayer\graph.py", line 143, in build_graph
#     import_graph(g, model, args)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\hiddenlayer\pytorch_builder.py", line 70, in import_graph
#     trace, out = torch.jit._get_trace_graph(model, args)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\jit\_trace.py", line 1310, in _get_trace_graph
#     outs = ONNXTracedModule(
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\nn\modules\module.py", line 1532, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\nn\modules\module.py", line 1541, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\jit\_trace.py", line 138, in forward
#     graph, out = torch._C._create_graph_by_tracing(
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\jit\_trace.py", line 129, in wrapper
#     outs.append(self.inner(*trace_inputs))
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\nn\modules\module.py", line 1532, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\nn\modules\module.py", line 1541, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\nn\modules\module.py", line 1522, in _slow_forward
#     result = self.forward(*input, **kwargs)
#   File "D:\pythoncode\learn\a\deep_learning4.1.py", line 62, in forward
#     x = self.fc(x)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\nn\modules\module.py", line 1532, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\nn\modules\module.py", line 1541, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\nn\modules\module.py", line 1522, in _slow_forward
#     result = self.forward(*input, **kwargs)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\nn\modules\container.py", line 217, in forward
#     input = module(input)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\nn\modules\module.py", line 1532, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\nn\modules\module.py", line 1541, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\nn\modules\module.py", line 1522, in _slow_forward
#     result = self.forward(*input, **kwargs)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\torch\nn\modules\linear.py", line 116, in forward
#     return F.linear(input, self.weight, self.bias)
# RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x1152 and 1568x128)
#
# 进程已结束,退出代码1


# 版本变化问题 ，修改这里 _optimize_trace变成 _optimize_graph
#
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\hiddenlayer\pytorch_builder.py", line 71, in import_graph
#     torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)

# Traceback (most recent call last):
#   File "D:\pythoncode\learn\a\deep_learning4.1.py", line 88, in <module>
#     hl_graph = hl.build_graph(MyConvnet, torch.zeros([1, 1, 30, 30]))
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\hiddenlayer\graph.py", line 143, in build_graph
#     import_graph(g, model, args)
#   File "D:\anaconda3\envs\portfolio\lib\site-packages\hiddenlayer\pytorch_builder.py", line 71, in import_graph
#     torch_graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
# AttributeError: module 'torch.onnx' has no attribute '_optimize_trace'. Did you mean: '_optimize_graph'?



#  torch版本不适配问题，安装torch1.1 最终解决
# Traceback (most recent call last):
#   File "D:\pythoncode\learn\a\deep_learning4.1.py", line 87, in <module>
#     hl_graph = hl.build_graph(MyConvnet, torch.zeros([1, 1, 30, 30]))
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\hiddenlayer\graph.py", line 136, in build_graph
#     import_graph(g, model, args)
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\hiddenlayer\pytorch_builder.py", line 68, in import_graph
#     hl_node = Node(uid=pytorch_id(torch_node), name=None, op=op, params=params)
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\hiddenlayer\pytorch_builder.py", line 43, in pytorch_id
#     return node.scopeName() + "/outputs/" + "/".join([o.uniqueName() for o in node.outputs()])
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\hiddenlayer\pytorch_builder.py", line 43, in <listcomp>
#     return node.scopeName() + "/outputs/" + "/".join([o.uniqueName() for o in node.outputs()])
# AttributeError: 'torch._C.Value' object has no attribute 'uniqueName'
#
# 进程已结束,退出代码1


# graphviz没添加到环境变量，去下载exe安装程序安装然后添加到环境变量
# Traceback (most recent call last):
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\graphviz\backend\execute.py", line 85, in run_check
#     proc = subprocess.run(cmd, **kwargs)
#   File "D:\anaconda3\envs\deeplearning\lib\subprocess.py", line 423, in run
#     with Popen(*popenargs, **kwargs) as process:
#   File "D:\anaconda3\envs\deeplearning\lib\subprocess.py", line 729, in __init__
#     restore_signals, start_new_session)
#   File "D:\anaconda3\envs\deeplearning\lib\subprocess.py", line 1017, in _execute_child
#     startupinfo)
# FileNotFoundError: [WinError 2] 系统找不到指定的文件。
#
# The above exception was the direct cause of the following exception:
#
# Traceback (most recent call last):
#   File "D:\pythoncode\learn\a\deep_learning4.1.py", line 90, in <module>
#     hl_graph.save("deep Learning/MyConvnet_hl.png", format='png')
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\hiddenlayer\graph.py", line 356, in save
#     dot.render(file_name, directory=directory, cleanup=True)
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\graphviz\_tools.py", line 172, in wrapper
#     return func(*args, **kwargs)
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\graphviz\rendering.py", line 119, in render
#     rendered = self._render(*args, **kwargs)
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\graphviz\_tools.py", line 172, in wrapper
#     return func(*args, **kwargs)
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\graphviz\backend\rendering.py", line 320, in render
#     capture_output=True)
#   File "D:\anaconda3\envs\deeplearning\lib\site-packages\graphviz\backend\execute.py", line 88, in run_check
#     raise ExecutableNotFound(cmd) from e
# graphviz.backend.execute.ExecutableNotFound: failed to execute 'dot', make sure the Graphviz executables are on your systems' PATH
#
# 进程已结束,退出代码1
