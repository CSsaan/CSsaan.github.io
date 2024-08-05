---
layout:     post
title:      "常见损失函数"
subtitle:   "多种常用损失函数（loss function）"
date:       2024-08-02 18:36:00
author:     "CS"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - lossFunction
---

常见的损失函数
--------------


#### **1\. L1范数损失 L1Loss**

计算 output 和 target 之差的绝对值。

公式：
![L1Loss](https://i-blog.csdnimg.cn/blog_migrate/2e545d499616ad8315e205e178f282cc.png) 

$$
\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left| x_n - y_n \right|
$$

$$
\ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}
$$

``` python
import torch
import torch.nn as nn
pred = torch.tensor([1.0, 2.0, 3.0, 4.0])   # 相应的真实值
target = torch.tensor([1.5, 2.5, 3.5, 4.5]) # 模型的预测值
criterion = nn.L1Loss(reduction='mean')     # none: 不使用约简；mean:返回loss和的平均值；sum:返回loss的和。默认：mean。
loss = criterion(pred, target)              # 计算 L1 损失
print(loss)                                 # 输出结果：tensor(0.5000)
```

**2 均方误差损失 MSELoss**

计算 output 和 target 之差的均方差。

![MSELoss](https://i-blog.csdnimg.cn/blog_migrate/442895b21406698a06831b733b224b37.png)

$$
\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
l_n = \left( x_n - y_n \right)^2
$$

``` python
import torch
import torch.nn as nn
pred = torch.tensor([1.0, 2.0, 3.0, 4.0])   # 真实值
target = torch.tensor([1.5, 2.5, 3.5, 4.5]) # 预测值
criterion = nn.MSELoss(reduction='mean')    # none: 不使用约简；mean:返回loss和的平均值；sum:返回loss的和。默认：mean。
loss = criterion(pred, target)              # 计算 L1 损失
print(loss)                                 # 输出结果：tensor(0.5000)
```

**3 交叉熵损失 CrossEntropyLoss（多分类）**

当训练有 C 个类别的分类问题时很有效. 可选参数 weight 必须是一个1维 Tensor, 权重将被分配给各个类别. 对于不平衡的训练集非常有效。

在多分类任务中，经常采用 softmax 激活函数+交叉熵损失函数，因为交叉熵描述了两个概率分布的差异，然而神经网络输出的是向量，并不是概率分布的形式。所以需要 softmax激活函数将一个向量进行“归一化”成概率分布的形式，再采用交叉熵损失函数计算 loss。

$$
\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore\_index}\}
$$

``` python
torch.nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean')
```

参数：

> weight (Tensor, optional) – 自定义的每个类别的权重. 必须是一个长度为 C 的 Tensor
> 
> ignore\_index (int, optional) – 设置一个目标值, 该目标值会被忽略, 从而不会影响到 输入的梯度。
> 
> reduction-三个值，none: 不使用约简；mean:返回loss和的平均值；sum:返回loss的和。默认：mean。

**4 KL 散度损失 KLDivLoss**

KL散度，又叫相对熵，用于衡量两个分布（离散分布和连续分布）之间的距离。
计算 input 和 target 之间的 KL 散度。KL 散度可用于衡量不同的连续分布之间的距离, 在连续的输出分布的空间上(离散采样)上进行直接回归时 很有效.

$$
L(y_{\text{pred}},\ y_{\text{true}})
            = y_{\text{true}} \cdot \log \frac{y_{\text{true}}}{y_{\text{pred}}}
            = y_{\text{true}} \cdot (\log y_{\text{true}} - \log y_{\text{pred}})
$$

``` python
import torch.nn.functional as F
kl_loss = nn.KLDivLoss(reduction="batchmean")
# input should be a distribution in the log space
input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
# Sample a batch of distributions. Usually this would come from the dataset
target = F.softmax(torch.rand(3, 5), dim=1)
output = kl_loss(input, target)
kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
log_target = F.log_softmax(torch.rand(3, 5), dim=1)
output = kl_loss(input, log_target)
```

参数：

> reduction-三个值，none: 不使用约简；mean:返回loss和的平均值；sum:返回loss的和。默认：mean。

**5 二进制交叉熵损失 BCELoss（二分类）**

二分类任务时的交叉熵计算函数。用于测量重构的误差, 例如自动编码机. 注意目标的值 ti 的范围为0到1之间.
> 要求模型的输出经过 Sigmoid 处理后再计算损失。

$$
\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]
$$

``` python
import torch
import torch.nn as nn
loss = nn.BCELoss(weight=None, reduction='mean', pos_weight=None)
input = torch.randn(3, 2, requires_grad=True)
target = torch.rand(3, 2, requires_grad=False)
output = loss(nn.Sigmoid(input), target)
```

参数：

> weight (Tensor, optional) – 自定义的每个 batch 元素的 loss 的权重. 必须是一个长度为 “nbatch” 的 的 Tensor

**6 BCEWithLogitsLoss（二分类）**

BCEWithLogitsLoss损失函数相当于BCELoss的进化版, 把 手动调用Sigmoid 层集成到了 BCELoss 类中. 该版比用一个简单的 Sigmoid 层和 BCELoss 在数值上更稳定, 因为把这两个操作合并为一个层之后, 可以利用 log-sum-exp 的 技巧来实现数值稳定.
> 直接接受模型的 logits 作为输入，无需手动调用 Sigmoid再输入。

$$
\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log \sigma(x_n)
        + (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right]
$$

``` python
import torch
import torch.nn as nn
loss = nn.BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None)
input = torch.randn(3, 2, requires_grad=True)
target = torch.rand(3, 2, requires_grad=False)
output = loss(input, target)
```

参数：

> weight (Tensor, optional) – 自定义的每个 batch 元素的 loss 的权重. 必须是一个长度 为 “nbatch” 的 Tensor

#### **7 MarginRankingLoss**

是一种常用于学习排名任务的损失函数，通过比较样本对之间的相似性或相关性来进行训练，适用于需要处理排序、匹配等问题的情景。
给定一对输入 ( x1 ) 和 ( x2 )，以及它们的标签 ( y )（1 表示相似，-1 表示不相似），该损失函数计算如下:

$$
\text{loss}(x1, x2, y) = \max(0, -y * (x1 - x2) + \text{margin})
$$

``` python
import torch
import torch.nn as nn
loss = nn.MarginRankingLoss(margin=0.0, reduction='mean')
input1 = torch.randn(3, requires_grad=True)
input2 = torch.randn(3, requires_grad=True)
target = torch.randn(3).sign()
output = loss(input1, input2, target)
```

对于 mini-batch(小批量) 中每个实例的损失函数如下:  
![Image 3](https://mmbiz.qpic.cn/mmbiz_png/jupejmznDCiblNT5PlMy5OhibID0G9aHibga45X8UAS3sECSH53Lu1X9ia6ubnm7JNJZLmicRLZLsAdL8SCGC2Fm7Tw/640?wx_fmt=png)  
参数：

> margin:默认值0

#### **8 HingeEmbeddingLoss**

是用于支持向量机（SVM）训练中常用的损失函数，用来度量两个输入之间的相似性或差异性。 常用于非线性词向量学习以及半监督学习。
对于一对输入 ( x1 ) 和 ( x2 )，以及它们的标签 ( y )（1 表示相似，-1 表示不相似），损失函数的计算基于它们之间的距离或相似度差异，如下：

$$
l_n = \begin{cases}
            x_n, & \text{if}\; y_n = 1,\\
            \max \{0, margin - x_n\}, & \text{if}\; y_n = -1,
        \end{cases}
$$

``` python
torch.nn.HingeEmbeddingLoss(margin=1.0,  reduction='mean')
```

参数：  

> margin:默认值1

#### **9 多标签分类损失 MultiLabelMarginLoss**

``` python
torch.nn.MultiLabelMarginLoss(reduction='mean')
```

对于mini-batch(小批量) 中的每个样本按如下公式计算损失:  
![Image 5](https://mmbiz.qpic.cn/mmbiz_png/jupejmznDCiblNT5PlMy5OhibID0G9aHibgFlPvCWiciaibpfibNuib5NcCk9ezblfEsFvq2HfW3T73QeuvMSmQibWN4fbQ/640?wx_fmt=png)  

#### **10 平滑版L1损失 SmoothL1Loss**

也被称为 Huber 损失函数。

``` python
torch.nn.SmoothL1Loss(reduction='mean')
```

![Image 6](https://mmbiz.qpic.cn/mmbiz_png/jupejmznDCiblNT5PlMy5OhibID0G9aHibgB7ibKEQBABtP2qBqxGpxTB8bO9vgoWzz7t3pzHibj3J83siaBpULblm1g/640?wx_fmt=png)  
其中  
![Image 7](https://mmbiz.qpic.cn/mmbiz_png/jupejmznDCiblNT5PlMy5OhibID0G9aHibg4XvsXicfmQWm0xpUPWmRA7hWs42h6QyPTBpZamrcYLPmIwo8LzJ9jzA/640?wx_fmt=png)  

#### **11 2分类的logistic损失 SoftMarginLoss**

``` python
torch.nn.SoftMarginLoss(reduction='mean')
```

![Image 8](https://mmbiz.qpic.cn/mmbiz_png/jupejmznDCiblNT5PlMy5OhibID0G9aHibgS998d8I36EHcRd9mfFYoaOss0tUnicY8UbqrtiaUJLGddCZ5GIkGAD0w/640?wx_fmt=png)  

#### **12 多标签 one-versus-all 损失 MultiLabelSoftMarginLoss**

``` python
torch.nn.MultiLabelSoftMarginLoss(weight=None, reduction='mean')
```

![Image 9](https://mmbiz.qpic.cn/mmbiz_png/jupejmznDCiblNT5PlMy5OhibID0G9aHibgQEZuRu32ia4xsSueuW1XPeVo3LFduakOjDJY6vrMW3iaqDaD0512iaNug/640?wx_fmt=png)  

#### **13 cosine 损失 CosineEmbeddingLoss**

``` python
torch.nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')
```

![Image 10](https://mmbiz.qpic.cn/mmbiz_png/jupejmznDCiblNT5PlMy5OhibID0G9aHibgqRaSTctd3XacwT6bKUs1XKKQtFClZ9zAkoY2Kcn9siboGPnsudMA1gA/640?wx_fmt=png)  
参数：  

> margin:默认值0

#### **14 多类别分类的hinge损失 MultiMarginLoss**

``` python
torch.nn.MultiMarginLoss(p=1, margin=1.0, weight=None,  reduction='mean')
```

![Image 11](https://mmbiz.qpic.cn/mmbiz_png/jupejmznDCiblNT5PlMy5OhibID0G9aHibg6oLicPaib4ERhVt9riaTDZjb28rEwtJ1yfyaCpicpiapAIicLUPGjf0ygtaA/640?wx_fmt=png)  
参数：  

> p=1或者2 默认值：1  
> margin:默认值1

#### **15 三元组损失 TripletMarginLoss**

和孪生网络相似，具体例子：给一个A，然后再给B、C，看看B、C谁和A更像。  

![Image 12](https://mmbiz.qpic.cn/mmbiz_jpg/jupejmznDCiblNT5PlMy5OhibID0G9aHibgmkCsospxqF5razYjBkxDLGTycxTrdlZCk3BTuxo8Icy3KmwzGVmeNQ/640?wx_fmt=jpeg)

``` python
torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-06, swap=False, reduction='mean')
```

![Image 13](https://mmbiz.qpic.cn/mmbiz_png/jupejmznDCiblNT5PlMy5OhibID0G9aHibgqPq3wTAiaKKk0JcNxkQJem6WmaV7VeImyKXVN6ibXoFs3rxjibGcelVdw/640?wx_fmt=png)  
其中：  
![Image 14](https://mmbiz.qpic.cn/mmbiz_png/jupejmznDCiblNT5PlMy5OhibID0G9aHibgV3s6OZeRys7zLHRHv0X7Bq2ajficlNj3nOAVZoOEAdYJKuAqPlymfuQ/640?wx_fmt=png)  

**16 连接时序分类损失 CTCLoss**

CTC连接时序分类损失，可以对没有对齐的数据进行自动对齐，主要用在没有事先对齐的序列化数据训练上。比如语音识别、ocr识别等等。

``` python
torch.nn.CTCLoss(blank=0, reduction='mean')
```

参数：

> reduction-三个值，none: 不使用约简；mean:返回loss和的平均值；sum:返回loss的和。默认：mean。

**17 负对数似然损失 NLLLoss**

负对数似然损失. 用于训练 C 个类别的分类问题.

``` python
torch.nn.NLLLoss(weight=None, ignore_index=-100,  reduction='mean')
```

参数：  

> weight (Tensor, optional) – 自定义的每个类别的权重. 必须是一个长度为 C 的 Tensor
> 
> ignore\_index (int, optional) – 设置一个目标值, 该目标值会被忽略, 从而不会影响到 输入的梯度.

**18 NLLLoss2d**

对于图片输入的负对数似然损失. 它计算每个像素的负对数似然损失.

``` python
torch.nn.NLLLoss2d(weight=None, ignore_index=-100, reduction='mean')
```

参数：  

> weight (Tensor, optional) – 自定义的每个类别的权重. 必须是一个长度为 C 的 Tensor
> 
> reduction-三个值，none: 不使用约简；mean:返回loss和的平均值；sum:返回loss的和。默认：mean。

**19 PoissonNLLLoss**

目标值为泊松分布的负对数似然损失

``` python
torch.nn.PoissonNLLLoss(log_input=True, full=False,  eps=1e-08,  reduction='mean')
```

参数：

> log\_input (bool, optional) – 如果设置为 True , loss 将会按照公 式 exp(input) - target \* input 来计算, 如果设置为 False , loss 将会按照 input - target \* log(input+eps) 计算.
> 
> full (bool, optional) – 是否计算全部的 loss, i. e. 加上 Stirling 近似项 target \* log(target) - target + 0.5 \* log(2 \* pi \* target).
> 
> eps (float, optional) – 默认值: 1e-8
