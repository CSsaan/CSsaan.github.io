---
layout:     post
title:      "动态调整学习率learning rate(Cosine-learning-rate & warmup-step-decay)"
subtitle:   "pytorch实现lr_scheduler"
date:       2024-07-31 17:11:00
author:     "CS"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Python
    - LearningRate
    - Cosine
    - warmupStep
---

PyTorch六个学习率调整策略
------------------

PyTorch学习率调整策略通过`torch.optim.lr_scheduler`接口实现。PyTorch提供的学习率调整策略分为三大类，分别是:
*   `有序调整`：等间隔调整(Step)，按需调整学习率(MultiStep)，指数衰减调整(Exponential)和余弦退火CosineAnnealing。
*   `自适应调整`：自适应调整学习率 ReduceLROnPlateau。
*   `自定义调整`：自定义调整学习率 LambdaLR。

------------------

### 1、等间隔调整学习率 StepLR

等间隔调整学习率，调整倍数为 gamma 倍，调整间隔为 step\_size。间隔单位是step。
``` python
'''
*   optimizer：优化器
*   gamma(float)：学习率调整倍数，默认为 0.1 倍，即下降 10 倍。
*   step_size(int)： 学习率下降间隔数，若为 30，则会在 30、 60、 90…个 step 时，将学习率调整为 lr*gamma。(step 通常是指 epoch，不要弄成 iteration 了)
*   last_epoch(int)：上一个 epoch 数，这个变量用来指示学习率是否需要调整。当last_epoch 符合设定的间隔时，就会对学习率进行调整。当为-1 时，学习率设置为初始值。
'''
torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
```

``` python
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision.models import AlexNet
import matplotlib.pyplot as plt

model = AlexNet(num_classes=2)
optimizer = optim.SGD(params=model.parameters(), lr=0.01)
# ----------------------------------------------------------------------------------------------------------------------------------------
# 官方用法
# lr_scheduler.StepLR()
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90
#
# scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()
# ----------------------------------------------------------------------------------------------------------------------------------------
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
plt.figure()
x = list(range(100))
y = []
for epoch in range(100):
    scheduler.step()
    lr = scheduler.get_lr()
    print(epoch, scheduler.get_lr()[0])
    y.append(scheduler.get_lr()[0])
plt.xlabel("epoch")
plt.ylabel("learning rate")
plt.plot(x, y)
```

![Image 4: 在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/5dc26aea3aeb831326fc18a08d1c10fc.png)

------------------

### 2、按需调整学习率 MultiStepLR

> 按设定的间隔调整学习率。这个方法适合后期调试使用，观察 loss 曲线，为每个实验定制学习率调整时机。
`与StepLR的区别是，调节的epoch是自己定义，无须一定是等差数列；请注意，这种衰减是由外部的设置来更改的`

``` python
'''
*   milestones(list)：一个 list，每一个元素代表何时调整学习率，list 元素必须是递增的。如milestones=[30,80,120]。
*   gamma(float): 学习率调整倍数，默认为 0.1 倍，即下降 10 倍。
*   last_epoch(int)：上一个epoch数，这个变量用来指示学习率是否需要调整。当last_epoch符合设定的间隔时，就会对学习率进行调整；当为-1时，学习率设置为初始值。
'''
torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
```

``` python
model = AlexNet(num_classes=2)
optimizer = optim.SGD(params = model.parameters(), lr=0.01)
# ----------------------------------------------------------------------------------------------------------------------------------------
# Assuming optimizer uses lr = 0.05 for all groups
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 80
# lr = 0.0005   if epoch >= 80
# scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()
# ----------------------------------------------------------------------------------------------------------------------------------------
#在指定的epoch值，如[5,20,25,80]处对学习率进行衰减，lr = lr * gamma
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[5,20,25,80], gamma=0.1)
plt.figure()
x = list(range(100))
y = []
for epoch in range(100):
    scheduler.step()
    lr = scheduler.get_lr()
    print(epoch, scheduler.get_lr()[0])
    y.append(scheduler.get_lr()[0])
plt.xlabel("epoch")
plt.ylabel("learning rate")
plt.plot(x,y)
```

------------------

### 3、指数衰减调整学习率 ExponentialLR

``` python 
'''
*   gamma：学习率调衰减的底数，选择不同的gamma值可以获得幅度不同的衰减曲线，指数为 epoch，即 gamma^{epoch}
'''
torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
```

![Image 5: 在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/bf9c8cf46d9bc9243bf432f1c10a4c0e.png)

------------------

### 4. Cosine learning rate decay

cosine decay,是让学习率随着训练过程曲线下降。  
对于cosine decay，假设总共有T个batch（不考虑warmup阶段），在第t个batch时，学习率η\_t为:

![](http://i-blog.csdnimg.cn/blog_migrate/aa9e8828f2e7696d40a0472585a7f34a.jpeg)  

![](http://i-blog.csdnimg.cn/blog_migrate/a053ee6359837c0a642417ec033d1b49.png)

> ___注意___：
> - 图中的lr是lambda1\*lr\_rate的结果;
> - 便于工程上的运用，起始学习率=0.00035，尾端防止学习率为0，当lr小于0.00035时，也设成0.00035。

``` python
lambda1 = lambda epoch: (0.9*epoch / t+0.1) if epoch < t else 0.1 if n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))<0.1 else n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))
```

``` python
# -*- coding:utf-8 -*-
import math
import matplotlib.pyplot as plt
import torch.optim as optim
from torchvision.models import resnet18

t=10  #warmup
T=120 #共有120个epoch，则用于cosine rate的一共有110个epoch
lr_rate = 0.0035
n_t = 0.5
model = resnet18(num_classes=10)
lambda1 = lambda epoch: (0.9*epoch / t+0.1) if epoch < t else 0.1 if n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))<0.1 else n_t * (1+math.cos(math.pi*(epoch - t)/(T-t)))
optimizer = optim.SGD(model.parameters(), lr=lr_rate, momentum=0.9, nesterov=True)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

index = 0
x = []
y = []
for epoch in range(T):
    x.append(index)
    y.append(optimizer.param_groups[0]['lr'])
    index += 1
    scheduler.step()

plt.figure(figsize=(10, 8))
plt.xlabel('epoch')
plt.ylabel('cosine rate')
plt.plot(x, y, color='r', linewidth=2.0, label='cosine rate')
plt.legend(loc='best')
plt.show()
```

除此还有torch.optim.lr_scheduler提供的CosineAnnealingLR，以初始学习率为最大学习率，以 2 ∗ T \_ m a x 2 ∗ T\\\_max 2∗T\_max 为周期，在一个周期内先下降，后上升。

```python
'''
*   T_max(int)：学习率下降到最小值时的epoch数，即当epoch=T_max时，学习率下降到余弦函数最小值，当epoch>T_max时，学习率将增大。
*   eta_min(float)：学习率的最小值，即在一个周期中，学习率最小会下降到 eta_min，默认值为 0。
*   上一个epoch数，这个变量用来指示学习率是否需要调整。当last_epoch符合设定的间隔时，就会对学习率进行调整；当为-1时，学习率设置为初始值。
'''
torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=0, last_epoch=-1)
```

![Image 6: 在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b98c7f1cb4a112b30a04b8d0ae242742.png)

------------------

### 5、自适应调整学习率 ReduceLROnPlateau

当某指标不再变化（下降或升高），调整学习率，这是非常实用的学习率调整策略。例如，当验证集的 loss 不再下降时，进行学习率调整；或者监测验证集的 accuracy，当accuracy 不再上升时，则调整学习率。

``` python
'''
*   mode(str)- 模式选择，有 min 和 max 两种模式， min 表示当指标不再降低(如监测loss)， max 表示当指标不再升高(如监测 accuracy)。
*   factor(float)- 学习率调整倍数(等同于其它方法的 gamma)，即学习率更新为 lr = lr * factor。
*   patience(int)- 忍受该指标多少个 step 不变化，当忍无可忍时，调整学习率。
*   verbose(bool)- 是否打印学习率信息， print(‘Epoch {:5d}: reducing learning rate of group {} to {:.4e}.’.format(epoch, i, new_lr))。
*   threshold_mode(str)- 选择判断指标是否达最优的模式，有两种模式， rel 和 abs。
*   cooldown(int)- “冷却时间”，当调整学习率之后，让学习率调整策略冷静一下，让模型再训练一段时间，再重启监测模式。
*   min_lr(float or list)- 学习率下限，可为 float，或者 list，当有多个参数组时，可用 list 进行设置。
*   eps(float)- 学习率衰减的最小值，当学习率变化小于 eps 时，则不调整学习率。
'''
torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
```

> * 当 threshold\_mode == rel，并且 mode == max 时，dynamic\_threshold = best \* (1 + threshold)；
> * 当 threshold\_mode == rel，并且 mode == min 时，dynamic\_threshold = best \* (1 - threshold)；
> * 当 threshold\_mode == abs，并且 mode == max 时，dynamic\_threshold = best + threshold； 
> * 当 threshold\_mode == rel，并且 mode == max 时，dynamic\_threshold = best - threshold； threshold(float)- 配合 threshold\_mode 使用。

------------------

### 6、自定义调整学习率 LambdaLR

为不同参数组设定不同学习率调整策略。这在fine-tune 中十分有用，我们不仅可为不同的层设定不同的学习率，还可以为其设定不同的学习率调整策略。

``` python
'''
*   lr_lambda：计算学习率调整的函数或者函数列表。
*   lr_lambda(function or list): 一个计算学习率调整倍数的函数，输入通常为 step，当有多个参数组时，设为 list。
'''
torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
```


------------------

### 总结

选择合适的学习率调整策略需要综合考虑任务类型、模型复杂度、训练资源以及经验等因素。在实践中，通常会根据具体情况进行调整和优化，以获得最佳的训练结果。