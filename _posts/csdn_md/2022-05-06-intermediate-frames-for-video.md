---
layout:     post
title:      "AI视频插帧 附带『视频插帧』工具"
subtitle:   "视频插帧工具来啦！"
date:       2022-05-06 18:49:08
author:     "CS"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Python
    - Interpolation
    - tool
---

# AI视频插帧 附带『视频插帧』工具
` 视频插帧工具来啦！`
**下载**链接在最下面。
# 前言

> - 继视频抠图工具以来，本人又考虑制作一款视频插帧的工具，最近一直在改各种问题（头都大了- _ -），还好该来的终于来了(^ _ ^)。现在自媒体越来越流行，很多人都开始自己做个小视频玩玩，各大视频平台也都开放了高刷视频功能。这次的灵感也就来源于之前搜了一搜目前视频插帧的工具和方法，要么下载各种乱七八糟的软件，要么就是折腾好一阵效果还是差强人意，总之很麻烦还浪费时间。于是还是想弄个几键就开始处理的，不需要配置太多就能用的工具。
>  - 所以，我就搜集了一下目前插帧常用的方法，基本都是基于光流法，网上一搜基本都是用SVP4的视频渲染软件，把视频提到60帧，但是这个下载和配置太麻烦了，而且仅仅也就60帧，再想提高就得掏money了。
>  - 之前看过一篇《Super SloMo: High Quality Estimation of Multiple Intermediate Frames for Video
> Interpolation》论文。于是想在其基础上优化一下，弄个轻量一点的模型。奈何效果还是有限，电脑跑了好久模型，处理速度还是太慢了。由于本人电脑用的nvidia显卡cuda加速，可能目前还有大部分电脑只能用cpu处理，所以速度太慢了。如果有人仅需处理一小段视频的话，可以下载用一用。

`提示：工具缺点就是处理速度较慢，还有一些功能后续再完善。`
<html ><hr> </html>



# 一、视频插帧效果

最终实现：**帧数翻倍**。
效果：在每帧图片较为清晰的情况下，效果还是可以的。但是是在图像较为模糊、两段视频转场时，效果还是差强人意。
## 1.效果
运动一般较为模糊，也是受影响较大的部分，由于原始视频帧就比较模糊，所以仅仅是提升了高刷新率的模糊帧，但是观感还是更舒服了一点，如果原始视频质量高的话，效果就会更好。
<center><img src="https://i-blog.csdnimg.cn/blog_migrate/8f1262d3db24cdd88a170941490cfebc.gif"   width="60%">
<center><b><font size ='3'>1.1 原始30fps</font></b></center></font>

<center><img src="https://i-blog.csdnimg.cn/blog_migrate/aef0446dcd33572da4fccd43b2b5017e.gif"   width="60%">
<center><b><font size ='3'>1.2 120fps</font></b></center></font>

> 总结：相较于其他方法，使用较为简便，但是由于每个人电脑配置不同，移植到不同设备上可能会有不同的bug，如果能用GPU加速的就用，如果不能用GPU的，我就不太建议视频插帧了，因为处理速度太慢了。目前视频插帧的普遍问题就是处理速度，这个受硬件限制太大了。但是你电脑不支持GPU加速也想尝试一下的话，我也加入了取消勾选GPU的按键，利用cpu来处理，但仅供娱乐了。
<html ><hr> </html>

# 二、采用的方法
<center><img src="https://i-blog.csdnimg.cn/blog_migrate/9a98e30b2e17f1d52c4dc42c021cf1f7.png"   width="80%">

&emsp;&emsp;作者原代码同样为pytorch环境上训练的类Unet结构模型。根据生成的双向光流与伪标签构建的无监督模型，这里供上人家的[文章]。对于这个模型，本人对其进行轻量化改进，引入MobileNet的可分离卷积，使得处理速度与模型大小稍有改进，但还是有限。
&emsp;&emsp;训练所使用的数据集使用了adobe240fps，对其进行了处理。采用的损失包括了：重建损失（lr）、感知损失（lp）、平滑损失（ls）和wrap损失。对于可用GPU的cuda加速的情况，我对模型数据采用半精度来提升处理速度，模型结构也稍作改变。但还是胳膊拧不过大腿，对于这种图像生成的模型，需要消耗比较多的时间。
&emsp;&emsp;由于基于torch框架，其所占空间较大，所以这次的工具占的空间也就较大。
<html ><hr> </html>


# 三、使用步骤
#### 判断GPU是否可用
```c
如何判断电脑是否支持GPU cuda加速:
1.首先是nvidia品牌，且cuda版本>=10.2，安装好显卡驱动。
2.其次可通过按下win+R组合键，打开cmd命令窗口。输入nvidia-smi命令，查看CUDA Version版本。
  如果CUDA Version<=10.2,尝试更新显卡驱动，看看显卡是否支持更高版本的cuda。
```
**直接打开Interpolation tool by CS.exe**，选择安装位置进行安装，并生成桌面快捷方式，然后可以直接打开。

**1.Data选择**
 - *Input Dir*：选择所需处理的视频；
- *Output Dir*：选择保存位置文件夹；

**2.Advance设置**
 - *GPU*：选择是否使用GPU。默认勾选，（如果运行时报错，可尝试判断自身显卡是否支持cuda10.2版本以上加速，若支持可尝试更新显卡驱动提升cuda版本；若不支持就取消勾选，采用cpu处理，但是速度非常慢）；
 - *BatchSize*：选择批处理大小。一般太大会爆显存或内存，自己尝试找到适合自己电脑的最大值（一般不超过5）；
 - *ScaleRate*：帧率提升的倍数。一般30帧视频可以选择设置为3倍，提升至90帧（帧数提升倍数越多，处理时间也会跟着翻倍，所以超过120帧就没必要了）；
 
 <center><img src="http://i-blog.csdnimg.cn/blog_migrate/e15f03e0ec91f4f13a0886fc913d4bed.png"   width="80%">
<center><b><font size ='3'>主界面</font></b></center></font>

**3.Start Run**
 - 点击*Run*按钮开始处理，可通过命令行窗口查看当前处理进度（注意处理时不要关闭这个窗口）。
 - 如果不想处理了，可以点击*Stop*按钮结束处理进程。
 
最终输出为output(+audio).mp4视频。
<html ><hr> </html>



**4.错误提示的解决**
**错误一：**

> 不支持显卡GPU cuda加速。 根据上面的方法：判断GPU是否可用，来自行判断。如果不支持GPU，则取消勾选GPU按钮再尝试重新运行。
 <center><img src="http://i-blog.csdnimg.cn/blog_migrate/9d4ee426b03984204666254f05c529d7.png"   width="80%">
<center><b><font size ='3'>错误一</font></b></center></font>

**错误二：**

> BatchSize设置太大了，爆显存、内存了，尝试调小。如果调到2还是爆内存，那就是电脑配置不行了。

 <center><img src="http://i-blog.csdnimg.cn/blog_migrate/f5787448a38f3825ab0403494f8ac701.png"   width="80%">
<center><img src="http://i-blog.csdnimg.cn/blog_migrate/33241d0e3aa206c1f4cedd48c7f72329.png"   width="80%">

<center><b><font size ='3'>错误二</font></b></center></font>

**错误三：**

> *Input Dir*、*Output Dir* 输入、输出路径没有选择。

 <center><img src="http://i-blog.csdnimg.cn/blog_migrate/a0abb8e2942f7a2d0badca76c18c3102.png"   width="80%">
<center><b><font size ='3'>错误三</font></b></center></font>

**成功运行：**

> 分别显示的是：*完成百分比*、*已运行时间*、*预计剩余时间*。

 <center><img src="http://i-blog.csdnimg.cn/blog_migrate/878add529f9c30e849f72e29bc4819cc.png"   width="80%">
<center><b><font size ='3'>成功运行</font></b></center></font>

# 总结


工具可从[网盘](https://pan.baidu.com/s/1WaKJpSJE_B1iT_oiddRhyw?pwd=1210)获取，提取码：1210
目前bug较多，处理速度和效果有限，硬件依赖大，仅供娱乐。
`制作不易，主界面有个小彩蛋，可以打赏一下呦，感激不尽! `
`如果大家喜欢的话，后面可以考虑再出个增强图片、视频分辨率的工具，让你的人像与场景更加清晰！`