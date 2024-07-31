---
layout:     post
title:      "AI图片、视频自动抠图工具"
subtitle:   "自动抠图工具来啦！"
date:       2022-04-22 16:50:00
author:     "CS"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Python
    - Matting
    - tool
---

@[TOC](文章目录)

---

# 前言
`本人之前相对视频做一个抠图处理，奈何手工逐帧抠肯定不行，本来想在网上找找有没有好用且免费的抠图工具，看到有非常多人推荐一款在线抠图的网站，叫什么无绿幕抠图。我就抱着好奇点开试了一试，结果效果还是很差，而且只是生成了gif动图。综合观察，奈何要么收费，要么就是效果很差。`

所以，本人也就搜集了一下目前比较好的抠图算法，看到一篇《Robust High-Resolution Video Matting with Temporal Guidance》论文。于是想着借鉴一下制作一个可以直接供人拿来用的抠图工具。但是效果还不是很好，而且工具还有一些bug，不过现在先凑合用吧，有需要的人可以试一试。

---

`提示：工具只是个半成品，打开、关闭还有处理过程中还没完善。`
# 一、Video、Picture处理效果
示例：在人像背景较为单调的场景，效果还是不错。但是背景与人像颜色相近的地方效果较差，这也是目前大多数算法主要解决的问题。
## 1.图片抠图效果

![原图](https://i-blog.csdnimg.cn/blog_migrate/d4b85bf7302fd0d3aebcf1828c553ba7.jpeg =400x)
 <center>1.1原图</center>
 

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b39b0350ec549bcf9d7ba4ff4dc65b4f.png =400x)
 <center>1.2 其他方法效果</center>

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/49ce707904142c09266a18a1d928acb1.jpeg =400x)
<center>1.3 本工具效果</center>

![原图](https://i-blog.csdnimg.cn/blog_migrate/3b9a8756b5fa8a37a363d3f4574801d9.jpeg =400x)
<center>2.1 原图</center>

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/38f5440c9d20e454c4f1e5509e1cc816.png =400x)

<center>2.2 其他方法效果</center>

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ba78dfeee7aaaa64be9aaabf0209da4b.jpeg =400x)
<center>2.3 本工具效果</center>

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/58737243408470ae89a4e9d0511c1d06.jpeg =400x)
<center>3.1 原图</center>

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a5f45963c3da1a3c6ede9a63e01f4573.png =400x)
<center>3.2 其他方法效果</center>

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ebdc9a08a5ce7f06f4ba8104cd87e3c4.jpeg =400x)
<center>3.3 本工具效果</center>

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0979c75fa2143bd3b826535e00ed3aae.jpeg =400x)
<center>4.1 原图</center>

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/80567ede3e4c3159461aa01bd00d6212.png =400x)

<center>4.2 其他方法效果</center>

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/06bf4b069e1c8c53be79a29ae0637017.jpeg =400x)
<center>4.3 本工具效果</center>

## 2.视频抠图效果



![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/8e015481eed018ddb7929002a1b310b8.gif =400x)<center>2.1 原视频</center>
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/e1e5e7d00932da633ea79f3a1a7efbd3.gif =400x)
<center>2.2 这是无绿幕免费在线处理效果</center>

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/079fcbb30675b5202179982ace0517ae.gif  =400x)
<center>2.3 本工具处理效果</center>

总结：
相较于免费的抠图工具，视频抠图效果还好，但是还有一些较为模糊的画面处理的不是很好。在图片抠图效果上，某些场景可能效果较差。
# 二、采用的方法
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/1c920e47fbd529aaf4b49ac1a0e56401.png)

作者原代码为pytorch环境上训练的类Unet结构模型。主要分类编码、解码与上采样输出几个部分，具体不再赘述，可参考一下别人的[分析博客](https://blog.csdn.net/m_buddy/article/details/120298395)。
训练所使用的数据集包括：人像语义分割数据集、matting抠图数据集。采用的损失包括了：alpha通道L1损失与拉普拉斯金字塔损失、时序相关性损失、fg部分的L1损失和时序相关损失。将模型权重文件转换为onnx模型，本来想做个量化，所以现在处理时间可能稍久···，所以可以考虑后续加快预测处理速度，增加半精度选择、模型剪枝量化处理吧。

# 三、使用步骤
直接打开根目录下的Matting tool by CS.exe。
## 1.Data选择
*Input Dir*：选择所需处理的视频、图片；
*Output Dir*：选择保存位置文件夹；
*Model weight*：模型权重，选择weights文件夹下的model.onnx；
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/27e2316c81146189da2aa64b7368a5df.png =600x)
## 2.Advance设置
*Show Result*：可以实时显示处理过程（但由于显示控件问题，弹出显示了），但是由于cv2显示会影响处理速度，可以考虑去掉勾勾；
*mode*：选择处理的是视频还是图片；
*background color*：选择抠图背景颜色，默认绿幕；
## 3.Start Run
点击*Start Run*按钮开始处理，可通过命令行窗口查看当前处理进度（注意处理时不要关闭这个窗口）。如果不想处理了，可以点击*Close Run*按钮结束处理进程。
最终输出为output(+audio).mp4视频或者out.jpg图片。
工具可从网盘获取，目前bug较多，效果有限，仅供娱乐。
···[百度云盘](https://pan.baidu.com/s/1XJvD893vrBfQxnmOdbmD1A?pwd=1122)··· 提取码：1122
# 总结
训练效果有限，处理速度受限于设备配置情况，后续考虑做个剪枝量化处理。
---

<font color=#BE2528 size=5>**更新：V1.3**</font>
又更新啦，有小伙伴反馈需要**批量处理图片**，我将原始功能按键做了修改，将输入改为了*Input Images*(选择多张图片的文件夹)和*Input Video*(选择单个视频文件)。从而可以批量处理图片了。
更新[b]下载地址在这里：
···[百度云盘](https://pan.baidu.com/s/1Eg2W927Gffqpv5-Vz6-tcw?pwd=1210)···
提取码：1210。
下载安装软件，里面有个小彩蛋，喜欢的可以打赏哦，感谢！
视频插帧软件还在加急制作中，五一劳动节快乐！

