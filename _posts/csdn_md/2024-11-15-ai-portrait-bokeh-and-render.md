---
layout:     post
title:      "人像模式与AI虚化效果渲染"
subtitle:   "主要侧重于虚化效果和从测试角度的一些关注点和注意事项，分别对比了目前最新几款手机的AI虚化效果"
date:       2024-11-15 16:11:00
author:     "CS"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Python
    - Portrait Bokeh Render
---

## 1. 引言

手机的**人像模式**就是模仿传统摄影大光圈浅景深的效果。

**虚化散景**是在一个专门的模式下呈现的，通常是人像或光圈模式。人像摄影通常将拍摄人物对象置于一个模糊的焦外背景的前面，以凸显人物。

在**物理定律**的基础上，使用大光圈镜头与大画幅图像传感器的相机可以轻松实现背景虚化。
景深的深度与光圈大小呈反比关系，<u>光圈大的镜头一般都贵；此外，由于其较小的图像传感器，难以有效模糊背景，导致拍摄对象和背景的清晰度相近。</u>

智能手机制造商采用了多种方法和技术来实现计算散景效果，包括焦点堆叠、双摄像头（立体视觉）、双像素技术，以及专用的景深传感器。

本文将对比几款主流手机产品的人像模式效果。

### 1.1 目的

- 对比几款主流产品，了解它们在人像模式或AI虚化功能上的表现和差异。
- 从技术角度对比几款主流产品，了解它们在这个功能上的优势和劣势。
- 从用户体验角度对比几款主流产品。

---

## 2. 技术与效果分析

通常将该任务分为: **深度估计**、**语义分割**、**经典渲染**。

**手机深度传感器：**

- iPhone 12 Pro起开始搭载激光雷达（LiDAR/ToF）
- [DP](https://zh.wikipedia.org/wiki/Dual_Pixel)(Dual Pixel)、[QP](https://zhihuhuhu.blogspot.com/2024/04/Quad-Pixel-Tech-Lookback-Outlook.html)(Quad Pixel, 或称Quad Bayer、4Cell) （与传感器类型有关）
- 双目视觉（分辨率更高）
- 单目深度估计（AI，人像模式还需人像抠图模型）

**虚化处理类型：**

- 图像处理：对处理时间要求比较宽松，可以采用更加高精度的算法。
- 视频处理：需要在33ms内完成，

### 2.1 虚化方式

虚化场景：图像虚化、视频虚化。

虚化方式：基于AI人像抠图的两距离平面虚化；AI深度估计景深虚化（物体与风景虚化）；AI人像抠图+深度估计；端到端的深度学习解决方案。

**AI人像抠图:**
<table>
  <tr>
    <td><img src=".\img\image0.png"/></td>
    <td><img src=".\img\image.png"/></td>
  </tr>
</table>

**深度估计:**
<table>
  <tr>
    <td><img src=".\img\image-1.png"/></td>
    <td><img src=".\img\image-2.png"/></td>
  </tr>
</table>

### 2.2 技术特点

- **实时处理**：是否支持实时处理，或者实时预览。
  - 在大部分手机中支持实时预览，但精度、效果、帧间连续性差。一般拍摄完成后会进行更加精细化的处理。
- **主体/背景分割**：精确分离拍摄对象和背景是实现数码单反相机效果的关键。
  - 但当前设备受到景深图解析度不足的影响，可能导致边缘不清晰和景深估计错误，尤其在拍摄移动场景时更为明显。
- **模糊渐变平滑度**：模糊强度随景深大小而变化。
  - 基本人像可能只包含两个距离平面，但大多数的场景都具有更复杂的三维构图，例如多平面图像([MPI](https://arxiv.org/abs/2004.11364), Multi-Plane Image)、深度图等。
  - 由于我们拍摄的时候场景并不是只有主体和一个远处的背景，而是在主体前后有不同的前景和背景，所有单纯的人像分割是不能满足的。
  - 把不同物体的地方按照与焦点距离进行恰到好处地“模糊”，才能得到近似于相机光学虚化的效果。因此，平滑的模糊渐变对于生成近乎数码单反相机的散景效果至关重要。
- **散景光点形状**：单反中一般是非圆形，与单反采用的镜头的机械结构形状有关。
  - 一般大光圈时接近为圆形。
  - 大部分手机中默认模拟的光圈形状为圆形，一些后期处理软件中也可以选择为其他形状。
- **取景fov/范围**：一般手机人像模式是使用相机的长焦镜头，它会获得更近的取景。使用长焦镜头的原因：
  - 这是因为要用两个镜头的画面进行比较来获取位置差异得到距离关系，而长焦的画面范围约是主摄的一半，如果在主摄拍的时候用长焦来计算位置差异的话，只有中间一块有数据没法处理四周的部分。
  - 镜头焦距越长景深越短，就算光圈不大也可以获得不错的浅景深，便宜大碗，所以大家都选择用长镜头拍人像。

---

## 3. 效果对比

- *标准：相机恢复默认，打开人像模式，拍摄直出。*

**iPhone 16 Pro Max、三星 Galaxy S24 Ultra、华为 Pura 70 Ultra**
<table>
  <tr>
    <td><img src=".\img\image-4.png"/></td>
    <td><img src=".\img\image-5.png"/></td>
    <td><img src=".\img\image-6.png"/></td>
  </tr>
</table>

> iPhone 16 Pro Max耳环周围有深度伪像(左), 三星 Galaxy S24 Ultra 耳环周围有深度伪像(中), 华为 Pura 70 Ultra出色的主体分割、大光点模拟(右)

**Oppo Find X7 Ultra、iPhone 15 Pro Max、华为 Mate 60 Pro+**
<table>
  <tr>
    <td><img src=".\img\image-7.png"/></td>
    <td><img src=".\img\image-8.png"/></td>
    <td><img src=".\img\image-9.png"/></td>
  </tr>
  <tr>
    <td><img src=".\img\image-10.png"/></td>
    <td><img src=".\img\image-11.png"/></td>
    <td><img src=".\img\image-12.png"/></td>
  </tr>
</table>

> Oppo Find X7 Ultra被摄体隔离准确，细节保留良好(左), iPhone 15 Pro Max被摄体的细微发丝保留不佳，散景形状较好(中), 华为 Mate 60 Pro+细节略有不准确,散景形状失真(右)

**小米 14 Ultra、华为 Pura 70 Ultra**
<table>
  <tr>
    <td><img src=".\img\image-13.png"/></td>
    <td><img src=".\img\image-14.png"/></td>
  </tr>
  <tr>
    <td><img src=".\img\image-15.png"/></td>
    <td><img src=".\img\image-16.png"/></td>
  </tr>
</table>

> 小米 14 Ultra尖峰背景上的深度估计误差(左), 华为 Pura 70 Ultra 较好的深度估计(右)

**Vivo X100 Pro、Vivo X90 Pro、iPhone 15 Pro Max**
<table>
  <tr>
    <td><img src=".\img\image-17.png"/></td>
    <td><img src=".\img\image-18.png"/></td>
    <td><img src=".\img\image-19.png"/></td>
  </tr>
</table>

> Vivo X100 Pro深度估计较准确，模糊梯度较好，散景光斑形状为圆形(左), Vivo X90 Pro模糊梯度较差，散景光斑形状为圆形(中), iPhone 15 Pro Max深度估计较准确，模糊梯度较好，散景光斑形状较小(右)

**荣耀 Magic4 至臻版、华为 P50 Pro、小米 11 Ultra**
<table>
  <tr>
    <td><img src=".\img\image-20.png"/></td>
    <td><img src=".\img\image-21.png"/></td>
    <td><img src=".\img\image-22.png"/></td>
  </tr>
</table>

> 荣耀 Magic4 至臻版眼镜框较准确，模糊梯度较弱(左), 华为 P50 Pro眼镜框较差，模糊梯度几乎无(中), 小米 11 Ultra眼镜框较差，模糊梯度较好(右)

| 功能/产品       | Apple iPhone 15 ProMax | 小米 14 pro      | 华为 Mate 60 Pro+  | PS(Depth Blur)       |
|----------------|----------------------|----------------------|----------------------|---------------------|
| **人像模式**   | 单目即可,双目时效果更好 | 只能双目(后置镜头)    | 多镜头 + AI算法      | AI深度估计 + focus object(保留精准的人物边缘细节) |
| **实时处理**    | 支持                 | 支持                 | 支持                 | (×)需联网进行云端模型推理 |
| **主体背景分割**    | 人像+物体        | 人像+物体             | 人像+物体            | 人像+物体          |
| **模糊渐变平滑度**  | √               | √                     | √                   | √                  |
| **散景形状**       | 圆形             | 圆形                  | 圆形较为不明显        | 圆形            |
| **距离与模糊程度** | √                 | √                    | √                 | √                |

<table>
  <tr>
    <td><img src=".\img\image-3.png"/></td>
  </tr>
  <tr>
    <td class="text-center">距离与模糊程度关系曲线</td>
</table>

<table>
  <tr>
    <td><img src=".\img\bokehd.png"/></td>
    <td><img src=".\img\bokeh.gif"/></td>
  </tr>
  <tr>
    <td class="text-center">深度图</td>
    <td class="text-center">虚化模拟(动态对焦距离，固定对焦范围)</td>
  </tr>
</table>

---

## 4. 结论

在人像模式和AI虚化功能方面，各家目前针对图像虚化的效果均良好，各个产品都有其独特的技术优势。

- 目前手机人像模式主要是针对于照片的优化，在预览中可生效（效果较差，分割伪像较多且存在闪动），在拍摄后，会进一步精细化处理，效果会更好一些。
所以需要考虑视频虚化时的帧间连续性，避免明显的闪动。

- 人像模式中，可以观察到不止人像聚焦，与人像距离相近的物体也会聚焦，可以猜测使用了人像分割与深度图信息。

- 在头发边缘细节处理上有一定的难度，除了前景与背景分割的准确性要求外，发丝、皮肤、衣服边缘细节保留，且模糊处理时与背景混色也需要考虑。

- 聚焦物体的焦平面内要清晰，模糊随着景深不同而不同。自然、真实的虚化散景光点形状也需要考虑。

### 参考文献

[1]. Wolf Hauser, Balthazar Neveu, Jean-Benoit Jourdain, Clément Viard, Frédéric Guichard, "Image quality benchmark of computational bokeh"  in Proc. IS&T Int’l. Symp. on Electronic Imaging: Image Quality and System Performance XV,  2018,  pp 340-1 - 340-10,  https://doi.org/10.2352/ISSN.2470-1173.2018.12.IQSP-340
