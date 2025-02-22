大家好，我是CS(CHenSHuai)，一名算法工程师，主要从事与机器视觉、计算机图像学和深度学习等领域。我热爱计算机科学和人工智能领域，致力于将先进的算法应用于实际问题中，解决现实世界中的挑战。

#### 联系我

- **电子邮箱**：cs1179593516@gmail.com
- [**Github**](https://github.com/CSsaan)：[https://github.com/CSsaan](https://github.com/CSsaan)

#### 技术栈

- **编程语言**：C/C++, Python, Java, GLSL
- **框架和库**：TensorFlow, PyTorch, OpenCV, OpenGL, Qt, MNN, NCNN, CMake, ImGui
- **算法领域**：计算机视觉,  深度学习, 生成式模型, LLM
- **开发工具**：Visual Studio, Visual Studio Code, PyCharm, Jupyter Notebook, Adroid Studio, Git, Docker

#### [**开源项目**](/my/open-sources/)

- **Computer Vision**
  - [Deep Learning of Machine Vision](https://github.com/CSsaan/Deep-learning-of-machine-vision)：探索深度学习在计算机视觉领域的应用。
  - [Vision-Trasformers Matting](https://github.com/CSsaan/EMA-ViTMatting#ema-vitmatting)：基于Transformer的人像抠图技术。
  - [YOLO Auto Target](https://github.com/CSsaan/YOLO_AutoTarget): YOLO FPS 游戏目标检测。
  - [GoogLeNet Inception/Alexnet/ResNet/VGG/ViT](https://github.com/CSsaan/EMA-GoogLeNet/tree/main)：实现经典的卷积神经网络架构。
  - [MNN with QT](https://github.com/CSsaan/qtMnn)：将MNN模型推理框架与QT框架结合，实现跨平台的深度学习应用。
  - [OpenCV Demo](https://github.com/CSsaan/OpenCVtest)：OpenCV库的示例代码和应用。
- **LLM**
  - [Lover LLaMA](https://gitee.com/cehs/lover_llama)：LLaMA的恋爱对话模型，包括数据集、微调、量化、合并、api部署等。
- **Android & windows Aplication**
  - [Android Camera Special Effect by CS](https://github.com/CSsaan/Camera-Special-Effect-Face-Reshape):安卓美颜相机，瘦脸、大眼、亮牙。使用OpenCV+MNN+NCNN+GLM+OpenGLES3.0实现。
  - [BokehDepthByCS](https://github.com/CSsaan/BokehDepthByCS):WindowsAI实时虚化软件。这是一个 C++17 CMake OpenGL 项目，模型使用 NCNN 和 MNN 部署，它包括以下render库：GLFW、Glew和glm，它还使用ImGui作为 GUI。
  - [Matting Tool by CS](https://blog.csdn.net/sCs12321/article/details/124331491)：基于深度学习的人像抠图技术（全网下载量1W+）。
  - [Interpolation Tool by CS](https://blog.csdn.net/sCs12321/article/details/124550893)：基于深度学习的图像插值技术。
- **OpenGL/Shader**
  - [7 Sharpen Methods](https://blog.csdn.net/sCs12321/article/details/129459772): 7种锐化方法介绍与C++、Python、shader、GLSL实现。
  - [OpenGL CMake by CS](https://github.com/CSsaan/OpenGL_CMake_CS)：一键使用CMake构建OpenGL项目，结合ImGui。
  - [OpenGL with QT](https://github.com/CSsaan/HelloOpenGL)：使用Qt来构建OpenGL项目。
  - [OpenGL Colors Filters](https://github.com/CSsaan/OpenGL-colors-filters)：OpenGL颜色滤镜。
  
#### 项目经历

- **XX公司**： 2022-至今：算法工程师
  - 基于图传监视器的视频分析与视频处理`技术负责人`   ——   2019年10月-至今
    - 项目主要内容：在 Android/IOS 手机设备实时远程监控图传数据，支持软解/硬解下 Full Range/Limited Range，并对视频流数据进行实时高性能的图像处理、图像分析、渲染等。
      - 高性能 & 低功耗图像处理：NV12、YUYV视频源的图像实时60fps处理，内存优化，零拷贝纹理渲染，离线渲染，多处理的高性能叠加；
      - 全功能：实现比例缩放、3DLUT、九宫格、伪彩色、双阈值斑马纹、[锐化](https://blog.csdn.net/sCs12321/article/details/129459772)、放大镜、伪彩色、峰值对焦、波形图、直方图、矢量图、ToF点云渲染等。
      - 跨平台：部署落地在 Android、iOS；输出 Windows、Linux 端 Demo。
    - 项目收益：近一年（2023.12-2024.12）仅在 Apple Store 中下载量为34.5W（Android 暂无具体统计参数）。
  - 直播美颜`项目负责人`   ——   2023年11月-至今
    - 项目主要内容：Android(高通8250)、Windows(GTX1050Ti及以上)平台下，支持任意直播场景中实时美颜处理，提升皮肤美感，弥补ISP中人脸稍暗缺陷，提升直播质量。AI结果转零拷贝纹理耗时优化，OpenGL线程状态同步优化，预初始化AI加载优化，链路耗时波动优化，最终渲染耗时少于5ms，可以稳定30fps推流。
      - 贴纸：主要负责链路中图像算法与渲染实现，具体包括：贴纸的平移、旋转、缩放、镜像等；美颜中人脸贴纸图层对齐人脸关键点的三角剖分与渲染。
      - 皮肤分割：采用轻量化皮肤matting模型，优化不同肤色五官区域数据集优化处理，并制作优化Trimap与alpha数据集。准确识别和处理皮肤区域，提升美颜效果的针对性和精确度。采用snpe2.22中PTQ量化，snpe DSP 推理后端，链路上采用分割模型并行推理方式，每帧整体链路耗时大幅减少14ms。
      - 自研绿幕抠图：高效地去除视频背景，实现前景与背景的精确分离。涵盖了颜色转换、距离计算、图像腐蚀、高斯模糊、前景合成等功能。将 RGB 颜色空间转换为 YUV 颜色空间，用于计算与目标绿色的距离，提高抠图精度。图像处理：实现了基于距离图的腐蚀与高斯模糊处理，平滑边缘，减少噪声，提高合成效果。前景合成：根据 alpha 通道与前景图像进行精确合成，确保前景与背景的自然过渡。
      - 美白LUT调色模块：根据不同程度生成美白 .cube 与 .3dl，渲染3DLut。采用pbo零拷贝方案，隔帧方案。后处理将AI皮肤分割结果结合传统皮肤检测方法（高斯概率分布）、卡尔曼滤波，增强皮肤检测准确性与稳定性。
      - 磨皮: 基于OpenGL实现磨皮算法，解决传统磨皮效果过于生硬的问题，确保皮肤细腻自然。解决gamma提亮方法中皮肤被提亮问题，并通过结合高斯权重与颜色距离权重方法解决边缘溢色的问题。
      - [瘦脸\大眼\亮牙(预研)](https://github.com/CSsaan/Camera-Special-Effect-Face-Reshape) 实现瘦脸、大眼、亮牙等特效，增强用户的个性化需求，提升直播的趣味性和吸引力。
    - 项目收益：正式上线公司相机产品与直播平台。
  - 实时alpha抠图模型及部署`项目负责人`   ——   2023年11月-至今
    - 项目主要内容：将单人或多人(5人)环境下人体皮肤区域进行快速、稳定地alpha抠图。
      - 模型优化：基于 MobileNetV2 backbone 的 deeplabV3+ 模型，采用四通道输入（通过不降分辨率，而采用叠加通道方式来提升性能），将模型输入层通道维数3*480*480改为4*(3*320*320)，可将耗时减少约6ms，同时反卷积替换resize算子，以适应在alpha Matting任务中的高性能低延时需求。
      - 帧间稳定性：基于人脸检测区域进行卡尔曼滤波，实现帧间稳定性。
      - 模型部署：移动端采用snpe的PTQ量化方式，运行在DSP设备上; PC端采用NCNN进行部署。采用OpenCL进行前处理,在保持精度的同时,推理耗时11ms。
  - AI虚化直播:`技术负责人`   ——   2023年11月-至今
    - 项目主要内容：[AI深度估计模型](https://github.com/CSsaan/GitPod_Python/tree/main/Depth-Anything-V2-with-OpenGLBokeh-ONNX)，[模拟DSLR的光斑模糊渲染](https://blog.csdn.net/sCs12321/article/details/143893389)，模仿传统摄影大光圈浅景深的效果。
      - 深度估计模型: 采用 DPT V2 单目深度估计模型，实现DataLoder与训练代码。同时提供轻量化的PyDNet2单目深度估计模型，模型转换MNN/NCNN格式，编写C++推流代码。
      - 对焦距离帧间稳定性: 基于人脸检测区域进行卡尔曼滤波，实现帧间稳定性。
      - 深度图帧间稳定性: 基于帧卡尔曼滤波，实现模拟DSLR的光斑时的模糊稳定。
      - DSLR渲染：结合AI深度图、人像抠图mask、DP等不同格式类型的输入，来提升虚化准确性与效果。使用OpenGL模拟不同镜头F值、焦段下的散景效果，相较于市面上其他主流竞品的效果更加真实。
  - 视频降噪：`技术负责人`   ——   2023年11月-至今
    - 项目主要内容：基于nlMeans的视频降噪，降低传感器存在的亮度噪声与彩色噪声。
    - 性能：基于GPU加速的OpenGL优化实现，满足60fps的视频降噪。

#### 论文专利

[1]. Radar Reflectivity and Meteorological Factors Merging‐Based Precipitation Estimation Neural Network[J].Earth and Space Science,2021,8(10). [[pdf]](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2021EA001811)

[2]. Offline Single-Polarization Radar Quantitative Precipitation Estimation Based on a Spatiotemporal Deep Fusion Model[J].Advances in Meteorology,2021.

[3]. A Cloud-Removal Method for Snow Product Based on Denoising Autoencoder Neural Network[J].Journal of Nanjing University of Information Science and Technology (Natural Science Edition),2023,15(02).

[4]. Cloud-Removal Algorithm and Application Research for Snow Product at Basin Scale[D].2022.DOI:10.27248/d.cnki.gnjqc.2022.001085.

[5]. A Vehicle Vibration Noise Detection Alarm Device[P].Jiangsu Province:CN202120495866.X,2022-03-15.

#### 教育背景

- **南京信息工程大学**：2019-2022：硕士研究生
- **南京信息工程大学**：2015-2019：本科

#### 个人博客

- [我的个人博客](https://www.csblog.site/)：分享技术文章、项目经验和行业动态。

#### 社交媒体

- [**知乎**](https://www.zhihu.com/people/ou-nuo-40)：[https://www.zhihu.com/people/ou-nuo-40](https://www.zhihu.com/people/ou-nuo-40)
- [**CSDN**](https://blog.csdn.net/sCs12321/)：[https://blog.csdn.net/sCs12321/](https://blog.csdn.net/sCs12321/)

我热爱探索新技术，解决实际问题，享受与同行交流与合作。如果您对我的工作感兴趣或有合作意向，请随时联系我。我期待与您一起探索计算机科学的无限可能！