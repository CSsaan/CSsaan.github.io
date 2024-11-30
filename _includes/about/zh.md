大家好，我是CS(CHenSHuai)，一名算法工程师，主要从事与机器视觉、计算机图像学和深度学习等领域。我热爱计算机科学和人工智能领域，致力于将先进的算法应用于实际问题中，解决现实世界中的挑战。

#### 联系我

- **电子邮箱**：cs1179593516@gmail.com
- [**Github**](https://github.com/CSsaan)：[https://github.com/CSsaan](https://github.com/CSsaan)

#### 技术栈

- **编程语言**：Python, C/C++, Java, GLSL
- **框架和库**：TensorFlow, PyTorch, OpenCV, OpenGL, Qt, MNN, NCNN, CMake, ImGui
- **算法领域**：计算机视觉,  深度学习, 生成式模型, LLM
- **开发工具**：Visual Studio, Visual Studio Code, PyCharm, Jupyter Notebook, Adroid Studio, Git, Docker

#### [**开源项目**](/my/open-sources/)

- **Android & windows Aplication**
  - [Android Camera Special Effect by CS](https://github.com/CSsaan/Camera-Special-Effect-Face-Reshape):安卓美颜相机，瘦脸、大眼、亮牙。OpenCV+MNN+NCNN+GLM+OpenGLES3.0实现。
  - [Matting Tool by CS](https://blog.csdn.net/sCs12321/article/details/124331491)：基于深度学习的人像抠图技术。
  - [Interpolation Tool by CS](https://blog.csdn.net/sCs12321/article/details/124550893)：基于深度学习的图像插值技术。
- **Computer Vision**
  - [Deep Learning of Machine Vision](https://github.com/CSsaan/Deep-learning-of-machine-vision)：探索深度学习在计算机视觉领域的应用。
  - [Vision-Trasformers Matting](https://github.com/CSsaan/EMA-ViTMatting#ema-vitmatting)：基于Transformer的人像抠图技术。
  - [YOLO Auto Target](https://github.com/CSsaan/YOLO_AutoTarget): YOLO FPS 游戏目标检测。
  - [GoogLeNet Inception/Alexnet/ResNet/VGG/ViT](https://github.com/CSsaan/EMA-GoogLeNet/tree/main)：实现经典的卷积神经网络架构。
  - [MNN with QT](https://github.com/CSsaan/qtMnn)：将MNN模型推理框架与QT框架结合，实现跨平台的深度学习应用。
  - [OpenCV Demo](https://github.com/CSsaan/OpenCVtest)：OpenCV库的示例代码和应用。
- **LLM**
  - [Lover LLaMA](https://gitee.com/cehs/lover_llama)：LLaMA的恋爱对话模型，包括数据集、微调、量化、合并、api部署等。
- **OpenGL/Shader**
  - [7 Sharpen Methods](https://blog.csdn.net/sCs12321/article/details/129459772): 7种锐化方法介绍与C++、Python、shader、GLSL实现。
  - [OpenGL CMake by CS](https://github.com/CSsaan/OpenGL_CMake_CS)：一键使用CMake构建OpenGL项目，结合ImGui。
  - [OpenGL with QT](https://github.com/CSsaan/HelloOpenGL)：使用Qt来构建OpenGL项目。
  - [OpenGL Colors Filters](https://github.com/CSsaan/OpenGL-colors-filters)：OpenGL颜色滤镜。
  
#### 项目经历

- **XX公司**： 2022-至今：算法工程师
  - 基于图传监视器的视频分析与视频处理`技术负责人`   ——   2019年10月-至今
    - 项目主要内容：基于Android/IOS手机设备实时远程监控图传数据，并对视频流数据进行实时高性能的图像处理、图像分析、渲染等。
      - 高性能 & 低功耗图像处理：NV12、YUYV视频源的图像实时60fps处理，内存优化，零拷贝纹理渲染，离线渲染，多处理的高性能叠加；
      - 全功能：实现比例缩放、3DLUT、伪彩色、双阈值斑马纹、[锐化](https://blog.csdn.net/sCs12321/article/details/129459772)、放大镜、伪彩色、峰值对焦、波形图、直方图、矢量图、ToF点云渲染等。
      - 跨平台：部署落地在 Android、iOS；输出 Windows、Linux 端 Demo。
    - 项目收益：近一年（2023.12-2024.12）在 Apple Store 中下载量为34.5W。
  - 直播美颜`项目负责人`   ——   2023年11月-至今
    - 项目主要内容：支持任意直播场景中实时美颜处理，提升皮肤美感，弥补ISP中人脸稍暗缺陷，提升直播质量。
      - 皮肤分割：
      - 自研绿幕抠图：
      - 美白LUT调色模块：
      - 磨皮:
      - [瘦脸\大眼\亮牙](https://github.com/CSsaan/Camera-Special-Effect-Face-Reshape)
  - 实时alpha抠图模型及部署`项目负责人`   ——   2023年11月-至今
    - 项目主要内容：将单人或多人(5人)环境下人体皮肤区域进行快速、稳定地alpha抠图。
      - 数据采集：
      - 模型设计：基于 MobileNetV2 backbone 的deeplabV3+模型，采用四通道输入（通过不降分辨率，而采用叠加通道方式来提升性能），改进输出Head、损失函数来适应在alpha Matting任务中的需求。
      - 帧间稳定性：
      - 模型部署：移动端采用snpe的PTQ量化方式，运行在DSP设备上; PC端采用NCNN进行部署。采用OpenCL进行前处理,在保持精度的同时,推理耗时11ms。
  - AI虚化直播:`技术负责人`   ——   2023年11月-至今
    - 项目主要内容：[AI深度估计模型](https://github.com/CSsaan/GitPod_Python/tree/main/Depth-Anything-V2-with-OpenGLBokeh-ONNX)，[模拟DSLR的光斑模糊渲染](https://blog.csdn.net/sCs12321/article/details/143893389)，模仿传统摄影大光圈浅景深的效果。
      - 深度估计模型: 采用SOTA模型
      - 对焦距离帧间稳定性:基于人脸检测区域进行卡尔曼滤波，实现帧间稳定性。
      - 深度图帧间稳定性:基于帧卡尔曼滤波，实现模拟DSLR的光斑时的模糊稳定。
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