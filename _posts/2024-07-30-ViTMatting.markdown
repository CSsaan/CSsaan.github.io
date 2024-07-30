---
layout:     post
title:      "Hello ViTMatting"
subtitle:   " \"Hello World, Hello Blog\""
date:       2024-07-30 18:00:00
author:     "CS"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Meta
---

Title: GitHub - CSsaan/EMA-ViTMatting

URL Source: https://github.com/CSsaan/EMA-ViTMatting

Markdown Content:
EMA-ViTMatting
--------------

[](https://github.com/CSsaan/EMA-ViTMatting#ema-vitmatting)

[\[Project Page\]](https://github.com/CSsaan/EMA-ViTMatting/) [\[中文主页\]](https://github.com/CSsaan/EMA-ViTMatting/blob/main/README_CN.md)

Using EMA to train Matting task.

Single RGB image input, single alpha image output.

This project focuses on the field of image alpha matting. Currently, there are few open-source end-to-end alpha matting models available, most of which are based on convolutional neural network models with large parameter sizes. Therefore, this paper adopts a mobile ViT combined with an improved cascaded decoder module to create a lightweight alpha matting model with reduced computational complexity. The innovation lies in the combination of a lightweight ViT model and an improved decoder module, bringing a more efficient solution to the alpha matting field.

👀 Demo
-------

[](https://github.com/CSsaan/EMA-ViTMatting#-demo)

Demo: [Bilibili Video](https://www.bilibili.com/)

| **Original Image** | **Label** | **Our Results** | **Our Result** | \--- | **Original Image** | **Label** | **Our Result** | **Our Result** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [![Image 1](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/p_f7b2317f.jpg)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/p_f7b2317f.jpg) | [![Image 2](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/lab_p_f7b2317f.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/lab_p_f7b2317f.png) | [![Image 3](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/pre_p_f7b2317f.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/pre_p_f7b2317f.png) | [![Image 4](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/green_p_f7b2317f.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/green_p_f7b2317f.png) | \--- | [![Image 5](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/p_f89c7881.jpg)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/p_f89c7881.jpg) | [![Image 6](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/lab_p_f89c7881.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/lab_p_f89c7881.png) | [![Image 7](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/pre_p_f89c7881.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/pre_p_f89c7881.png) | [![Image 8](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/green_p_f89c7881.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/green_p_f89c7881.png) |
| [![Image 9](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/p_f30f22fd.jpg)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/p_f30f22fd.jpg) | [![Image 10](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/lab_p_f30f22fd.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/lab_p_f30f22fd.png) | [![Image 11](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/pre_p_f30f22fd.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/pre_p_f30f22fd.png) | [![Image 12](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/green_p_f30f22fd.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/green_p_f30f22fd.png) | \--- | [![Image 13](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/p_fcb9a19e.jpg)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/p_fcb9a19e.jpg) | [![Image 14](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/lab_p_fcb9a19e.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/lab_p_fcb9a19e.png) | [![Image 15](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/pre_p_fcb9a19e.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/pre_p_fcb9a19e.png) | [![Image 16](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/green_p_fcb9a19e.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/green_p_fcb9a19e.png) |
| [![Image 17](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/p_f053bec5.jpg)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/p_f053bec5.jpg) | [![Image 18](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/lab_p_f053bec5.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/lab_p_f053bec5.png) | [![Image 19](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/pre_p_f053bec5.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/pre_p_f053bec5.png) | [![Image 20](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/green_p_f053bec5.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/green_p_f053bec5.png) | \--- | [![Image 21](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/p_fe6a4bfe.jpg)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/p_fe6a4bfe.jpg) | [![Image 22](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/lab_p_fe6a4bfe.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/lab_p_fe6a4bfe.png) | [![Image 23](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/pre_p_fe6a4bfe.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/pre_p_fe6a4bfe.png) | [![Image 24](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/green_p_fe6a4bfe.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/green_p_fe6a4bfe.png) |
| [![Image 25](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/p_f879fac6.jpg)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/p_f879fac6.jpg) | [![Image 26](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/lab_p_f879fac6.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/lab_p_f879fac6.png) | [![Image 27](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/pre_p_f879fac6.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/pre_p_f879fac6.png) | [![Image 28](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/green_p_f879fac6.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/green_p_f879fac6.png) | \--- | [![Image 29](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/p_fdaa48dd.jpg)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/p_fdaa48dd.jpg) | [![Image 30](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/lab_p_fdaa48dd.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/lab_p_fdaa48dd.png) | [![Image 31](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/pre_p_fdaa48dd.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/pre_p_fdaa48dd.png) | [![Image 32](https://github.com/CSsaan/EMA-ViTMatting/raw/main/result/green_p_fdaa48dd.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/result/green_p_fdaa48dd.png) |

Model structure: [![Image 33](https://github.com/CSsaan/EMA-ViTMatting/raw/main/.png)](https://github.com/CSsaan/EMA-ViTMatting/blob/main/.png)

📦 Prerequisites
----------------

[](https://github.com/CSsaan/EMA-ViTMatting#-prerequisites)

#### Requirements:

[](https://github.com/CSsaan/EMA-ViTMatting#requirements)

*   Python >= 3.8
*   torch >= 2.2.2
*   CUDA Version >= 11.7

🔧 Install
----------

[](https://github.com/CSsaan/EMA-ViTMatting#-install)

#### Configure Environment:

[](https://github.com/CSsaan/EMA-ViTMatting#configure-environment)

git clone git@github.com:CSsaan/EMA-ViTMatting.git
cd EMA-ViTMatting
conda create -n ViTMatting python=3.10 -y
conda activate ViTMatting
pip install -r requirements.txt

🚀 Quick Start
--------------

[](https://github.com/CSsaan/EMA-ViTMatting#-quick-start)

#### train script:

[](https://github.com/CSsaan/EMA-ViTMatting#train-script)

```
Dataset directory structure：
data
└── AIM500
    ├── train
    │   ├── original
    │   └── mask
    └── test
        ├── original
        └── mask
```

python train.py --use\_model\_name 'VisionTransformer' --reload\_model False --local\_rank 0 --world\_size 4 --batch\_size 16 --data\_path '/data/AIM500' --use\_distribute False

*   `--use_model_name 'VisionTransformer'`: The name of the model to load
*   `--reload_model False`: Model checkpoint continuation training
*   `--local_rank 0`: The local rank of the current process
*   `--world_size 4`: The total number of processes
*   `--batch_size 16`: Batch size
*   `--data_path '/data/AIM500'`: Data path
*   `--use_distribute False`: Whether to use distributed training

#### test script:

[](https://github.com/CSsaan/EMA-ViTMatting#test-script)

python inferenceCS.py --image\_path data/AIM500/test/original/o\_dc288b1a.jpg --model\_name MobileViT\_194\_pure

📖 Paper
--------

[](https://github.com/CSsaan/EMA-ViTMatting#-paper)

None

🎯 Todo
-------

[](https://github.com/CSsaan/EMA-ViTMatting#-todo)

*   Data preprocessing -> dataset\\AIM\_500\_datasets.py
*   Data augmentation -> dataset\\AIM\_500\_datasets.py
*   Model loading -> config.py & Trainer.py
*   Loss functions -> benchmark\\loss.py
*   Dynamic learning rate -> train.py
*   Distributed training -> train.py
*   Model visualization -> model\\mobile\_vit.py
*   Model parameters -> benchmark\\config\\model\_MobileViT\_parameters.yaml
*   Training -> train.py
*   Model saving -> Trainer.py
*   Test visualization ->
*   Model inference -> inferenceCS.py
*   Pytorch model to onnx -> onnx\_demo
*   Model acceleration ->
*   Model optimization ->
*   Model tuning ->
*   Model integration ->
*   Model quantization, compression, deployment ->

📂 Repo structure (WIP)
-----------------------

[](https://github.com/CSsaan/EMA-ViTMatting#-repo-structure-wip)

```
├── README.md
├── benchmark 
│   ├── loss.py              -> loss functions
│   └── config               -> all model's parameters
├── utils
│   ├── testGPU.py
│   ├── yuv_frame_io.py
│   └── print_structure.py
├── onnx_demo
│   ├── export_onnx.py
│   └── infer_onnx.py
├── data                     -> dataset
├── dataset                  -> dataloder
├── log                      -> tensorboard log
├── model
├── Trainer.py               -> load model & train.
├── config.py                -> all models dictionary
├── dataset.py               -> dataLoader
├── demo_Fc.py               -> model inder
├── pyproject.toml           -> project config
├── requirements.txt
├── train.py                 -> main
└── inferenceCS.py           -> model inference
```