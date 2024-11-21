---
layout:     post
title:      "安卓应用逆向分析"
subtitle:   "逆向OpenGL的加密shader/glsl文件"
date:       2024-11-15 16:11:00
author:     "CS"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - smali 
    - Android
    - 逆向
    - 反编译
    - apk
    - glsl
---

我的主页：[https://www.csblog.site/about/](https://www.csblog.site/about/)

## 1. 引言

在一些安卓相机app中大量使用了OpenGL来实现各种特效，有着比较好的效果。而作为菜鸟的我们，如何来‘借鉴’一下它们的实现原理？
当我们直接解包apk文件时，我们欣喜的发现assert中存在着很多以`.glsl`或`.shader`结尾的着色器文件。我们打开一些时发现可以直接读取，但都是一些较为简单的着色器。
当我们想要读取更复杂的着色器时，发现这些`.glsl`或`.shader`文件是加密的，无法直接读取。

### 1.1 目的

**那我们如何来查看这些加密的着色器文件呢？**

我当时有着两个思路：
**1. 分析加密算法，来解密：**

- 这就比较难了，他们选择的加密算法也不是我这种小卡拉米能够破解的。

**2. 反编译：**

- 想到OpenGL着色器的编译一般都是代码运行时来进行的，那么就可以在运行时进行利用它自己的解密算法，直接读取打印！

---

## 2. 软件准备

软件安装与介绍参考别人写的[CSDN博客](https://blog.csdn.net/whbk101/article/details/102551257)

我这次用到的软件：网盘下载:放阿里云盘了，搜*apkUndecryptTool.zip*

**安卓软件：**

- **Apktool M** (在手机端运行，反编译apk文件，在电脑版apktool无法正确反编译时，尝试使用这个)

**电脑软件：**

- **jadx-gui** (方便查看反编译后的代码，可以方便的查看反混淆的Java与smali代码。但是无法直接编辑代码，所以我使用VSCode编辑代码)
- **apktool** (进行反编译、打包、签名、字节对齐等操作)
- **Java2Smali** (对比查看分析smali语法，并生成smali代码来修改反编译后的工程代码)
- **Android Studio** (使用logcat查看打印结果)

---

## 3. 反编译过程

### 3.1 思路

- 1.直接使用 Apktool M 进行反编译，然后再打包，发现软件同意协议后立刻闪退。
- 2.通过jadx-gui进行反编译，发现在代码中有检测机制，重新打包后的apk文件无法正常运行。**解决思路**：
  - 修改退出的代码，让其不主动退出；
  - 查找apk低版本软件，发现低版本的大小是现在版本的一半，里面就没有检测机制，可以正常反编译、打包运行。
- 3.使用电脑版 apktool 将这个低版本的apk文件进行反编译，发现生成的文件内容不是正常的apk解包内容，缺少很多文件。**解决办法**：
  - 重新使用 Apktool M 进行反编译，发现生成的文件内容是正常的，所以将 Apktool M 的反编译再打包的 apk 复制到电脑版 apktool 进行反编译。
- 4.依次执行签名、字节对齐、安装，发现提示V2签名失败，所以需要使用v1签名。**去掉V2签名方法**：

``` text
1.反编译 apk 后，删除 original 里面的 META-INF 文件夹，或者也需要删除根目录 META-INF 文件夹的即可；
2.在 apktool.yml 中修改 targetSdkVersion：
   sdkInfo:
    minSdkVersion: '21'
    targetSdkVersion: '29'
3.在 AndroidManifest.xml 中修改字段：
    android:compileSdkVersion="29"
    android:compileSdkVersionCodename="15"
```

- 5.签名、签名验证、字节对齐的指令，在run.bat脚本：

``` bash
@echo off
rem 【先删除已经生成的文件】
set code_dir=result\ReLens
set sourceFolder=%code_dir%\dist
set targetFolder=%code_dir%\build
set signed_apk=result\signed.apk
set aligned4_apk=result\4_aligned.apk

if exist %sourceFolder% (
    rmdir /s /q %sourceFolder%
) else (
    echo %sourceFolder% NOT EXIT.
)
if exist %targetFolder% (
    rmdir /s /q %targetFolder%
) else (
    echo %targetFolder% NOT EXIT.
)

timeout /t 1 >nul

if exist %signed_apk% (
    del %signed_apk%
) else (
    echo %signed_apk% NOT EXIT.
)
if exist %aligned4_apk% (
    del %aligned4_apk%
) else (
    echo %aligned4_apk% NOT EXIT.
)

timeout /t 1 >nul

rem 【1. 使用apktool打包ReLens】
echo --------------------- 1.run apktool pack %code_dir%... ---------------------
apktool.bat b %code_dir% && (
    rem 【2. 使用jarsigner对apk进行签名】
    echo --------------------- 2.run jarsigner apk sign... ---------------------
    jarsigner  -verbose  -keystore  cs.keystore  -storepass 123457800 -signedjar  %signed_apk%  %code_dir%\dist\*.apk cs && (
        rem 【3. 使用apksigner验证签名】
        echo --------------------- 3.run apksigner verify... ---------------------
        E:\AndroidC\Sdk\build-tools\33.0.0\apksigner.bat verify -v %signed_apk% && (
            rem 【4. 使用zipalign对apk进行优化】
            echo --------------------- 4.run zipalign apk zipalign... ---------------------
            E:\AndroidC\Sdk\build-tools\30.0.0\zipalign.exe 4 %signed_apk% %aligned4_apk%
            color 0A
            echo --------------------- done. ---------------------
        )
    )
)
```

- 6.然后尝试修改代码（只修改文本内容，不新增代码、不修改类型），然后再重新打包，验证没问题。

### 3.1 修改代码

- 1.使用 jadx-gui 软件搜索关键字 `xxx.glsl` 查找到加载着色器的smali代码。可以在软件中将smali转Java阅读。代码如下：

``` java
// 发现所有着色器的加载代码，均通过EncryptShaderUtil方法的解密代码加载成String。代码如下：
public C5638h() {
  super(EncryptShaderUtil.instance.getShaderStringFromAsset("focusshader/bokenh_postprocess_fs.glsl"));
  this.f21869z = 12.0f;
  this.f21852A = 0.5f;
}

// 然后跳转到 getShaderStringFromAsset() 函数:
public String getShaderStringFromAsset(String str) {
  return privateGetShaderStringFromAsset(str, this.isEncrypt);
}

// 再跳转到 privateGetShaderStringFromAsset() 函数:
private String privateGetShaderStringFromAsset(String str, boolean z) {
  try {
    byte[] privateGetBinFromAsset = privateGetBinFromAsset(str, z);
    if (privateGetBinFromAsset != null && privateGetBinFromAsset.length != 0) {
        return new String(privateGetBinFromAsset, "utf-8");
    }
    return "";
  } catch (Exception e2) {
    e2.printStackTrace();
    return "";
  }
}

// 最终，找到了着色器内容String的结果为：return new String(privateGetBinFromAsset, "utf-8");
```

- 2.我们的目标是添加打印代码，就要尝试修改smali代码：
  - 修改smali代码，有一些寄存器等需要注意，可以参考博客：[Android逆向工程：实战！讲解修改插入Smali代码的规则，带你快速进行二次改造！](https://blog.csdn.net/qq_34149335/article/details/82699029)
  - smali与java对应代码：

``` java
// JAVA代码
private String privateGetShaderStringFromAsset(String str, boolean z) {
  try {
    byte[] privateGetBinFromAsset = privateGetBinFromAsset(str, z);
    if (privateGetBinFromAsset != null && privateGetBinFromAsset.length != 0) {
      return new String(privateGetBinFromAsset, "utf-8");
    }
    return "";
  } catch (Exception e2) {
    e2.printStackTrace();
    return "";
  }
}

// smali代码
.method private privateGetShaderStringFromAsset(Ljava/lang/String;Z)Ljava/lang/String;
  .locals 2            // 2个寄存器（新增时，需要在这里增加）

  const-string v0, ""  // 寄存器0

  :try_start_0
  invoke-direct {p0, p1, p2}, Lcom/lightcone/utils/EncryptShaderUtil;->privateGetBinFromAsset(Ljava/lang/String;Z)[B

  move-result-object p1

  if-eqz p1, :cond_1

  array-length p2, p1

  if-nez p2, :cond_0

  goto :goto_0

  :cond_0
  new-instance p2, Ljava/lang/String;

  const-string v1, "utf-8"

  invoke-direct {p2, p1, v1}, Ljava/lang/String;-><init>([BLjava/lang/String;)V
  :try_end_0
  .catch Ljava/lang/Exception; {:try_start_0 .. :try_end_0} :catch_0

  return-object p2     // 返回着色器String结果，需要打印p2

  :cond_1
  :goto_0
  return-object v0

  :catch_0
  move-exception p1

  invoke-virtual {p1}, Ljava/lang/Exception;->printStackTrace()V

  return-object v0
.end method
```

- 3.然后在smali代码中添加打印代码：
  - 使用 Java2Smali 查看`System.out.println(str);`对应的smali代码：
    - sget-object v0, Ljava/lang/System;->out:Ljava/io/PrintStream;                   // 打印流寄存器v0
    - invoke-virtual {v0, p1}, Ljava/io/PrintStream;->println(Ljava/lang/String;)V    // 将p1的String打印至v0
  - smali修改后的代码：

``` java
.method private privateGetShaderStringFromAsset(Ljava/lang/String;Z)Ljava/lang/String;
  .locals 4                   // 4个寄存器（新增2个：v2, v3）

  const-string v0, ""

  const-string v2, "[CS]WTF"  // 新增字符串寄存器v2

  sget-object v3, Ljava/lang/System;->out:Ljava/io/PrintStream;  // 新增打印流寄存器v3

  :try_start_0
  invoke-virtual {v3, p1}, Ljava/io/PrintStream;->println(Ljava/lang/String;)V  // 打印输入的文件名str到流寄存器v3
  invoke-direct {p0, p1, p2}, Lcom/lightcone/utils/EncryptShaderUtil;->privateGetBinFromAsset(Ljava/lang/String;Z)[B

  move-result-object p1

  if-eqz p1, :cond_1

  array-length p2, p1

  if-nez p2, :cond_0

  goto :goto_0

  :cond_0
  new-instance p2, Ljava/lang/String;

  const-string v1, "utf-8"

  invoke-direct {p2, p1, v1}, Ljava/lang/String;-><init>([BLjava/lang/String;)V
  :try_end_0
  .catch Ljava/lang/Exception; {:try_start_0 .. :try_end_0} :catch_0

  invoke-virtual {v3, v2}, Ljava/io/PrintStream;->println(Ljava/lang/String;)V

  invoke-virtual {v3, p2}, Ljava/io/PrintStream;->println(Ljava/lang/String;)V

  return-object p2

  :cond_1
  :goto_0
  return-object v0

  :catch_0
  move-exception p1

  invoke-virtual {p1}, Ljava/lang/Exception;->printStackTrace()V

  return-object v0
.end method
```

- 4.安装apk至手机，数据线连接至电脑，在 Android Studio 的 logcat 查看打印：
  - 搜索关键字 `[CS]WTF` 即可看到对应的着色器代码与着色器文件路径。

大功告成，耗时两天，实现了查看加密着色器的功能。
