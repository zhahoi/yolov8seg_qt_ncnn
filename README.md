# yolov8seg_qt_ncnn
使用ncnn框架部署yolov8-seg，外加qt进行界面可视化操作，用于图片和视频实例分割推理。



## 起

一直以来自己做算法部署或者是模型推理，最后总是需要在命令行敲各种命令，然后通常借助opencv进行可视化。老实说，虽然最后的结果也展现出来了，但是总感觉操作比较繁琐并且敲指令对于那些非专业的人来说，门槛也相对高了，不如在UI界面操作直观、简单。因此，便一直有学习QT的打算，一方面是可以再点亮一个技能分支，另一方面也能让部署算法在实行图片或者视频推理的结果更加的直观。再加上QT框架使用C++编写的，而现在的自己对C++有了部分掌握，所以学习成本相对也没那么高。

在学习完基本的QT之后，发现能够简单看懂别人工程的实现，但是暂时没办法按照自己的思路绘制UI以及实现按钮功能，因此便只能站在巨人的肩膀上做二次修改（低情商：抄代码）。



## 承 

因为我之前部署的模型大部分都是基于ncnn框架实现的，同时一般都需要载入模型、加载图像或者视频等等，因此我希望可以找到一个QT工程让我可以不用怎么修改UI界面和槽函数定义与实现，同时可以将我已有的ncnn模型移植进去那就最完美了，因为可以减少我的很多工作量。

在Gayhub找了一番之后，还真的让我找到了。这个QT仓库的地址为：[QT_Learning](https://github.com/KeepTryingTo/QT_Learning)，该仓库实现了很多常用的QT案例，其中就有我所需要的有关qt+ncnn部署方面的案例，如果我将该案例搞懂并在此技术上进行模型修改的话，应该可以节省我的很多学习成本。

在花了一点时间熟悉了该工程之后，想着要实战试试。因此我选择了之前实现的基于ncnn框架实现的yolov8-seg实例分割：[ncnn_yolov8_seg](https://github.com/zhahoi/ncnn_yolov8_seg)，直接用现成的代码更是可以省略我训练模型和后处理代码的时间。经过几天的学习和调试，最后在chatgpt和我自己的努力下终于实现了基本的功能，本仓库包含完成的程序代码。

整体的UI界面如下：

![4ca4bd137d0da3d565b114be0948ac86.jpeg](https://ice.frostsky.com/2024/11/14/4ca4bd137d0da3d565b114be0948ac86.jpeg)

如果要进行图片推理，先载入图片或者模型权重都可以，可以设置推理图片的分辨率、类别阈值和NMS阈值，再进行检测。图片推理结果演示如下：

![96846e78ea1923ceb4e5a2581bfeccc2.jpeg](https://ice.frostsky.com/2024/11/14/96846e78ea1923ceb4e5a2581bfeccc2.jpeg)

如果要进行视频推理，需要先载入权重，设置好推理图片的分辨率、类别阈值和NMS阈值，再进行视频推理。视频推理演示如下：

![0851963c2277b63befaa87f8323a3759.jpeg](https://ice.frostsky.com/2024/11/14/0851963c2277b63befaa87f8323a3759.jpeg)



## 转

本仓库的代码我分别有在windows和ubuntu上都运行过，都能正常实现检测功能。

以下介绍实现该工程所需的一些依赖：

1. windows端
   - Visual Studio 2019(安装qt插件)
   - opencv-3.4.10
   - ncnn-20240820-full-source
   - protobuf-3.4.0
2. ubuntu端
   - Qt Creator
   - opencv-3.4.10
   - ncnn-20240820-full-source

在Windows端之所以不用Qt Creator的原因是因为不知道为什么在运行代码时会报"platform.h"的错误，在网山搜索了一大圈没找到解决方法，最后只能在Visual Studio进行。本仓库的代码是在Ubuntu上运行的关键代码，相比Windows上所含的文件少很多，同时很直观。

以下是一些在执行代码中需要注意的点：

（1）在UI界面上，载入模型时默认是载入**".param"**格式的权重文件，选择".bin"会报错，弹窗让你重新选择。

（2）在选择输入分辨率候选框，无论选择哪种分辨率，结果都可以正常推理，理论上分辨率更高，推理的结果更准确。



一些安装库可以参考的博客文章：

- [（一篇即成功）Ubuntu下搭建qt环境和安装Qt Creator](https://blog.csdn.net/m0_73450461/article/details/143316194)
- [Ubuntu16.04安装NCNN和Opencv](https://blog.csdn.net/weixin_44855366/article/details/130165967)
- [Visual Studio 安装 Qt 插件](https://blog.csdn.net/m0_58648890/article/details/143214248)



## 合

本仓库QT界面的设计合代码实现，理论上可以进行二次开发，适配各种框架模型或者算法的部署，只需要修改里面的算法代码即可。通过本次学习，让我掌握了Qt的基本用法，希望以后可以对此掌握更加熟练，以便于实现更多的功能。



### reference

-[QT_Learning](https://github.com/KeepTryingTo/QT_Learning)

-[ncnn_yolov8_seg](https://github.com/zhahoi/ncnn_yolov8_seg)
