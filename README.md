
# Ultra-Light-Fast-Generic-Face-Detector-1MB 
# 轻量级人脸检测模型
该代码由论文的[PyTorch官方实现](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)通过[X2Paddle](https://github.com/PaddlePaddle/X2Paddle)转换而来，转换过程参加[教程](https://github.com/SunAhong1993/X2Paddle/blob/code_convert_last/docs/pytorch_project_convertor/demo.md#ultra-light-fast-generic-face-detector)。  

## 简介
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/27.jpg)
该模型是针对边缘计算设备设计的轻量人脸检测模型。


## 环境依赖
* [python 3.5+](https://www.continuum.io/downloads)
* [paddlepaddle 或 paddlepaddle-gpu 2.0.1+](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip)
* [x2Paddle 1.2+](https://github.com/PaddlePaddle/X2Paddle)


## 生成VOC格式训练数据集以及训练流程

1. 下载widerface官网数据集或者下载我提供的训练集解压放入./data文件夹内：

  （1）过滤掉10px*10px 小人脸后的干净widerface数据压缩包 ：[百度云盘 (提取码：cbiu)](https://pan.baidu.com/s/1MR0ZOKHUP_ArILjbAn03sw ) 、[Google Drive](https://drive.google.com/open?id=1OBY-Pk5hkcVBX1dRBOeLI4e4OCvqJRnH )
  
  （2）未过滤小人脸的完整widerface数据压缩包 ：[百度云盘 (提取码：ievk)](https://pan.baidu.com/s/1faHNz9ZrtEmr_yw48GW7ZA ) 、[Google Drive](https://drive.google.com/open?id=1sbBrDRgctEkymIpCh1OZBrU5qBS-SnCP )
  
2. **（PS:如果下载的是过滤后的上述(1)中的数据包，则不需要执行这步）** 由于widerface存在很多极小的不清楚的人脸，不太利于高效模型的收敛，所以需要过滤训练，默认过滤人脸大小10像素x10像素以下的人脸。
运行./data/wider_face_2_voc_add_landmark.py
```Python
 python3 ./data/wider_face_2_voc_add_landmark.py
```
程序运行和完毕后会在./data目录下生成 **wider_face_add_lm_10_10**文件夹，该文件夹数据和数据包（1）解压后相同，完整目录结构如下：
```Shell
  data/
    retinaface_labels/
      test/
      train/
      val/
    wider_face/
      WIDER_test/
      WIDER_train/
      WIDER_val/
    wider_face_add_lm_10_10/
      Annotations/
      ImageSets/
      JPEGImages/
    wider_face_2_voc_add_landmark.py
```

3. 至此VOC训练集准备完毕，项目根目录下分别有 **train-version-slim.sh** 和 **train-version-RFB.sh** 两个脚本，前者用于训练**slim版本**模型，后者用于训练**RFB版本**模型，默认参数已设置好，参数如需微调请参考 **./train.py** 中关于各训练超参数的说明。

4. 运行**train-version-slim.sh** 或 **train-version-RFB.sh**即可
```Shell
sh train-version-slim.sh 或者 sh train-version-RFB.sh
```

