
# Ultra-Light-Fast-Generic-Face-Detector-1MB 
# 轻量级人脸检测模型
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/27.jpg)
该模型是针对边缘计算设备设计的轻量人脸检测模型。


## 环境依赖
python >= 3.6  
paddlepaddle >= 2.0.1 或 paddlepaddle-gpy >= 2.0.1
x2paddle >= 1.2.0
*** paddlepaddle安装: ***
```shell
pip install paddlepaddle-gpu==2.0.1
```

*** x2paddle安装: ***
```shell
pip install x2paddle==1.2.0
```


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

## 检测图片效果（输入分辨率：640x480）
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/26.jpg)
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/2.jpg)
![img1](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/blob/master/readme_imgs/4.jpg)
## PS

 - 若生产实际场景为中近距离、人脸大、人脸数少，则建议采用输入尺寸input_size：320（320x240）分辨率训练，并采用 320x240/160x120/128x96 图片大小输入进行预测推理，如使用提供的预训练模型 **version-slim-320.pth** 或者 **version-RFB-320.pth** 进行推理。
 - 若生产实际场景为中远距离、人脸中小、人脸数多，则建议采用：
 
 （1）最优：输入尺寸input_size：640（640x480）分辨率训练，并采用同等或更大输入尺寸进行预测推理,如使用提供的预训练模型 **version-slim-640.pth** 或者 **version-RFB-640.pth** 进行推理，更低的误报。
 
 （2）次优：输入尺寸input_size：320（320x240）分辨率训练，并采用480x360或640x480大小输入进行预测推理，对于小人脸更敏感，误报会增加。
 
 - 各个场景的最佳效果需要调整输入分辨率从而在速度和精度中间取得平衡。
 - 过大的输入分辨率虽然会增强小人脸的召回率，但是也会提高大、近距离人脸的误报率，而且推理速度延迟成倍增加。
 - 过小的输入分辨率虽然会明显加快推理速度，但是会大幅降低小人脸的召回率。
 - 生产场景的输入分辨率尽量与模型训练时的输入分辨率保持一致，上下浮动不宜过大。
 

