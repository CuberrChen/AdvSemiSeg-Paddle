# AdvSemiSeg-Paddle
Reproduction of Paper "Adversarial Learning for Semi-Supervised Semantic Segmentation" with PaddlePaddle.

# 项目说明

**Reproduction of Paper "Adversarial Learning for Semi-Supervised Semantic Segmentation" with PaddlePaddle.**

默认的配置是**ResNet101+Deeplabv2+VOC2012+1/8Label**

其余训练设置与原文项目一致


**复现指标：70.4 miou**

原文指标：69.5 miou


1、模型的设置在相应的代码中：
- train.py中204行处;
- evaluate.py中197行处;

2、标签率设置：--labeled-ratio 0.125 表示1/8标签率

3、训练指令和评估指令如下方示例

4、评估结果存放在results中的txt文件中。

AiStudio中已经提供了训练好的模型和可视化日志（checkpoints中）和评估结果（results中）。
可直接在AiStudio中打开可视化工具查看训练曲线。
train_advSemiSeg_voc_res101.log为非可视化的训练日志。

# AiStudio

**在AiStudio中一键可运行**

项目“advSeg”共享链接：https://aistudio.baidu.com/aistudio/projectdetail/2884884?contributionType=1&shared=1

预训练模型和可视化日志的网盘链接：链接: https://pan.baidu.com/s/1zPllmHyIgZzvQfTKLZYtdQ 提取码: pgqv 复制这段内容后打开百度网盘手机App，操作更方便

## 数据集准备

```python
#解压数据集 AiStudio中
!unzip -q data/data4379/pascalvoc.zip -d data/data4379/
!unzip -q data/data117898/SegmentationClassAug.zip -d data/data4379/pascalvoc/VOCdevkit/VOC2012/
```

应有的数据集结构：
```
pascalvoc/VOCdevkit/VOC2012
├── Annotations
├── ImageSets
├── JPEGImages
├── __MACOSX
├── SegmentationClass
├── SegmentationClassAug # 原文增加 参考原文项目说明
└── SegmentationObject
```
## 训练advSemiSeg

注意：根据环境不同，需在代码中更改数据集路径。AIStudio中可直接运行。
```python
!python train.py --checkpoint_dir ./checkpoints/voc_semi_0_125 --labeled-ratio 0.125 --ignore-label 255 --num-classes 21 --use_vdl
```


## 评估advSemiSeg


```python
!python evaluate.py --dataset pascal_voc --num-classes 21 --restore-from ./checkpoints/voc_semi_0_125/20000.pdparams 
```

    2021-11-24 10:15:59 [INFO]	
    ------------Environment Information-------------
    platform: Linux-4.13.0-36-generic-x86_64-with-debian-stretch-sid
    Python: 3.7.4 (default, Aug 13 2019, 20:35:49) [GCC 7.3.0]
    Paddle compiled with cuda: True
    NVCC: Cuda compilation tools, release 10.1, V10.1.243
    cudnn: 7.6
    GPUs used: 1
    CUDA_VISIBLE_DEVICES: None
    GPU: ['GPU 0: Tesla V100-SXM2-16GB']
    GCC: gcc (Ubuntu 7.5.0-3ubuntu1~16.04) 7.5.0
    PaddlePaddle: 2.2.0
    OpenCV: 4.1.1
    ------------------------------------------------
    Namespace(data_dir='/home/aistudio/data/data4379/pascalvoc/VOCdevkit/VOC2012', data_list='./data/voc_list/val.txt', dataset='pascal_voc', gpu=0, ignore_label=255, model='Deeplabv2', num_classes=21, restore_from='./checkpoints/voc_semi_0_125/20000.pdparams', save_dir='results', save_output_images=False)
    W1124 10:15:59.974874  2351 device_context.cc:447] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.0, Runtime API Version: 10.1
    W1124 10:15:59.974916  2351 device_context.cc:465] device: 0, cuDNN Version: 7.6.
    2021-11-24 10:16:03 [INFO]	No pretrained model to load, ResNet_vd will be trained from scratch.
    2021-11-24 10:16:03 [INFO]	Loading pretrained model from ./checkpoints/voc_semi_0_125/20000.pdparams
    2021-11-24 10:16:03 [INFO]	There are 538/538 variables loaded into DeepLabV2.
    0 processd
    evaluate.py:238: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      gt = np.asarray(label[0].numpy()[:size[0],:size[1]], dtype=np.int)
    evaluate.py:245: DeprecationWarning: `np.int` is a deprecated alias for the builtin `int`. To silence this warning, use `int` by itself. Doing this will not modify any behavior and is safe. When replacing `np.int`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
    100 processd
    200 processd
    300 processd
    400 processd
    500 processd
    600 processd
    700 processd
    800 processd
    900 processd
    1000 processd
    1100 processd
    1200 processd
    1300 processd
    1400 processd
    class  0 background   IU 0.93
    class  1 aeroplane    IU 0.85
    class  2 bicycle      IU 0.41
    class  3 bird         IU 0.85
    class  4 boat         IU 0.67
    class  5 bottle       IU 0.78
    class  6 bus          IU 0.90
    class  7 car          IU 0.84
    class  8 cat          IU 0.84
    class  9 chair        IU 0.32
    class 10 cow          IU 0.72
    class 11 diningtable  IU 0.38
    class 12 dog          IU 0.81
    class 13 horse        IU 0.73
    class 14 motorbike    IU 0.81
    class 15 person       IU 0.83
    class 16 pottedplant  IU 0.44
    class 17 sheep        IU 0.78
    class 18 sofa         IU 0.43
    class 19 train        IU 0.72
    class 20 tvmonitor    IU 0.71
    meanIOU: 0.7040637094579905
    


## 训练 fully-supervised Baseline (FSL)


```python
!python train_full_pd.py --dataset pascal_voc  \
                        --checkpoint-dir ./checkpoints/voc_full \
                        --ignore-label 255 \
                        --num-classes 21 
```
