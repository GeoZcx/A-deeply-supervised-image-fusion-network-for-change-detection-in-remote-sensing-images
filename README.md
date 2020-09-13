# A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images
# 深度监督影像融合网络DSIFN用于高分辨率双时相遥感影像变化检测

Official implement of the Paper：A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images. If you find this work helps in your research, please consider citing:

论文《A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sening images》的官方模型代码。如果该代码对你的研究有所帮助，烦请引用：

> [Zhang, C., Yue, P., Tapete, D., Jiang, L., Shangguan, B., Huang, L., & Liu, G. (2020). A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images. ISPRS Journal of Photogrammetry and Remote Sensing, 166, 183-200.](https://www.sciencedirect.com/science/article/abs/pii/S0924271620301532)


## Introduction
This repository includes DSIFN implementations in PyTorch and Keras version and datasets in the paper

该库包含了DSIFN网络的pytorch和keras版本的代码实现以及论文中使用的数据

## Model Structure
The overview of Deeply supervised image fusion network (DSIFN). The network has two sub-networks: DFEN with pre-trained VGG16 as the backbone for deep feature extraction and DDN with deep feature fusion modules and deep supervision branches for change map reconstruction.

深度监督影像融合网络框架。该网络包含两个子网络：DFEN（深度特征提取网络）以VGG16为网络基底实现深度特征提取；DDN（差异判别网络）由深度特征融合模块和深度监督分支搭建实现影像变化图重建。

![1](imgs/1.png)

## Pytorch version requirements
- Python3.7
- PyTorch 1.6.0
- torchversion 0.7.0 

## Keras version requirements
- Python 3.6
- Tensorflow-gpu 1.13.1
- Keras 2.2.4

## Reference
> [Zhang, C., Yue, P., Tapete, D., Jiang, L., Shangguan, B., Huang, L., & Liu, G. (2020). A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images. ISPRS Journal of Photogrammetry and Remote Sensing, 166, 183-200.](https://www.sciencedirect.com/science/article/abs/pii/S0924271620301532)


## License
Code and datasets are released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.
