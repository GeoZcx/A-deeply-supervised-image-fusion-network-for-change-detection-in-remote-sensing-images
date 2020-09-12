# A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images
# 深度监督影像融合网络DSIFN用于高分辨率双时相遥感影像变化检测
Official implement of Paper：A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sening images

论文《A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sening images》的官方模型代码

## Introduction
The codes are pytorch and keras implements for paper: A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sening images

包含了pytorch和keras版本的模型代码，原文链接：

> [Zhang, C., Yue, P., Tapete, D., Jiang, L., Shangguan, B., Huang, L., & Liu, G. (2020). A deeply supervised image fusion network for change detection in high resolution bi-temporal remote sensing images. ISPRS Journal of Photogrammetry and Remote Sensing, 166, 183-200.](https://www.sciencedirect.com/science/article/abs/pii/S0924271620301532)

## Structure
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
