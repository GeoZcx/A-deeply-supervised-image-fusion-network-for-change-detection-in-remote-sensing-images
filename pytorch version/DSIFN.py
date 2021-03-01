
# credits: https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images

import torch
import torch.nn as nn
from torchvision.models import vgg16
import numpy as np

class vgg16_base(nn.Module):
    def __init__(self):
        super(vgg16_base,self).__init__()
        features = list(vgg16(pretrained=True).features)[:30]
        self.features = nn.ModuleList(features).eval()

    def forward(self,x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3,8,15,22,29}:
                results.append(x)
        return results

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention,self).__init__()
        self.conv1 = nn.Conv2d(2,1,7,padding=3,bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x,dim=1,keepdim=True)
        max_out = torch.max(x,dim=1,keepdim=True,out=None)[0]

        x = torch.cat([avg_out,max_out],dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

def conv2d_bn(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1),
        nn.PReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Dropout(p=0.6),
    )

class DSIFN(nn.Module):
    def __init__(self, model_A, model_B):
        super().__init__()
        self.t1_base = model_A
        self.t2_base = model_B
        self.sa = SpatialAttention()
        self.sigmoid = nn.Sigmoid()

        # branch1
        self.ca1 = ChannelAttention(in_channels=1024)
        self.bn_ca1 = nn.BatchNorm2d(1024)
        self.o1_conv1 = conv2d_bn(1024, 512)
        self.o1_conv2 = conv2d_bn(512, 512)
        self.bn_sa1 = nn.BatchNorm2d(512)
        self.o1_conv3 = nn.Conv2d(512, 1, 1)
        self.trans_conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)

        # branch 2
        self.ca2 = ChannelAttention(in_channels=1536)
        self.bn_ca2 = nn.BatchNorm2d(1536)
        self.o2_conv1 = conv2d_bn(1536, 512)
        self.o2_conv2 = conv2d_bn(512, 256)
        self.o2_conv3 = conv2d_bn(256, 256)
        self.bn_sa2 = nn.BatchNorm2d(256)
        self.o2_conv4 = nn.Conv2d(256, 1, 1)
        self.trans_conv2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        # branch 3
        self.ca3 = ChannelAttention(in_channels=768)
        self.o3_conv1 = conv2d_bn(768, 256)
        self.o3_conv2 = conv2d_bn(256, 128)
        self.o3_conv3 = conv2d_bn(128, 128)
        self.bn_sa3 = nn.BatchNorm2d(128)
        self.o3_conv4 = nn.Conv2d(128, 1, 1)
        self.trans_conv3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)

        # branch 4
        self.ca4 = ChannelAttention(in_channels=384)
        self.o4_conv1 = conv2d_bn(384, 128)
        self.o4_conv2 = conv2d_bn(128, 64)
        self.o4_conv3 = conv2d_bn(64, 64)
        self.bn_sa4 = nn.BatchNorm2d(64)
        self.o4_conv4 = nn.Conv2d(64, 1, 1)
        self.trans_conv4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        # branch 5
        self.ca5 = ChannelAttention(in_channels=192)
        self.o5_conv1 = conv2d_bn(192, 64)
        self.o5_conv2 = conv2d_bn(64, 32)
        self.o5_conv3 = conv2d_bn(32, 16)
        self.bn_sa5 = nn.BatchNorm2d(16)
        self.o5_conv4 = nn.Conv2d(16, 1, 1)

    def forward(self,t1_input,t2_input):
        t1_list = self.t1_base(t1_input)
        t2_list = self.t2_base(t2_input)

        t1_f_l3,t1_f_l8,t1_f_l15,t1_f_l22,t1_f_l29 = t1_list[0],t1_list[1],t1_list[2],t1_list[3],t1_list[4]
        t2_f_l3,t2_f_l8,t2_f_l15,t2_f_l22,t2_f_l29,= t2_list[0],t2_list[1],t2_list[2],t2_list[3],t2_list[4]

        x = torch.cat((t1_f_l29,t2_f_l29),dim=1)
        #optional to use channel attention module in the first combined feature
        #在第一个深度特征叠加层之后可以选择使用或者不使用通道注意力模块
        # x = self.ca1(x) * x
        x = self.o1_conv1(x)
        x = self.o1_conv2(x)
        x = self.sa(x) * x
        x = self.bn_sa1(x)

        branch_1_out = self.sigmoid(self.o1_conv3(x))

        x = self.trans_conv1(x)
        x = torch.cat((x,t1_f_l22,t2_f_l22),dim=1)
        x = self.ca2(x)*x
        #According to the amount of the training data, appropriately reduce the use of conv layers to prevent overfitting
        #根据训练数据的大小，适当减少conv层的使用来防止过拟合
        x = self.o2_conv1(x)
        x = self.o2_conv2(x)
        x = self.o2_conv3(x)
        x = self.sa(x) *x
        x = self.bn_sa2(x)

        branch_2_out = self.sigmoid(self.o2_conv4(x))

        x = self.trans_conv2(x)
        x = torch.cat((x,t1_f_l15,t2_f_l15),dim=1)
        x = self.ca3(x)*x
        x = self.o3_conv1(x)
        x = self.o3_conv2(x)
        x = self.o3_conv3(x)
        x = self.sa(x) *x
        x = self.bn_sa3(x)

        branch_3_out = self.sigmoid(self.o3_conv4(x))

        x = self.trans_conv3(x)
        x = torch.cat((x,t1_f_l8,t2_f_l8),dim=1)
        x = self.ca4(x)*x
        x = self.o4_conv1(x)
        x = self.o4_conv2(x)
        x = self.o4_conv3(x)
        x = self.sa(x) *x
        x = self.bn_sa4(x)

        branch_4_out = self.sigmoid(self.o4_conv4(x))

        x = self.trans_conv4(x)
        x = torch.cat((x,t1_f_l3,t2_f_l3),dim=1)
        x = self.ca5(x)*x
        x = self.o5_conv1(x)
        x = self.o5_conv2(x)
        x = self.o5_conv3(x)
        x = self.sa(x) *x
        x = self.bn_sa5(x)

        branch_5_out = self.sigmoid(self.o5_conv4(x))

        return branch_5_out,branch_4_out,branch_3_out,branch_2_out,branch_1_out
