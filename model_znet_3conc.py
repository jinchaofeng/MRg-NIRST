#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np
import os

# os.environ["CUDA_VISIBLE_DEVICES"]= "0,1"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        # self.bn1 = nn.BatchNorm3d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        # self.bn2 = nn.BatchNorm3d(self.out_channels)

        # if self.pooling:
        #     self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
            # self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        # x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        # x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        before_pool = x
        if self.pooling:
            [a,b,c,d,e] = x.shape
            c = int(c/2)
            d = int(d/2)
            e = int(e/2)
            x = F.adaptive_max_pool3d(x,(c,d,e))
        return x, before_pool


class upConv4(nn.Module):

    def __init__(self, in_channels, out_channels, pooling=True):
        super(upConv4, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = upconv2x2(self.in_channels, self.out_channels)
        # self.bn1 = nn.BatchNorm3d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        # self.bn2 = nn.BatchNorm3d(self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool3d(kernel_size=1, stride=2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)

        # before_pool = x
        # if self.pooling:
        #     x = self.pool(x)
        # return x, before_pool

        return x

class Bottom(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Bottom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm3d(self.out_channels)
        self.conv2 = nn.Conv3d(self.out_channels, self.out_channels,
                               kernel_size=(2, 2), stride=(1, 1), padding=(0, 1))
        self.conv3 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm3d(self.out_channels)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.01)

        return x


class ImBottom(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImBottom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv1x1(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm3d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm3d(self.out_channels)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        # x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        # x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)

        return x


class UpConv4(nn.Module):

    def __init__(self, in_channels, out_channels,
                 merge_mode='add', up_mode='transpose'):
        super(UpConv4, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels * 2, self.in_channels * 2,
                                mode=self.up_mode)

        if self.merge_mode == 'add':
            self.conv1 = conv3x3(
                self.in_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same,concat
            self.conv1 = conv3x3(2 * self.in_channels, self.out_channels)
            self.bn1 = nn.BatchNorm3d(self.out_channels)
            self.conv2 = conv3x3(self.out_channels, self.out_channels)
            self.bn2 = nn.BatchNorm3d(self.out_channels)

    def forward(self, from_down1, from_down2):
        """ Forward pass
        Arguments:
            from_down1: tensor from the data encoder pathway
            from_down2: tensor from the das encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        # from_up = self.upconv(from_up)
        # print('from_up:',from_up.shape)
        # print('from_down1:',from_down1.shape)
        # print('from_down2:',from_down2.shape)

        if self.merge_mode == 'add':
            x = from_down1 + from_down2

        else:
            # concat
            x = torch.cat((from_down1, from_down2), 1)
            # print('x.shape',x.shape)
        x = self.upconv(x)
        # print('x.shape', x.shape)
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        # x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        # x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        # x = self.upconv(x)

        return x


class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels,
                 merge_mode='add', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.out_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'add':
            self.conv1 = conv3x3(
                self.in_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same,concat
            self.conv1 = conv3x3(3 * self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm3d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm3d(self.out_channels)

    def forward(self, from_down1, from_down2, from_up):
        """ Forward pass
        Arguments:
            from_down1: tensor from the data encoder pathway
            from_down2: tensor from the das encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        # from_up = self.upconv(from_up)
        # print('from_up:',from_up.shape)
        # print('from_down1:',from_down1.shape)
        # print('from_down2:',from_down2.shape)

        if self.merge_mode == 'add':
            x = from_up + from_down1 + from_down2

        else:
            # concat
            x = torch.cat((from_up, from_down1, from_down2), 1)
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        # print('x1.shape', x.shape)
        x = self.upconv(x)
        #
        # x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        # x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        return x


class UpConv6(nn.Module):

    def __init__(self, in_channels, out_channels,
                 merge_mode='add', up_mode='transpose'):
        super(UpConv6, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.out_channels, self.out_channels,
                                mode=self.up_mode)

        if self.merge_mode == 'add':
            self.conv1 = conv3x3(
                self.in_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same,concat
            self.conv1 = conv3x3(3 * self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm3d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm3d(self.out_channels)

    def forward(self, from_down1, from_down2, from_up):
        """ Forward pass
        Arguments:
            from_down1: tensor from the data encoder pathway
            from_down2: tensor from the das encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        # from_up = self.upconv(from_up)
        # print('from_up:',from_up.shape)
        # print('from_down1:',from_down1.shape)
        # print('from_down2:',from_down2.shape)

        if self.merge_mode == 'add':
            x = from_up + from_down1 + from_down2

        else:
            # concat
            x = torch.cat((from_up, from_down1, from_down2), 1)
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)

        # x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        # x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        return x

# Model 1 modified Unet for beamforming
class ZNet_3d(nn.Module):

    def __init__(self, in_channels=3, up_mode='transpose', merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(ZNet_3d, self).__init__()
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.in_channels = in_channels
        self.down0 = DownConv(1, 64)
        self.upp1 = upConv4(64, 32)  ###
        self.upp2 = upConv4(32, 16)
        self.upp3 = upConv4(16, 8)

        self.downn1 = DownConv(64, 128)
        self.downn2 = DownConv(128, 256)
        # self.upp4 = upConv4(64,128)
        self.down1 = DownConv(1, 8)  ###
        self.down2 = DownConv(8, 16)
        self.down3 = DownConv(16, 32)
        self.down4 = DownConv(32, 64)
        self.down5 = DownConv(64, 128)
        self.down6 = DownConv(128, 256)
        # self.bottom = Bottom(1,16)
        self.up1 = UpConv4(256, 128, merge_mode=self.merge_mode)

        self.up2 = UpConv(128, 64, merge_mode=self.merge_mode)
        self.up3 = UpConv(64, 32, merge_mode=self.merge_mode)
        self.up4 = UpConv(32, 16, merge_mode=self.merge_mode)
        self.up5 = UpConv(16, 8, merge_mode=self.merge_mode)

        self.up6 = UpConv6(8, 3, merge_mode=self.merge_mode)
        self.bottom = ImBottom(3, 3)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv3d):
            m.weight = init.kaiming_normal_(m.weight)
            init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, bfimg):
        # encoder1: raw data
        # print('x:', x.shape)
        # x = F.interpolate(x, (16, 16, 18), mode='trilinear')
        x = F.adaptive_max_pool3d(x,(16,16,18))
        # print('x:',x.shape)

        x, xbefore_pool = self.down0(x)
        # print('x, xbefore_pool', x.shape, xbefore_pool.shape)
        x1 = self.upp1(xbefore_pool)  # 1280, 64,32
        # print('x1', x1.shape)
        x2 = self.upp2(x1)  # 1280, 64,32
        # print('x2', x2.shape)
        x3 = self.upp3(x2)  # 1280, 64,32
        # print('x3', x3.shape)

        dx1, dxbefore_pool1 = self.downn1(xbefore_pool)
        # print('dx1:',dx1.shape,'dxbefore_pool1:',dxbefore_pool1.shape)
        dx2, dxbefore_pool2 = self.downn2(dx1)
        # print('dx2:', dx2.shape, 'dxbefore_pool2:', dxbefore_pool2.shape)

        # encoder2: bf
        bx1, bxbefore_pool1 = self.down1(bfimg)
        # print('bx1,bxbefore_pool1',bx1.shape,bxbefore_pool1.shape)
        bx2, bxbefore_pool2 = self.down2(bx1)
        # print('bx2,bxbefore_pool2', bx2.shape, bxbefore_pool2.shape)
        bx3, bxbefore_pool3 = self.down3(bx2)
        # print('bx3,bxbefore_pool3', bx3.shape, bxbefore_pool3.shape)
        bx4, bxbefore_pool4 = self.down4(bx3)
        # print('bx4,bxbefore_pool4',bx4.shape,bxbefore_pool4.shape)
        bx5, bxbefore_pool5 = self.down5(bx4)
        # print('bx5,bxbefore_pool5', bx5.shape, bxbefore_pool5.shape)
        bx6, bxbefore_pool6 = self.down6(bx5)
        # print('bx6,bxbefore_pool6', bx6.shape, bxbefore_pool6.shape)

        out1 = self.up1(dx2, bxbefore_pool6)  # 16, 16,128  此时out为5的
        # print('out1:', out1.shape)
        out1 = F.adaptive_max_pool3d(out1, (8, 8, 9))

        out2 = self.up2(dx1, bxbefore_pool5, out1)  # 32, 32,64
        # print('out2:', out2.shape)
        out3 = self.up3(xbefore_pool, bxbefore_pool4, out2)  # 64, 64,32
        # print('out3:', out3.shape)
        out4 = self.up4(x1, bxbefore_pool3, out3)  # 128, 128,1
        # print('out4:',out4.shape)
        out5 = self.up5(x2, bxbefore_pool2, out4)  # 128, 128,1
        # print('out5:', out5.shape)
        out6 = self.up6(x3, bxbefore_pool1, out5)  # 128, 128,1
        # print('out6:', out6.shape)
        out = self.bottom(out6)
        # print('out:', out.shape)

        return out


if __name__ == "__main__":
    """
    testing
    """


    # device =  torch.device('cuda:0')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def weight_init(m):
        if isinstance(m, nn.Conv3d):
            m.weight = init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)

    x = Variable(torch.FloatTensor(np.random.random((1, 1, 15, 16, 9))), requires_grad=True)
    img = Variable(torch.FloatTensor(np.random.random((1, 1, 128, 128, 144))), requires_grad=True)

    model = ZNet_3d(in_channels=1, merge_mode='concat')
    model = torch.nn.DataParallel(model)
    model.apply(weight_init)
    # model = model.to(device)

    # print(model)
    # x_mean = torch.mean(x)
    # print(x_mean)
    # img_mean = torch.mean(img)
    # print(img_mean)
    out = model(x, img)
    print("out:", out.shape)
    loss = torch.mean(out)

    loss.backward()

    print(loss)

    # print('# discriminator parameters:', sum(param.numel() for param in model.parameters()))
