# -*- coding:utf-8 -*-
# author: Xinge
# @file: segmentator_3d_asymm_spconv.py

import numpy as np
import spconv
import torch
from torch import nn


def conv3x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


def conv1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 3), stride=stride,
                             padding=(0, 1, 1), bias=False, indice_key=indice_key)


def conv1x1x3(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 1, 3), stride=stride,
                             padding=(0, 0, 1), bias=False, indice_key=indice_key)


def conv1x3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(1, 3, 1), stride=stride,
                             padding=(0, 1, 0), bias=False, indice_key=indice_key)


def conv3x1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 1), stride=stride,
                             padding=(1, 0, 0), bias=False, indice_key=indice_key)


def conv3x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=(3, 1, 3), stride=stride,
                             padding=(1, 0, 1), bias=False, indice_key=indice_key)


def conv1x1(in_planes, out_planes, stride=1, indice_key=None):
    return spconv.SubMConv3d(in_planes, out_planes, kernel_size=1, stride=stride,
                             padding=1, bias=False, indice_key=indice_key)


class ResContextBlock(nn.Module):
    """

    Asymmetrical Residual Block

    """

    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ResContextBlock, self).__init__()

        #! 1 x 3 x 3
        self.conv1 = conv1x3(in_filters, out_filters,
                             indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.LeakyReLU()

        #! 3 x 1 x 3
        self.conv1_2 = conv3x1(out_filters, out_filters,
                               indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.LeakyReLU()

        #! 3 x 1 x 3
        self.conv2 = conv3x1(in_filters, out_filters,
                             indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        #! 1 x 3 x 3
        self.conv3 = conv1x3(out_filters, out_filters,
                             indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        # Initialize the weights of the BatchNorm1d modules
        self.weight_initialization()

    def weight_initialization(self):
        """

        Assign constant initial weight and bias to the
        BatchNorm1d module.

        """

        for m in self.modules():

            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # * Process input in (1 x 3 x 3) sparse convolution module
        shortcut = self.conv1(x)
        shortcut.features = self.act1(shortcut.features)
        shortcut.features = self.bn0(shortcut.features)

        # Process the output of above layer in
        # (3 x 1 x 3) sparse convolution module
        shortcut = self.conv1_2(shortcut)
        shortcut.features = self.act1_2(shortcut.features)
        shortcut.features = self.bn0_2(shortcut.features)

        # * Process input in (3 x 1 x 3) sparse convolution module
        resA = self.conv2(x)
        resA.features = self.act2(resA.features)
        resA.features = self.bn1(resA.features)

        # Process the output of above layer in
        # (1 x 3 x 3) sparse convolution module
        resA = self.conv3(resA)
        resA.features = self.act3(resA.features)
        resA.features = self.bn2(resA.features)

        # At the end output the summation of the two bracnhes.
        resA.features = resA.features + shortcut.features

        return resA


class ResBlock(nn.Module):

    """

    Asymmetrical Downsample Block (AD)

    """

    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3, 3), stride=1,
                 pooling=True, drop_out=True, height_pooling=False, indice_key=None):
        super(ResBlock, self).__init__()

        # Whether the features will be pooled or not
        self.pooling = pooling
        # Whether the dropout will be applied or not
        self.drop_out = drop_out

        #! 3 x 1 x 3
        self.conv1 = conv3x1(in_filters, out_filters,
                             indice_key=indice_key + "bef")
        self.act1 = nn.LeakyReLU()
        self.bn0 = nn.BatchNorm1d(out_filters)

        #! 1 x 3 x 3
        self.conv1_2 = conv1x3(out_filters, out_filters,
                               indice_key=indice_key + "bef")
        self.act1_2 = nn.LeakyReLU()
        self.bn0_2 = nn.BatchNorm1d(out_filters)

        #! 1 x 3 x 3
        self.conv2 = conv1x3(in_filters, out_filters,
                             indice_key=indice_key + "bef")
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        #! 3 x 1 x 3
        self.conv3 = conv3x1(out_filters, out_filters,
                             indice_key=indice_key + "bef")
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        # If the pooling is True
        if pooling:
            # Based on whether height pooling is True or not,
            # constuct pooling object/function
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)

        # Initialize the weights of the BatchNorm1d modules
        self.weight_initialization()

    def weight_initialization(self):
        """

        Assign constant initial weight and bias to the
        BatchNorm1d module.

        """
        for m in self.modules():

            if isinstance(m, nn.BatchNorm1d):

                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # * Process input with 3 x 1 x 3 sparse convolutional kernels
        shortcut = self.conv1(x)
        shortcut.features = self.act1(shortcut.features)
        shortcut.features = self.bn0(shortcut.features)

        # Process the result of above layer with 1 x 3 x 3 sparse convolutional kernels
        shortcut = self.conv1_2(shortcut)
        shortcut.features = self.act1_2(shortcut.features)
        shortcut.features = self.bn0_2(shortcut.features)

        # * Process input with 1 x 3 x 3 sparse convolutional kernels
        resA = self.conv2(x)
        resA.features = self.act2(resA.features)
        resA.features = self.bn1(resA.features)

        # Process the result of above layer with 1 x 3 x 3 sparse convolutional kernels
        resA = self.conv3(resA)
        resA.features = self.act3(resA.features)
        resA.features = self.bn2(resA.features)

        # Sum the features coming from two branches
        resA.features = resA.features + shortcut.features

        # If the pooling option is True,
        if self.pooling:
            # Then, apply the pooling
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(nn.Module):

    """

    Asymmetrical Upsample Block (AU)

    """

    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), indice_key=None, up_key=None):
        super(UpBlock, self).__init__()

        # ? Unused
        # self.drop_out = drop_out

        #! 3 x 3 x 3
        self.trans_dilao = conv3x3(
            in_filters, out_filters, indice_key=indice_key + "new_up")
        self.trans_act = nn.LeakyReLU()
        self.trans_bn = nn.BatchNorm1d(out_filters)

        #! 1 x 3 x 3
        self.conv1 = conv1x3(out_filters, out_filters, indice_key=indice_key)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(out_filters)

        #! 3 x 1 x 3
        self.conv2 = conv3x1(out_filters, out_filters, indice_key=indice_key)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm1d(out_filters)

        #! 3 x 3 x 3
        self.conv3 = conv3x3(out_filters, out_filters, indice_key=indice_key)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm1d(out_filters)

        # ? Unused
        # self.dropout3 = nn.Dropout3d(p=dropout_rate)

        #! 3 x 3 x 3 De/Inverse Convolution Block
        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

        # Initialize the weights of the BatchNorm1d modules
        self.weight_initialization()

    def weight_initialization(self):
        """

        Assign constant initial weight and bias to the
        BatchNorm1d module.

        """

        for m in self.modules():

            if isinstance(m, nn.BatchNorm1d):

                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, skip):

        # * Process input with 3 x 3 x 3 sparse convolutional kernels
        upA = self.trans_dilao(x)
        upA.features = self.trans_act(upA.features)
        upA.features = self.trans_bn(upA.features)

        # Upsample the result of the above layer
        upA = self.up_subm(upA)
        
        #* Sum the upsammpled results with the low level features
        upA.features = upA.features + skip.features

        # Process the result of above layer with 1 x 3 x 3 sparse convolutional kernels
        upE = self.conv1(upA)
        upE.features = self.act1(upE.features)
        upE.features = self.bn1(upE.features)

        # Process the result of above layer with 3 x 1 x 3 sparse convolutional kernels
        upE = self.conv2(upE)
        upE.features = self.act2(upE.features)
        upE.features = self.bn2(upE.features)

        # Process the result of above layer with 3 x 3 x 3 sparse convolutional kernels
        upE = self.conv3(upE)
        upE.features = self.act3(upE.features)
        upE.features = self.bn3(upE.features)

        return upE


class ReconBlock(nn.Module):
    """

    Dimension Decomposition Based Context Modeling module (DDCM)

    """

    def __init__(self, in_filters, out_filters, kernel_size=(3, 3, 3), stride=1, indice_key=None):
        super(ReconBlock, self).__init__()

        #! 3 x 1 x 1
        self.conv1 = conv3x1x1(in_filters, out_filters,
                               indice_key=indice_key + "bef")
        self.bn0 = nn.BatchNorm1d(out_filters)
        self.act1 = nn.Sigmoid()

        #! 1 x 3 x 1
        self.conv1_2 = conv1x3x1(
            in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_2 = nn.BatchNorm1d(out_filters)
        self.act1_2 = nn.Sigmoid()

        #! 1 x 1 x 3
        self.conv1_3 = conv1x1x3(
            in_filters, out_filters, indice_key=indice_key + "bef")
        self.bn0_3 = nn.BatchNorm1d(out_filters)
        self.act1_3 = nn.Sigmoid()

    def forward(self, x):

        #* Process the input with 3 x 1 x 1 sparse convolution kernels
        shortcut = self.conv1(x)
        shortcut.features = self.bn0(shortcut.features)
        shortcut.features = self.act1(shortcut.features)

        # Process the result of the above layer with
        # 1 x 3 x 1 sparse convolution kernels
        shortcut2 = self.conv1_2(x)
        shortcut2.features = self.bn0_2(shortcut2.features)
        shortcut2.features = self.act1_2(shortcut2.features)

        # Process the result of the above layer with
        # 1 x 1 x 3 sparse convolution kernels
        shortcut3 = self.conv1_3(x)
        shortcut3.features = self.bn0_3(shortcut3.features)
        shortcut3.features = self.act1_3(shortcut3.features)
        shortcut.features = shortcut.features + shortcut2.features + shortcut3.features

        shortcut.features = shortcut.features * x.features

        return shortcut


class Asymm_3d_spconv(nn.Module):
    """

    Asymmetrical 3D Sparse Convolution Network

    The whole U-Net-like structure

    """

    def __init__(self,
                 output_shape,
                 use_norm=True,
                 num_input_features=128,
                 nclasses=20, n_height=32, strict=False, init_size=16):

        super(Asymm_3d_spconv, self).__init__()

        # Number of unique classes
        self.nclasses = nclasses

        # ?
        self.nheight = n_height

        # ?
        self.strict = False

        # Grid size
        sparse_shape = np.array(output_shape)

        # ? Unused
        # sparse_shape[0] = 11

        # Grid size
        self.sparse_shape = sparse_shape

        #! Asymmetrical Residual Block
        self.downCntx = ResContextBlock(
            num_input_features, init_size, indice_key="pre")

        #! Asymmetrical Downsmaple Block (AD)
        self.resBlock2 = ResBlock(
            init_size, 2 * init_size, 0.2, height_pooling=True, indice_key="down2")

        #! Asymmetrical Downsmaple Block (AD)
        self.resBlock3 = ResBlock(
            2 * init_size, 4 * init_size, 0.2, height_pooling=True, indice_key="down3")

        #! Asymmetrical Downsmaple Block (AD)
        self.resBlock4 = ResBlock(4 * init_size, 8 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down4")

        #! Asymmetrical Downsmaple Block (AD)
        self.resBlock5 = ResBlock(8 * init_size, 16 * init_size, 0.2, pooling=True, height_pooling=False,
                                  indice_key="down5")

        #! Asymmetrical Upsample Block (AU)
        self.upBlock0 = UpBlock(16 * init_size, 16 *
                                init_size, indice_key="up0", up_key="down5")

        #! Asymmetrical Upsample Block (AU)
        self.upBlock1 = UpBlock(
            16 * init_size, 8 * init_size, indice_key="up1", up_key="down4")

        #! Asymmetrical Upsample Block (AU)
        self.upBlock2 = UpBlock(
            8 * init_size, 4 * init_size, indice_key="up2", up_key="down3")

        #! Asymmetrical Upsample Block (AU)
        self.upBlock3 = UpBlock(
            4 * init_size, 2 * init_size, indice_key="up3", up_key="down2")

        #! Dimension Decomposition Based Context Modeling module (DDCM)
        self.ReconNet = ReconBlock(
            2 * init_size, 2 * init_size, indice_key="recon")

        #! Final log-probabilities
        self.logits = spconv.SubMConv3d(4 * init_size, nclasses, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

    def forward(self, voxel_features, coors, batch_size):
        
        # ? Unused        
        # x = x.contiguous()

        # ? Unused
        # import pdb
        # pdb.set_trace()

        # Make sure the coordinates are integer 
        coors = coors.int()
        
        # Transform the input arguments into an individual Sparse Tensor
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        
        #* Apply Asymmetrical Residual Block
        ret = self.downCntx(ret)

        #* Apply Asymmetrical Downsmaple Block (AD1) to the output of above layer
        down1c, down1b = self.resBlock2(ret)

        #* Apply Asymmetrical Downsmaple Block (AD2) to the output of above layer
        down2c, down2b = self.resBlock3(down1c)

        #* Apply Asymmetrical Downsmaple Block (AD3) to the output of above layer
        down3c, down3b = self.resBlock4(down2c)

        #* Apply Asymmetrical Downsmaple Block (AD4) to the output of above layer
        down4c, down4b = self.resBlock5(down3c)

        #* Apply Asymmetrical Upsample Block (AU1) to the output of above layer and the 
        #* skip connection coming from (AD4)
        up4e = self.upBlock0(down4c, down4b)

        #* Apply Asymmetrical Upsample Block (AU2) to the output of above layer and the 
        #* skip connection coming from (AD3)
        up3e = self.upBlock1(up4e, down3b)

        #* Apply Asymmetrical Upsample Block (AU3) to the output of above layer and the 
        #* skip connection coming from (AD2)
        up2e = self.upBlock2(up3e, down2b)

        #* Apply Asymmetrical Upsample Block (AU4) to the output of above layer and the 
        #* skip connection coming from (AD1)
        up1e = self.upBlock3(up2e, down1b)

        #* Apply Dimension Decomposition Based Context Modeling module (DDCM) 
        #* to the output of the above layer
        up0e = self.ReconNet(up1e)

        # Concatenate the features coming from DDCM and AU4
        up0e.features = torch.cat((up0e.features, up1e.features), 1)

        #* Calculate the log-probabilities
        logits = self.logits(up0e)

        # Transform into a dense form 
        y = logits.dense()

        # Return the dense log-probabilities
        return y
