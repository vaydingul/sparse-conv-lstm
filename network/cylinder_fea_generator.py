# -*- coding:utf-8 -*-
# author: Xinge

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numba as nb
import multiprocessing
import torch_scatter


class cylinder_fea(nn.Module):
    """

    Point cloud feature extractor module.

    """

    def __init__(self, grid_size, fea_dim=3,
                 out_pt_fea_dim=64, max_pt_per_encode=64, fea_compre=None):
        super(cylinder_fea, self).__init__()

        # Main point processing network
        self.PPmodel = nn.Sequential(
            nn.BatchNorm1d(fea_dim),

            nn.Linear(fea_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, out_pt_fea_dim)
        )

        #? Unused
        self.max_pt = max_pt_per_encode

        #? Unused
        self.grid_size = grid_size

        #? Unused
        kernel_size = 3
        self.local_pool_op = torch.nn.MaxPool2d(kernel_size, stride=1,
                                                padding=(kernel_size - 1) // 2,
                                                dilation=1)

        # Whether output will be compressed more or not
        self.fea_compre = fea_compre

        # The output dimension of the point processing network
        self.pool_dim = out_pt_fea_dim

        # Point feature compression module
        if self.fea_compre is not None:

            # Small Linear_ReLU network to downsize
            # the output
            self.fea_compression = nn.Sequential(
                nn.Linear(self.pool_dim, self.fea_compre),
                nn.ReLU())

            # If the data is compressed, then the output
            # size will equal to the size of the output
            # of the compression module
            self.pt_fea_dim = self.fea_compre

        else:

            # If it is not compressed, then the output
            # size will equal to the size of the output
            # of the point processing module
            self.pt_fea_dim = self.pool_dim

    def forward(self, pt_fea, xy_ind):
        """

        xy_ind = X-Y-Z coordinated of each point as integers.
        pt_fea = Feature vectors for each point (9 features).

        """

        # Fetch the current device
        cur_dev = pt_fea[0].get_device()

        # Concatenate the padded indices/coordinates of the points
        cat_pt_ind = []
        # For each batch
        for i_batch in range(len(xy_ind)):
            # Pad the input tensor
            cat_pt_ind.append(
                F.pad(xy_ind[i_batch], (1, 0), 'constant', value=i_batch))

        # Concatenate the tensors to have one operational variable
        cat_pt_fea = torch.cat(pt_fea, dim=0)
        cat_pt_ind = torch.cat(cat_pt_ind, dim=0)

        # Number of points
        pt_num = cat_pt_ind.shape[0]

        # Create index list of shuffled integers
        shuffled_ind = torch.randperm(pt_num , device='cuda:0')

        # Shuffle the tensors via using the above list
        cat_pt_fea = cat_pt_fea[shuffled_ind, :]
        cat_pt_ind = cat_pt_ind[shuffled_ind, :]

        # Fetch the unique point indices
        unq, unq_inv, unq_cnt = torch.unique(
            cat_pt_ind, return_inverse=True, return_counts=True, dim=0)

        # Convert to the Int64 type
        unq = unq.type(torch.int64)

        # Process the point features
        processed_cat_pt_fea = self.PPmodel(cat_pt_fea)

        # It performs a max-pooling on the point which corresponds to the same points (index-wise)
        pooled_data = torch_scatter.scatter_max(
            processed_cat_pt_fea, unq_inv, dim=0)[0]


        # If the feature compression is True,
        if self.fea_compre:

            # Then, reduce the output size by perceptron layer
            processed_pooled_data = self.fea_compression(pooled_data)

        else:

            processed_pooled_data = pooled_data

        return unq, processed_pooled_data
