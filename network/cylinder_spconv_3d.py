# -*- coding:utf-8 -*-
# author: Xinge
# @file: cylinder_spconv_3d.py

from torch import nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
import spconv

REGISTERED_MODELS_CLASSES = {}


def register_model(cls, name=None):
    global REGISTERED_MODELS_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_MODELS_CLASSES, f"exist class: {REGISTERED_MODELS_CLASSES}"
    REGISTERED_MODELS_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_MODELS_CLASSES
    assert name in REGISTERED_MODELS_CLASSES, f"available class: {REGISTERED_MODELS_CLASSES}"
    return REGISTERED_MODELS_CLASSES[name]


@register_model
class cylinder_asym(nn.Module):

    def __init__(self,
                 cylin_model,
                 segmentator_spconv,
                 sparse_conv_lstm_net,
                 sparse_shape,
                 ):
        super().__init__()

        # The name of the model
        self.name = "cylinder_asym"

        # Cylindrical point processing module
        self.cylinder_3d_generator = cylin_model

        #
        self.cylinder_3d_spconv_seg = segmentator_spconv

        self.sparse_conv_lstm_net = sparse_conv_lstm_net

        # Grid size
        self.sparse_shape = sparse_shape

        #! Final log-probabilities
        # self.logits = spconv.SubMConv3d(self.sparse_conv_lstm_net.hidden_dim, 21, kernel_size=3, stride=1, padding=1,
        #                                bias = True)

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):

        # Process the point features, and max-pool them according to the unique point coordinates
        # As an output, get unique point coordinates, and the processed features of the points
        coords, features_3d=self.cylinder_3d_generator(
            train_pt_fea_ten, train_vox_ten)

        # Process the calculated features and the unique point coordinates in
        # Asymmetrical Sparse Convolutional Network and calculate the
        # log-probabilities per point
        # spatial_features = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)

        sparse_features, predictions=self.cylinder_3d_spconv_seg(
            features_3d, coords, batch_size)

        sparse_features_= [sparse_features.features[coords[:, 0] == i] for i in range(batch_size)]
        coords_=  [coords[[coords[:, 0] == i]] for i in range(batch_size)]
        
        for k in range(batch_size):
            coords_[k][:,0] = 0

        sparse_features_padded=pad_sequence(sparse_features_, True)
        coords_padded=pad_sequence(coords_, True, 1000)

        sparse_features_conv_tensor_list=[]


        for k in range(batch_size):


            sparse_features_conv_tensor_list.append(spconv.SparseConvTensor(
                sparse_features_padded[k], coords_padded[k].int(), self.sparse_shape, 1))

        out=self.sparse_conv_lstm_net(sparse_features_conv_tensor_list, coords_padded.int(), 1)

        out_sparse=out[1][0][0]

        out_sparse.features=out_sparse.features[:sparse_features_[int((len(coords_)+1)*0.5) - 1].shape[0]]
        out_sparse.indices=out_sparse.indices[:coords_[int((len(coords_)+1)*0.5) - 1].shape[0]]

        # logits = self.logits(out_sparse)

        # predictions = logits.dense()

        predictions=out_sparse.dense()

        return predictions
