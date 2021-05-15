# -*- coding:utf-8 -*-
# author: Xinge
# @file: cylinder_spconv_3d.py

from torch import nn

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
                 sparse_shape,
                 ):
        super().__init__()

        # The name of the model
        self.name = "cylinder_asym"

        # Cylindrical point processing module
        self.cylinder_3d_generator = cylin_model

        # 
        self.cylinder_3d_spconv_seg = segmentator_spconv

        # Grid size
        self.sparse_shape = sparse_shape

    def forward(self, train_pt_fea_ten, train_vox_ten, batch_size):
        
        # Process the point features, and max-pool them according to the unique point coordinates
        # As an output, get unique point coordinates, and the processed features of the points
        coords, features_3d = self.cylinder_3d_generator(train_pt_fea_ten, train_vox_ten)

        
        spatial_features = self.cylinder_3d_spconv_seg(features_3d, coords, batch_size)

        return spatial_features
