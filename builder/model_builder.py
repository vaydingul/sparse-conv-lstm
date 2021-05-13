# -*- coding:utf-8 -*-
# author: Xinge
# @file: model_builder.py

from network.cylinder_spconv_3d import get_model_class
from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
from network.cylinder_fea_generator import cylinder_fea


def build(model_config):

    # Grid size of each voxel
    output_shape = model_config['output_shape']
    # Number classes/categories
    num_class = model_config['num_class']
    # Number of input features
    num_input_features = model_config['num_input_features']

    # Whether batchnorm will be used or not!
    use_norm = model_config['use_norm']

    # ?
    init_size = model_config['init_size']

    # Feature dimension
    fea_dim = model_config['fea_dim']
    # Output feature dimension
    out_fea_dim = model_config['out_fea_dim']

    # Asymmetrical 3D Sparse Convolution Network
    cylinder_3d_spconv_seg = Asymm_3d_spconv(
        output_shape=output_shape,
        use_norm=use_norm,
        num_input_features=num_input_features,
        init_size=init_size,
        nclasses=num_class)

    # Cylindrical Feature Network
    cy_fea_net = cylinder_fea(grid_size=output_shape,
                              fea_dim=fea_dim,
                              out_pt_fea_dim=out_fea_dim,
                              fea_compre=num_input_features)

    # The general model architecture will be the one in the 
    # model_params/model_architecture, e.g. cylinder_asym
    model = get_model_class(model_config["model_architecture"])(
        cylin_model=cy_fea_net,
        segmentator_spconv=cylinder_3d_spconv_seg,
        sparse_shape=output_shape
    )

    return model
