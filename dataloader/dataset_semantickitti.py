# -*- coding:utf-8 -*-
# author: Xinge

"""
SemKITTI dataloader
"""
import os
import numpy as np
import torch
import random
import time
import numba as nb
import yaml
from torch.utils import data
import pickle

REGISTERED_DATASET_CLASSES = {}


def register_dataset(cls, name=None):
    global REGISTERED_DATASET_CLASSES
    if name is None:
        name = cls.__name__
    assert name not in REGISTERED_DATASET_CLASSES, f"exist class: {REGISTERED_DATASET_CLASSES}"
    REGISTERED_DATASET_CLASSES[name] = cls
    return cls


def get_model_class(name):
    global REGISTERED_DATASET_CLASSES
    assert name in REGISTERED_DATASET_CLASSES, f"available class: {REGISTERED_DATASET_CLASSES}"
    return REGISTERED_DATASET_CLASSES[name]


@register_dataset
class voxel_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, 50, 1.5], min_volume_space=[-50, -50, -3]):
        # Initialization
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.flip_aug = flip_aug
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        # Denotes the total number of samples
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        # Generates one sample of data
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # Random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360)
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # Random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        max_bound = np.percentile(xyz, 100, axis=0)
        min_bound = np.percentile(xyz, 0, axis=0)

        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # Get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size

        intervals = crop_range / (cur_grid_size - 1)
        if (intervals == 0).any():
            print("Zero interval!")

        grid_ind = (np.floor((np.clip(xyz, min_bound, max_bound) -
                              min_bound) / intervals)).astype(np.int)

        # Process voxel position
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(
            self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        # Process labels
        processed_label = np.ones(
            self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort(
            (grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(
            np.copy(processed_label), label_voxel_pair)

        data_tuple = (voxel_position, processed_label)

        # Center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * \
            intervals + min_bound
        return_xyz = xyz - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate(
                (return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)
        return data_tuple


def cart2polar(input_xyz):

    # Transformation between Cartesian coordinates and Cylindrical coordinates

    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)


def polar2cat(input_xyz_polar):

    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


@register_dataset
class cylinder_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False,
                 transform_aug=False, trans_std=[0.1, 0.1, 0.1],
                 min_rad=-np.pi / 4, max_rad=np.pi / 4):
        # Point cloud dataset, e.g. SemanticKITTI or NuScenes
        self.point_cloud_dataset = in_dataset

        # Voxel grid size
        self.grid_size = np.asarray(grid_size)

        # Data augmentation via rotation
        self.rotate_aug = rotate_aug

        # Data augmentation via flipping
        self.flip_aug = flip_aug

        # Data augmentation via scaling
        self.scale_aug = scale_aug

        # ?
        self.ignore_label = ignore_label

        # ?
        self.return_test = return_test

        # Whether fixed volume space or not
        self.fixed_volume_space = fixed_volume_space

        # Maximum volume space
        self.max_volume_space = max_volume_space

        # Minimum volume space
        self.min_volume_space = min_volume_space

        # Data augmentation via transformation/translation
        self.transform = transform_aug

        # ?
        self.trans_std = trans_std

        # Probability density of the rotation
        self.noise_rotation = np.random.uniform(min_rad, max_rad)

    def __len__(self):

        # Denotes the total number of samples
        return len(self.point_cloud_dataset)

    def rotation_points_single_angle(self, points, angle, axis=0):

        # Points: [N, 3]

        # Sinus of the rotation angle
        rot_sin = np.sin(angle)
        # Cosinus of the rotation angle
        rot_cos = np.cos(angle)

        # Various rotation matrices based on the specified axis
        if axis == 1:
            rot_mat_T = np.array(
                [[rot_cos, 0, -rot_sin], [0, 1, 0], [rot_sin, 0, rot_cos]],
                dtype=points.dtype)

        elif axis == 2 or axis == -1:
            rot_mat_T = np.array(
                [[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]],
                dtype=points.dtype)

        elif axis == 0:
            rot_mat_T = np.array(
                [[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]],
                dtype=points.dtype)

        else:
            raise ValueError("Axis should in rankge")

        # Apply the transformation
        return points @ rot_mat_T

    def __getitem__(self, index):
        """
        Generates one sample of data
        """

        # Get the data from the point cloud dataset, e.g. SemanticKITTI or NuScenes
        data = self.point_cloud_dataset[index]

        # If only coordinates and labels
        if len(data) == 2:
            xyz, labels = data
        # If also there is a signal strength
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # Random data augmentation by rotation
        if self.rotate_aug:

            # 0-45 degree random angle, in radians
            rotate_rad = np.deg2rad(np.random.random() * 90) - np.pi / 4
            # Calculate the sinus and cosinus of this angle
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            # Construct rotation matrix
            j = np.matrix([[c, s], [-s, c]])
            # Then rotate it, only the x-y coordinate-wise
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # Random data augmentation by flip x , y or x+y
        if self.flip_aug:
            # There are three possible modes, choose it randomly
            flip_type = np.random.choice(4, 1)

            # X-Flip
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            # Y-Flip
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            # XY-Flip
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]

        # Data augmentation via scaling
        if self.scale_aug:

            # Choose a random scale
            noise_scale = np.random.uniform(0.95, 1.05)
            # Simply scale it
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]

        # Data augmentation via translation
        if self.transform:
            # Construct random translation vector
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(
                                            0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T
            # Add it up to the original point cloud vector
            xyz[:, 0:3] += noise_translate

        # Convert cartesian coordinate into polar coordinates
        xyz_pol = cart2polar(xyz)

        # Get maximum resultant radius
        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        # Get minimum resultant radius
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)

        # Get maximum theta and z
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        # Get minimum theta and z
        min_bound = np.min(xyz_pol[:, 1:], axis=0)

        # Concatenate maximum and minimum results separately
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))

        # If there is a predefined fixed volume space boundary
        # forget about the calculated one, use it directly.
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)

        # Range of radius, theta, and z
        crop_range = max_bound - min_bound
        # Specified grid size
        cur_grid_size = self.grid_size
        # The required interval of (radius, theta, z) to complete specified grid size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any():
            print("Zero interval!")

        # Linearly space the grid indeces based on the interval and range
        grid_ind = (np.floor(
            (np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        # Initialize an empty array
        voxel_position = np.zeros(self.grid_size, dtype=np.float32)

        # ?
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1

        # Calculate the centers of the voxels 
        voxel_position = np.indices(
            self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)

        # Since, these are calculated in polar coordinates,
        # turn them into cartesian coordinates
        voxel_position = polar2cat(voxel_position)

        # ?
        processed_label = np.ones(
            self.grid_size, dtype=np.uint8) * self.ignore_label

        # Concatenate voxels and labels
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)

        # Perform a very sophisticated sorting
        label_voxel_pair = label_voxel_pair[np.lexsort(
            (grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]

        # Some preprocessing on the labels
        processed_label = nb_process_label(
            np.copy(processed_label), label_voxel_pair)

        # New tuple consisting of voxel position and the processed label
        data_tuple = (voxel_position, processed_label)

        # Center of the voxels
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * \
            intervals + min_bound

        # Concentrated on the origin form
        return_xyz = xyz_pol - voxel_centers

        # Construct a new variable consisting of 
        #
        # (polar coordinates concantrated on the origin form, 
        # polar coordinates original form, cartesian form)
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        # Construct partial output based on the input 
        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate(
                (return_xyz, sig[..., np.newaxis]), axis=1)
        

        # ?
        if self.return_test:
            
            data_tuple += (grid_ind, labels, return_fea, index)
        
        else:

            #  Return final output as:
            # (Voxel position, processed label, grid indices, corresponding label for each grid, partial output)
            data_tuple += (grid_ind, labels, return_fea)
        return data_tuple


@register_dataset
class polar_dataset(data.Dataset):
    def __init__(self, in_dataset, grid_size, rotate_aug=False, flip_aug=False, ignore_label=255, return_test=False,
                 fixed_volume_space=False, max_volume_space=[50, np.pi, 2], min_volume_space=[0, -np.pi, -4],
                 scale_aug=False):
        self.point_cloud_dataset = in_dataset
        self.grid_size = np.asarray(grid_size)
        self.rotate_aug = rotate_aug
        self.flip_aug = flip_aug
        self.scale_aug = scale_aug
        self.ignore_label = ignore_label
        self.return_test = return_test
        self.fixed_volume_space = fixed_volume_space
        self.max_volume_space = max_volume_space
        self.min_volume_space = min_volume_space

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data = self.point_cloud_dataset[index]
        if len(data) == 2:
            xyz, labels = data
        elif len(data) == 3:
            xyz, labels, sig = data
            if len(sig.shape) == 2:
                sig = np.squeeze(sig)
        else:
            raise Exception('Return invalid data tuple')

        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 45) - np.pi / 8
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        xyz_pol = cart2polar(xyz)

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)
        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))
        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        # get grid index
        crop_range = max_bound - min_bound
        cur_grid_size = self.grid_size
        intervals = crop_range / (cur_grid_size - 1)

        if (intervals == 0).any():
            print("Zero interval!")
        grid_ind = (np.floor(
            (np.clip(xyz_pol, min_bound, max_bound) - min_bound) / intervals)).astype(np.int)

        voxel_position = np.zeros(self.grid_size, dtype=np.float32)
        dim_array = np.ones(len(self.grid_size) + 1, int)
        dim_array[0] = -1
        voxel_position = np.indices(
            self.grid_size) * intervals.reshape(dim_array) + min_bound.reshape(dim_array)
        voxel_position = polar2cat(voxel_position)

        processed_label = np.ones(
            self.grid_size, dtype=np.uint8) * self.ignore_label
        label_voxel_pair = np.concatenate([grid_ind, labels], axis=1)
        label_voxel_pair = label_voxel_pair[np.lexsort(
            (grid_ind[:, 0], grid_ind[:, 1], grid_ind[:, 2])), :]
        processed_label = nb_process_label(
            np.copy(processed_label), label_voxel_pair)
        data_tuple = (voxel_position, processed_label)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * \
            intervals + min_bound
        return_xyz = xyz_pol - voxel_centers
        return_xyz = np.concatenate((return_xyz, xyz_pol, xyz[:, :2]), axis=1)

        if len(data) == 2:
            return_fea = return_xyz
        elif len(data) == 3:
            return_fea = np.concatenate(
                (return_xyz, sig[..., np.newaxis]), axis=1)

        if self.return_test:
            data_tuple += (grid_ind, labels, return_fea, index)
        else:
            data_tuple += (grid_ind, labels, return_fea)

        return data_tuple


@nb.jit('u1[:,:,:](u1[:,:,:],i8[:,:])', nopython=True, cache=True, parallel=False)
def nb_process_label(processed_label, sorted_label_voxel_pair):
    label_size = 256
    counter = np.zeros((label_size,), dtype=np.uint16)
    counter[sorted_label_voxel_pair[0, 3]] = 1
    cur_sear_ind = sorted_label_voxel_pair[0, :3]
    for i in range(1, sorted_label_voxel_pair.shape[0]):
        cur_ind = sorted_label_voxel_pair[i, :3]
        if not np.all(np.equal(cur_ind, cur_sear_ind)):
            processed_label[cur_sear_ind[0], cur_sear_ind[1],
                            cur_sear_ind[2]] = np.argmax(counter)
            counter = np.zeros((label_size,), dtype=np.uint16)
            cur_sear_ind = cur_ind
        counter[sorted_label_voxel_pair[i, 3]] += 1
    processed_label[cur_sear_ind[0], cur_sear_ind[1],
                    cur_sear_ind[2]] = np.argmax(counter)
    return processed_label


def collate_fn_BEV(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz


def collate_fn_BEV_test(data):
    data2stack = np.stack([d[0] for d in data]).astype(np.float32)
    label2stack = np.stack([d[1] for d in data]).astype(np.int)
    grid_ind_stack = [d[2] for d in data]
    point_label = [d[3] for d in data]
    xyz = [d[4] for d in data]
    index = [d[5] for d in data]
    return torch.from_numpy(data2stack), torch.from_numpy(label2stack), grid_ind_stack, point_label, xyz, index
