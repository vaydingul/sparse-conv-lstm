# -*- coding:utf-8 -*-
# author: Xinge
# @file: metric_util.py 

import numpy as np


def fast_hist(pred, label, n):

    # Accept the indices whose label is greater than 0 and less than n
    k = (label >= 0) & (label < n)

    # Calculate the occurences of each element for (0, n ** 2) bins/interval
    bin_count = np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2)

    # Reshape the calculated bins as (n, n) then return it
    return bin_count[:n ** 2].reshape(n, n)


def per_class_iu(hist):

    # Smart formulation for per class intersection over union
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def fast_hist_crop(output, target, unique_label):
    # Calculate the histogram
    hist = fast_hist(output.flatten(), target.flatten(), np.max(unique_label) + 2)

    # Fetch only the cells belonging to the unique labels
    hist = hist[unique_label + 1, :]
    hist = hist[:, unique_label + 1]
    # Return it
    return hist
