# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/26

import numpy as np
import torch.utils.data as tordata


class SoftmaxSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        w = (np.asarray(dataset.label) > 2).astype('float')
        w[w == 1] = 1 / ((w == 1).sum() * 2)
        w[w == 0] = 1 / ((w == 0).sum() * 2)
        self.weight = w

    def __iter__(self):
        while (True):
            sample_indices = np.random.choice(
                a=self.dataset.data_size,
                size=self.batch_size,
                replace=False, p=self.weight)
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size
