# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/26

import torch.utils.data as tordata


class DataSet(tordata.Dataset):
    def __init__(self, data, label):
        super(DataSet, self).__init__()
        # Separate wlr, dvb, hlq features
        self.radi_data = data[:, :1691]
        self.dvb_data = data[:, 1691:1755]
        self.hlq_data = data[:, 1755:]
        self.label = label
        self.data_size = len(self.label)
        self.label_set = set(self.label)

    def __getitem__(self, index):
        return self.radi_data[index], self.dvb_data[index], self.hlq_data[index], self.label[index]

    def __len__(self):
        return len(self.label)
