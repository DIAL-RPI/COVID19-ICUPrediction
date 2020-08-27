# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/26

import torch
import torch.nn as nn


class BaseDense(nn.Module):
    def __init__(self, inp, out, bn=False):
        super(BaseDense, self).__init__()
        self.fc = nn.Linear(inp, out)
        self.relu = nn.LeakyReLU(inplace=True)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm1d(out)

    def forward(self, x):
        x = self.relu(self.fc(x))
        if self.bn is not None:
            x = self.bn(x)
        return x


class DVBBranch(nn.Module):
    def __init__(self, num_classes, dout=False):
        super(DVBBranch, self).__init__()
        self.fuse_1 = BaseDense(5, 16)
        self.fuse_2 = BaseDense(16, 8)
        self.fc_pred = nn.Linear(8, num_classes)
        self.dout = None
        if dout:
            self.dout = nn.Dropout()
        print(self.dout)

    def forward(self, dvb_data):
        feature = self.fuse_1(dvb_data)
        feature = self.fuse_2(feature)
        feature = feature / (feature.norm(dim=1, keepdim=True) + 1e-9)
        if self.dout is not None:
            feature = self.dout(feature)
        dvb_pred = self.fc_pred(feature)

        return dvb_pred, feature


class HLQBranch(nn.Module):
    def __init__(self, num_classes, dout=False):
        super(HLQBranch, self).__init__()
        self.fc_1 = BaseDense(64, 16)
        self.fc_pred = nn.Linear(16, num_classes)
        self.dout = None
        if dout:
            self.dout = nn.Dropout()
        print(self.dout)

    def forward(self, hlq_data):
        feature = self.fc_1(hlq_data)

        feature = feature / (feature.norm(dim=1, keepdim=True) + 1e-9)
        if self.dout is not None:
            feature = self.dout(feature)

        hlq_pred = self.fc_pred(feature)

        return hlq_pred, feature


class RadiBranch(nn.Module):
    def __init__(self, num_classes, dout=False):
        super(RadiBranch, self).__init__()
        self.shape_fc = BaseDense(17, 16)
        self.first_ord_fc = BaseDense(324, 16)
        self.second_ord_fc = BaseDense(375, 16)
        self.high_level_fc = BaseDense(975, 16)
        self.fc_1 = BaseDense(64, 16)
        self.fc_pred = nn.Linear(16, num_classes)

        self.dout = None
        if dout:
            self.dout = nn.Dropout()
        print(self.dout)

    def forward(self, radi):
        shape = radi[:, :17]
        first = radi[:, 17:341]
        second = radi[:, 341:716]
        high = radi[:, 716:]
        shape = self.shape_fc(shape)
        first = self.first_ord_fc(first)
        second = self.second_ord_fc(second)
        high = self.high_level_fc(high)
        radi = torch.cat([shape, first, second, high], dim=1)
        feature = self.fc_1(radi)

        feature = feature / (feature.norm(dim=1, keepdim=True) + 1e-9)

        if self.dout is not None:
            feature = self.dout(feature)

        radi_pred = self.fc_pred(feature)

        return radi_pred, feature


class COVIDNet(nn.Module):
    def __init__(self, num_classes=2, dout=False):
        super(COVIDNet, self).__init__()

        print('num_classes', num_classes)
        self.num_classes = num_classes
        self.radi_branch = RadiBranch(num_classes, dout)
        self.dvb_branch = DVBBranch(num_classes, dout)
        self.hlq_branch = HLQBranch(num_classes, dout)
        self.fuse_pred = nn.Linear(16 + 16 + 8, num_classes)
        self.dout = None
        if dout:
            self.dout = nn.Dropout()

    def forward(self, radi_data, dvb_data, hlq_data):
        radi_pred, radi_feature = self.radi_branch(radi_data)
        dvb_pred, dvb_feature = self.dvb_branch(dvb_data)
        hlq_pred, hlq_feature = self.hlq_branch(hlq_data)

        feature = torch.cat([hlq_feature, dvb_feature, radi_feature], dim=1)
        feature = feature / (feature.norm(dim=1, keepdim=True) + 1e-9)
        pred = self.fuse_pred(feature)

        return pred, radi_pred, dvb_pred, hlq_pred
