# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/26

import os
import os.path as osp
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tordata

from dataset import DataSet
from net import COVIDNet
from sampler import SoftmaxSampler


class Model:
    def __init__(
            self,
            dout,
            lr,
            num_classes,
            num_workers,
            batch_size,
            restore_iter,
            total_iter,
            save_name,
            model_name,
            pretrain_point,
            train_source,
            val_source,
            test_source, ):

        self.dout = dout
        self.lr = lr
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.restore_iter = restore_iter
        self.total_iter = total_iter
        self.save_name = save_name
        self.model_name = model_name
        self.train_source = train_source
        self.val_source = val_source
        self.test_source = test_source

        self.encoder = nn.DataParallel(COVIDNet(self.num_classes, dout=self.dout).cuda().float())
        self.ce = nn.DataParallel(nn.CrossEntropyLoss(reduction='none').cuda())

        optimizer = optim.Adam([
            {'params': self.encoder.parameters()},
        ], lr=self.lr, weight_decay=1e-4)
        self.optimizer = optimizer

        self.loss = []

    def fit(self):
        if self.restore_iter != 0:
            self.load_model()

        self.encoder.train()

        softmax_sampler = SoftmaxSampler(self.train_source, self.batch_size)
        train_loader = tordata.DataLoader(
            dataset=self.train_source,
            batch_sampler=softmax_sampler,
            num_workers=self.num_workers)

        _time1 = datetime.now()
        for radi_data, dvb_data, hlq_data, labels in train_loader:
            if self.restore_iter > self.total_iter:
                break
            self.restore_iter += 1
            self.optimizer.zero_grad()

            pred, radi_pred, dvb_pred, hlq_pred = self.encoder(
                radi_data.cuda().float(), dvb_data.cuda().float(),
                hlq_data.cuda().float())

            labels = (labels > 2).int()
            main_loss = self.ce(pred, labels.cuda().long()).mean()
            dvb_loss = self.ce(dvb_pred, labels.cuda().long()).mean()
            radi_loss = self.ce(radi_pred, labels.cuda().long()).mean()
            hlq_loss = self.ce(hlq_pred, labels.cuda().long()).mean()

            total_loss = main_loss + (hlq_loss + dvb_loss + radi_loss) / 3
            _total_loss = total_loss.cpu().data.numpy()
            self.loss.append(_total_loss)

            if _total_loss > 1e-9:
                total_loss.backward()
                self.optimizer.step()

            if self.restore_iter % 100 == 0:
                self.save_model()

    def transform(self, subset='test', batch_size=1):
        self.encoder.eval()
        assert subset in ['train', 'val', 'test']
        source = self.test_source
        if subset == 'train':
            source = self.train_source
        elif subset == 'val':
            source = self.val_source
        data_loader = tordata.DataLoader(
            dataset=source,
            batch_size=batch_size,
            sampler=tordata.sampler.SequentialSampler(source),
            num_workers=self.num_workers)

        pred_list = list()
        feature_list = list()
        label_list = list()

        with torch.no_grad():
            for i, x in enumerate(data_loader):
                radi_data, dvb_data, hlq_data, labels = x
                pred, radi_pred, dvb_pred, hlq_pred = self.encoder(
                    radi_data.cuda().float(), dvb_data.cuda().float(),
                    hlq_data.cuda().float())

                pred_list.append(pred.data.cpu().numpy())
                label_list.append(labels.numpy())

        pred_list = np.concatenate(pred_list, 0)
        label_list = np.concatenate(label_list, 0)

        return pred_list, label_list

    def save_model(self):
        torch.save(self.encoder.state_dict(), osp.join(
            'model', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, self.restore_iter)))
        torch.save(self.optimizer.state_dict(), osp.join(
            'model', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, self.restore_iter)))

    def load_model(self, restore_iter=None):
        if restore_iter is None:
            restore_iter = self.restore_iter
        self.encoder.load_state_dict(torch.load(osp.join(
            'model', self.model_name,
            '{}-{:0>5}-encoder.ptm'.format(self.save_name, restore_iter))))
        self.optimizer.load_state_dict(torch.load(osp.join(
            'model', self.model_name,
            '{}-{:0>5}-optimizer.ptm'.format(self.save_name, restore_iter))))

    def load_pretrain(self, pretrain_point):
        self.encoder.load_state_dict(torch.load(osp.join(
            'model', pretrain_point)), False)


def init_model(fold, train_data, train_label, val_data, val_label, test_data, test_label):
    train_source = DataSet(train_data, train_label)
    val_source = DataSet(val_data, val_label)
    test_source = DataSet(test_data, test_label)
    print('train_len:', len(train_source))
    print('test_len:', len(test_source))
    _lr = 1e-4
    print('Initialize lr as %f' % _lr)
    model_config = {
        'dout': True,
        'lr': _lr,
        'num_classes': 2,
        'num_workers': 8,
        'batch_size': 64,
        'restore_iter': 0,
        'total_iter': 5000,
        'model_name': 'MGH-dw-all-' + fold,
        'pretrain_point': None,
        'train_source': train_source,
        'val_source': val_source,
        'test_source': test_source
    }
    model_config['save_name'] = '_'.join([
        '{}'.format(model_config['model_name']),
        '{}'.format(model_config['dout']),
        '{}'.format(0.0001),
        '{}'.format(model_config['batch_size']),
    ])

    os.makedirs(osp.join('model', model_config['model_name']), exist_ok=True)

    return Model(**model_config)
