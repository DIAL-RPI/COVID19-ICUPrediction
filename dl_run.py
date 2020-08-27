# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/26


import numpy as np

from DL_CV.validation import detail_test
from data import load_data

(dvb_feature, dvb_feature_name,
 hlq_feature, hlq_feature_name,
 wlr_feature, wlr_feature_name, label) = load_data()
_f = np.concatenate([wlr_feature, hlq_feature, dvb_feature], axis=1)
_l = label.astype('int')
tpr, auc_all, auc_mean, auc_std, sen, spe, ppv, npv = detail_test(_f, _l)
print('auc:', np.mean(auc_mean), 'sen:', np.mean(sen), 'spe:', np.mean(spe),
      'ppv:', np.mean(ppv), 'npv:', np.mean(npv))
print('auc all', auc_all)
print('auc mean', auc_mean)
print('auc std', auc_std)
print('sen', sen)
print('spe', spe)
print('ppv', ppv)
print('npv', npv)
