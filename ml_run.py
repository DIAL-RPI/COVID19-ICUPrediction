# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/01

import numpy as np

from data import load_data
from feature_rank import sort_feature
from validation import cv_test, detail_test, draw_mean_auc

(dvb_feature, dvb_feature_name,
 hlq_feature, hlq_feature_name,
 wlr_feature, wlr_feature_name, label) = load_data()

_f = np.concatenate([dvb_feature, hlq_feature, wlr_feature], axis=1)
_f_n = dvb_feature_name + hlq_feature_name + wlr_feature_name
_l = (label > 2).astype('int')
indices, s_f = sort_feature(_f, _l, _f_n, True)
b_k, b_auc = cv_test(_f, _l, indices, fn_interval=1)
print('BEST K:', b_k, 'BEST AUC:', b_auc)

tpr, auc_all, auc_mean, auc_std, sen, spe, ppv, npv, acc = detail_test(_f,_l, indices, 12)
draw_mean_auc(tpr, np.mean(sen), np.std(sen, ddof=1),
              np.mean(spe), np.std(spe, ddof=1),
              np.mean(auc_mean), np.std(auc_mean, ddof=1), 'WLR+HLQ+DVB')
print('auc:', np.mean(auc_mean), 'sen:', np.mean(sen), 'spe:', np.mean(spe),
      'ppv:', np.mean(ppv), 'npv:', np.mean(npv), 'acc:', np.mean(acc))
print('auc all', auc_all)
print('auc mean', auc_mean)
print('auc std', auc_std)
print('sen', sen)
print('spe', spe)
print('ppv', ppv)
print('npv', npv)
print('acc', acc)
