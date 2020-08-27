# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/26


from datetime import datetime

import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

from model import init_model


def cross_val(X, y, seed):
    np.random.seed(seed)
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    prob_list = list()
    auc_list = list()
    y_list = list()
    index_group = []
    for train_index, test_index in kf.split(X, y):
        index_group.append(test_index.tolist())

    for i in range(len(index_group)):
        fold = str(i) + '-5'
        test_idx = index_group[i]
        _val_i = i + 1
        if _val_i >= len(index_group): _val_i = 0
        val_idx = index_group[_val_i]
        _train_i = [_ for _ in range(len(index_group)) if _ != i and _ != _val_i]
        train_idx = []
        for _ in _train_i:
            train_idx += index_group[_]

        m = init_model(
            fold, X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx], )
        print('Training model:', fold)
        _time = datetime.now()
        m.fit()
        print('Training Time:', datetime.now() - _time)
        print('Testing model:', fold)

        test_corr = 0
        val_max = 0
        val_i = -1
        for j in range(500, 5001, 100):
            m.load_model(j)
            pred, label = m.transform('test', 24)
            pred = softmax(pred, axis=1)
            test_auc = roc_auc_score(label, pred[:, 1])
            pred, label = m.transform('val', 24)
            pred = softmax(pred, axis=1)
            val_auc = roc_auc_score(label, pred[:, 1])
            if val_auc > val_max:
                val_i = j
                val_max = val_auc
                test_corr = test_auc

        print('Best iter:', val_i, 'Best V auc:', val_max, 'Corr T auc:', test_corr)
        m.load_model(val_i)
        pred, label = m.transform('test', 24)
        pred = softmax(pred, axis=1)[:, 1]
        test_auc = roc_auc_score(label, pred)
        prob_list.append(pred)
        auc_list.append(test_auc)
        y_list.append(label)
    return np.concatenate(prob_list), np.concatenate(y_list), np.mean(auc_list), np.std(auc_list, ddof=1)


def detail_test(features, label, ppv_th=0.7):
    def get_sen_spe(pred, label):

        def criteria(x, th):
            return (x > th).astype('int')

        for j in range(0, 1000, 1):
            j = j / 1000
            TP = ((label == 1) * (criteria(pred, j) == 1))
            TN = ((label == 0) * (criteria(pred, j) == 0))
            FP = ((label == 0) * (criteria(pred, j) == 1))
            FN = ((label == 1) * (criteria(pred, j) == 0))
            sensitivity = TP.sum() / (TP.sum() + FN.sum() + 1e-9)
            specifity = TN.sum() / (TN.sum() + FP.sum() + 1e-9)
            ppv = TP.sum() / (TP.sum() + FP.sum() + 1e-9)
            npv = TN.sum() / (TN.sum() + FN.sum() + 1e-9)
            if ppv >= ppv_th:
                break
        return sensitivity, specifity, ppv, npv

    X_selected = features
    auc_all = list()
    auc_mean = list()
    auc_std = list()
    sensitivity = list()
    specificity = list()
    ppv = list()
    npv = list()
    tpr = list()
    for i in range(5):
        y_pred, _y, _auc_mean, _auc_std = cross_val(X_selected, label, i * 10)
        test_auc = roc_auc_score(_y, y_pred)
        _tpr = inter_auc(y_pred, _y)
        tpr.append(_tpr)
        auc_all.append(test_auc)
        auc_mean.append(_auc_mean)
        auc_std.append(_auc_std)
        _sen, _spe, _ppv, _npv = get_sen_spe(y_pred, _y)
        sensitivity.append(_sen)
        specificity.append(_spe)
        ppv.append(_ppv)
        npv.append(_npv)
    return tpr, auc_all, auc_mean, auc_std, sensitivity, specificity, ppv, npv


def inter_auc(y_pred, y):
    inter_fpr = np.linspace(0, 1, 1000)
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    inter_tpr = np.interp(inter_fpr, fpr, tpr)
    inter_tpr[0] = 0.0
    inter_tpr[-1] = 1.0
    return inter_tpr
