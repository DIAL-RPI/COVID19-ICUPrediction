# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/01

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold


def cv_test(features, label, indices, fn_interval=5, num_K=100):
    def cross_val(X, y, seed):
        np.random.seed(seed)
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        prob_list = list()
        auc_list = list()
        y_list = list()
        for train_index, test_index in kf.split(X, y):
            model = RandomForestClassifier(
                n_estimators=300, criterion='gini',
                random_state=seed + 5, max_features='auto')
            model.fit(X[train_index], y[train_index])
            y_pred = model.predict_proba(X[test_index])[:, 1]
            test_auc = roc_auc_score(y[test_index], y_pred)
            prob_list.append(y_pred)
            auc_list.append(test_auc)
            y_list.append(y[test_index])
        return np.concatenate(prob_list), np.concatenate(y_list), np.mean(auc_list)

    best_k = 0
    best_auc = 0

    for ii in np.arange(num_K) + 1:
        if (ii - 1) * fn_interval >= features.shape[-1]:
            break
        f_n = np.clip(ii * fn_interval, 0, features.shape[-1])
        X_selected = features[:, indices[0:f_n]]
        auc_all = list()
        auc_mean = list()
        for i in range(5):
            y_pred, _y, auc = cross_val(X_selected, label, i * 10)
            test_auc = roc_auc_score(_y, y_pred)
            auc_all.append(test_auc)
            auc_mean.append(auc)
        auc_all = np.mean(auc_all)
        auc_mean = np.mean(auc_mean)
        print(f_n, 'auc all:', auc_all, 'auc mean:', auc_mean)
        if auc_mean > best_auc:
            best_auc = auc_mean
            best_k = f_n
    return best_k, best_auc


def detail_test(features, label, indices, f_n, ppv_th=0.7):
    def cross_val(X, y, seed):
        np.random.seed(seed)
        kf = StratifiedKFold(n_splits=5, shuffle=True)
        prob_list = list()
        auc_list = list()
        y_list = list()
        for train_index, test_index in kf.split(X, y):
            model = RandomForestClassifier(
                n_estimators=300, criterion='gini',
                random_state=seed + 5, max_features='auto')
            model.fit(X[train_index], y[train_index])
            y_pred = model.predict_proba(X[test_index])[:, 1]
            test_auc = roc_auc_score(y[test_index], y_pred)
            prob_list.append(y_pred)
            auc_list.append(test_auc)
            y_list.append(y[test_index])
        return np.concatenate(prob_list), np.concatenate(y_list), np.mean(auc_list), np.std(auc_list, ddof=1)

    def get_sen_spe(pred, label):
        label = (label > 0).astype('int')

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
            acc = (TP.sum() + TN.sum()) / (TN.sum() + FP.sum() + TP.sum() + FN.sum() + 1e-9)
            if ppv >= ppv_th:
                break
        return sensitivity, specifity, ppv, npv, acc

    def draw_auc(y_pred, y):
        inter_fpr = np.linspace(0, 1, 1000)
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        inter_tpr = np.interp(inter_fpr, fpr, tpr)
        inter_tpr[0] = 0.0
        inter_tpr[-1] = 1.0
        return inter_tpr

    X_selected = features[:, indices[0:f_n]]
    auc_all = list()
    auc_mean = list()
    auc_std = list()
    sensitivity = list()
    specificity = list()
    ppv = list()
    npv = list()
    acc = list()
    tpr = list()
    for i in range(5):
        y_pred, _y, _auc_mean, _auc_std = cross_val(X_selected, label, i * 10)
        test_auc = roc_auc_score(_y, y_pred)
        _tpr = draw_auc(y_pred, _y)
        tpr.append(_tpr)
        auc_all.append(test_auc)
        auc_mean.append(_auc_mean)
        auc_std.append(_auc_std)
        _sen, _spe, _ppv, _npv, _acc = get_sen_spe(y_pred, _y)
        sensitivity.append(_sen)
        specificity.append(_spe)
        ppv.append(_ppv)
        npv.append(_npv)
        acc.append(_acc)
    return tpr, auc_all, auc_mean, auc_std, sensitivity, specificity, ppv, npv, acc


def draw_mean_auc(
        tpr, mean_sen, std_sen,
        mean_spe, std_spe,
        mean_auc, std_auc,
        save_name):
    tpr = np.asarray(tpr)
    fpr = np.linspace(0, 1, 1000)
    fig, ax = plt.subplots()
    ax.patch.set_facecolor('white')
    ax.grid(color='gray', linestyle='-.', linewidth=0.7)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.plot([0, 1], [0, 1],
            linestyle='--', lw=2, color='r',
            label='Chance',
            alpha=.8)

    mean_tpr = np.mean(tpr, axis=0)
    ax.plot(fpr, mean_tpr, color='b',
            label=r'ROC (AUC = %0.2f$\pm$%0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tpr, axis=0, ddof=1)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.'
                    )

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title='ROC Curve of %s' % save_name
           )

    ax.errorbar(1 - mean_spe, mean_sen, xerr=std_spe, yerr=std_sen,
                color='g', fmt='.', markersize='7', ecolor='red', elinewidth=2, capsize=4,
                label='Point with PPV=0.7')
    ax.annotate('Sen=%0.1f%%$\pm$%0.2f%%\nSpe=%0.1f%%$\pm$%0.2f%%' % (
        round(mean_sen * 100, 1), round(std_sen * 100, 2),
        round(mean_spe * 100, 1), round(std_spe * 100, 2)),
                (1 - mean_spe + 0.02, mean_sen - std_sen - 0.1))
    ax.legend(loc="lower right")
    plt.savefig(save_name + '.pdf')
    plt.show()
