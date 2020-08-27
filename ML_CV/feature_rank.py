# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/01

import numpy as np
from sklearn.ensemble import RandomForestClassifier


def rf_feature_selection(X, y, seed=41):
    # compute the most signifucant features
    model = RandomForestClassifier(
        n_estimators=300, criterion='gini',
        random_state=seed, max_features='auto')
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    return indices


def sort_feature(X, y, feature_name, print_feature=False):
    rank = np.zeros(len(feature_name)).tolist()
    np.random.seed(0)
    seeds = np.random.randint(10000, size=100).tolist()
    for _s in seeds:
        _ind = rf_feature_selection(X, y, seed=_s)
        for i in range(len(_ind)):
            rank[_ind[i]] += i
    indices = np.argsort(rank).tolist()
    if print_feature:
        for i in range(len(feature_name)):
            print("%2d) %-*s" % (i + 1, 30, feature_name[indices[i]]))
    return indices, [feature_name[_] for _ in indices]
