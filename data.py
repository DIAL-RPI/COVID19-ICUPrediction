# -*- coding: utf-8 -*-
# @Author  : chq_N
# @Time    : 2020/8/01

import numpy as np
import pandas as pd

QUANTIFICATION_PATH = 'HLQ.csv'
RADIOMIC_PATH = 'WLR.csv'
PATIENT_PATH = 'DVB.csv'


def load_data():
    # load patient info & DVB
    patient_data = pd.read_csv(PATIENT_PATH, index_col=0)
    uid_list = patient_data.index.values.tolist()
    _feature_name = ['c_age', 'c_lym_r', 'c_lym', 'c_wbc', 'sex']
    np_data = patient_data.loc[:, _feature_name].values.astype('float')

    dvb_feature = np_data
    dvb_feature_name = _feature_name
    label = patient_data.loc[:, 'label_icu'].values

    # load HLQ
    data = pd.read_csv(QUANTIFICATION_PATH, index_col=0)
    data = data.loc[uid_list]
    hlq_feature_name = [x for x in data.columns.values.tolist() if x not in ['label_icu', 'label_died']]
    hlq_feature = data[hlq_feature_name].values

    hlq_feature = (hlq_feature - hlq_feature.mean(axis=0).reshape(1, -1)) / (
            hlq_feature.std(axis=0).reshape(1, -1) + 1e-9)

    # load WLR
    radiomics = pd.read_csv(RADIOMIC_PATH, index_col=0)
    radiomics = radiomics.loc[uid_list]
    feature_name = radiomics.columns.values.tolist()
    shape_clm = [_ for _ in feature_name if 'shape' in _]
    first_ord_clm = [_ for _ in feature_name if 'firstorder' in _]
    high_level_clm = [_ for _ in feature_name
                      if ('wavelet' in _ or 'log-sigma' in _) and 'firstorder' not in _]
    second_ord_clm = [_ for _ in feature_name
                      if _ not in shape_clm + first_ord_clm + high_level_clm]

    def norm_feature(f):
        return (f - f.mean(axis=0).reshape(1, -1)) / (f.std(axis=0).reshape(1, -1) + 1e-9)

    shape_feature = norm_feature(radiomics.loc[:, shape_clm].values)
    first_ord_feature = norm_feature(radiomics.loc[:, first_ord_clm].values)
    second_ord_feature = norm_feature(radiomics.loc[:, second_ord_clm].values)
    high_level_feature = norm_feature(radiomics.loc[:, high_level_clm].values)

    wlr_feature = np.concatenate([
        shape_feature,
        first_ord_feature,
        second_ord_feature,
        high_level_feature
    ], axis=1)
    wlr_feature_name = shape_clm + first_ord_clm + second_ord_clm + high_level_clm

    return (
        dvb_feature, dvb_feature_name,
        hlq_feature, hlq_feature_name,
        wlr_feature, wlr_feature_name, label)
