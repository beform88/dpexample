from __future__ import absolute_import, division, print_function
import torch
import logging
import copy
import os
import pandas as pd
import numpy as np
import csv
from typing import List, Optional
import joblib
from sklearn.preprocessing import (
    StandardScaler, 
    MinMaxScaler, 
    MaxAbsScaler, 
    RobustScaler, 
    Normalizer, 
    QuantileTransformer, 
    PowerTransformer, 
    FunctionTransformer,
)
from scipy.stats import skew, kurtosis
from ..utils import logger


SCALER_MODE = {
    'minmax': MinMaxScaler(),
    'standard': StandardScaler(),
    'robust': RobustScaler(),
    'maxabs': MaxAbsScaler(),
    'quantile': QuantileTransformer(),
    'power_box': PowerTransformer(method='box-cox'),
    'power_yeo': PowerTransformer(method='yeo-johnson'),
    'normalizer': Normalizer(),
    'log1p': FunctionTransformer(np.log1p),
}

# NORM_PARA = {
#     "feature_mean":np.load('./electrochemicalmodeling/example/norm/feature_mean.npy'),
#     "feature_std":np.load('./electrochemicalmodeling/example/norm/feature_std.npy'),
#     "label_mean":np.load('./electrochemicalmodeling/example/norm/label_mean.npy'),
#     "label_std":np.load('./electrochemicalmodeling/example/norm/label_std.npy'),
# }

class TargetScaler(object):
    def __init__(self, ss_method, task, load_dir=None):
        self.ss_method = ss_method
        self.task = task
        if load_dir and os.path.exists(os.path.join(load_dir, 'target_scaler.ss')):
            self.scaler = joblib.load(os.path.join(load_dir, 'target_scaler.ss'))
        else:
            self.scaler = None
    
    def transform(self, target):
        if self.ss_method == 'none':
            return target
        elif self.task == 'regression' or self.task == 'multilabel_regression':
            return self.scaler.transform(target)
        else:
            return target
        
    def fit(self, target, dump_dir):
        if self.ss_method == 'none':
            return 
        elif self.ss_method == 'auto':
            if self.task == 'regression' or self.task == 'multilabel_regression':
                if self.is_skewed(target):
                    self.scaler = SCALER_MODE['power_box'] if min(target) > 0 else SCALER_MODE['power_yeo']
                    logger.info('Auto select power transformer.')
                else:
                    self.scaler = SCALER_MODE['standard']
                self.scaler.fit(target)
        else:
            if self.task == 'regression' or self.task == 'multilabel_regression':
                self.scaler = SCALER_MODE[self.ss_method]
                self.scaler.fit(target)
        # save datascaler setting
        try:
            os.remove(os.path.join(dump_dir, 'target_scaler.ss'))
        except:
            pass
        os.makedirs(dump_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(dump_dir, 'target_scaler.ss'))

    def inverse_transform(self, target):
        if self.ss_method == 'none' or self.scaler is None:
            return target
        elif self.task == 'regression' or self.task == 'multilabel_regression':
            return self.scaler.inverse_transform(target)
        else:
            raise ValueError('Unknown scaler method: {}'.format(self.ss_method))
    
    def is_skewed(self, target):
        # 用于检查target的偏度skew和峰度是否在可接受的范围内
        return np.any(abs(skew(target)) > 5.0) or np.any(abs(kurtosis(target)) > 20.0)
        