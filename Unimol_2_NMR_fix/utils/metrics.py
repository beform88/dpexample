from tqdm import trange
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import os
import copy
np.set_printoptions(formatter={'float': '{:.3f}'.format})
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    precision_recall_curve,
    r2_score,
    roc_auc_score,
    accuracy_score,
    log_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    average_precision_score,
    recall_score,
    cohen_kappa_score,
)
from scipy.stats import (
    spearmanr,
    pearsonr
)
from .base_logger import logger


def multi_cohen_kappa_score(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()

    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()
    return np.mean([cohen_kappa_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])])

def my_roc_auc_score(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()

    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()

    score = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) == 1:
            # HotFix: ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
            score.append(1.0)
        else:
            score.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return np.mean(score)

def my_average_precision_score(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()

    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()

    score = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) == 1:
            # HotFix: ValueError: Only one class present in y_true. ROC AUC score is not defined in that case.
            score.append(1.0)
        else:
            score.append(average_precision_score(y_true[:, i], y_pred[:, i]))
    return np.mean(score)  

def my_r2_score(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()

    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()

    score = []
    for i in range(y_true.shape[1]):
        score.append(r2_score(y_true[:, i], y_pred[:, i]))
    return np.mean(score)

def mutli_accuracy(y_true, y_pred):
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()

    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()
    y_true = y_true.flatten()
    y_pred_idx = np.argmax(y_pred, axis=1)
    return np.mean(y_true == y_pred_idx)

def mean_absolute_percentage_error(y_true, y_pred):
    """
    计算平均绝对百分比误差（Mean Absolute Percentage Error，MAPE）
    """
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.to_numpy()

    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()

    # 计算百分比误差
    percentage_errors = np.abs((y_true - y_pred) / y_true) * 100

    # 忽略无穷大值和缺失值
    percentage_errors = percentage_errors[~np.isinf(percentage_errors)]
    percentage_errors = percentage_errors[~np.isnan(percentage_errors)]

    # 计算平均绝对百分比误差
    mape = np.mean(percentage_errors)

    return mape


## metric_name, is_increase, value_type
METRICS_REGISTER = {
    'regression': {
        "mae" : [mean_absolute_error, False, 'float'],
        "spearmanr": [lambda y_ture, y_pred: spearmanr(y_ture, y_pred)[0], True, 'float'],
        "mse" : [mean_squared_error, False, 'float'],
        "rmse": [lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), False, 'float'],
        "r2" : [r2_score, True, 'float'],
        "mape": [mean_absolute_percentage_error, False, 'float'],
    },
    'multilabel_regression':{
        "mae" : [mean_absolute_error, False, 'float'],
        "mape": [mean_absolute_percentage_error, False, 'float'],
        "mse": [mean_squared_error, False, 'float'],
        "rmse": [lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)), False, 'float'],
        "r2": [r2_score, True, 'float'],
        "mape": [mean_absolute_percentage_error, False, 'float'],
    }
}

DEFAULT_METRICS = {
    'regression': ['mse', 'mae', 'r2', 'spearmanr'],
    "multilabel_regression": ['mse', 'mae', 'rmse','mape'],
}
class Metrics(object):
    def __init__(self, task = None, metrics_str = None, **params):
        self.task = task
        self.threshold = np.arange(0, 1., 0.1)
        self.metric_dict = self._init_metrics(self.task, metrics_str, **params)
        self.METRICS_REGISTER = METRICS_REGISTER[task]

    def _init_metrics(self, task, metrics_str, **params):
        if task not in METRICS_REGISTER:
            raise ValueError('Unknown task: {}'.format(self.task))
        if not isinstance(metrics_str, str) or metrics_str == '' or metrics_str == 'none':
            metric_dict = {key:METRICS_REGISTER[task][key] for key in DEFAULT_METRICS[task]}
        else:
            for key in metrics_str.split(','):
                if key not in METRICS_REGISTER[task]:
                    raise ValueError('Unknown metric: {}'.format(key))

            priority_metric_list = metrics_str.split(',')
            metric_list = priority_metric_list + [key for key in METRICS_REGISTER[task] if key not in priority_metric_list]
            metric_dict = {key:METRICS_REGISTER[task][key] for key in metric_list}
        
        return metric_dict
    
    def cal_classification_metric(self, label, predict, threshold = None, visualization = False):
        r"""
            :param label:int
            :param predict:float
        """
        res_dict = {}
        for metric_type, metric_value in self.metric_dict.items():
            metric, is_increase, value_type =  metric_value
            if value_type == 'float':
                res_dict[metric_type] = metric(label.astype(int), predict.astype(np.float))
            elif value_type == 'int':
                thre = 0.5 if threshold is None else threshold
                res_dict[metric_type] = metric(label.astype(int), (predict > thre).astype(int))

        ## TO DO : add more metrics by grid search threshold

        if visualization:
            ##TODO 补充画图功能
            pass
        return res_dict

    def cal_reg_metric(self, label, predict, threshold = None, visualization = False):
        r"""
            :param label:int
            :param predict:float
        """
        res_dict = {}
        for metric_type, metric_value in self.metric_dict.items():
            metric, is_increase, value_type =  metric_value
            res_dict[metric_type] = metric(label, predict)

        if visualization:
            ##TODO 补充画图功能
            pass
        return res_dict
    
    def cal_multiclass_metric(self, label, predict, threshold = None, visualization = False):
        r"""
            :param label:int
            :param predict:float
        """
        res_dict = {}
        for metric_type, metric_value in self.metric_dict.items():
            metric, is_increase, value_type =  metric_value
            res_dict[metric_type] = metric(label, predict)

        if visualization:
            ##TODO 补充画图功能
            pass
        return res_dict
       
    def cal_multireg_metric(self, label, predict, threshold = None, visualization = False):
            r"""
                :param label:int
                :param predict:float
            """
            res_dict = {}
            for metric_type, metric_value in self.metric_dict.items():
                metric, is_increase, value_type =  metric_value
                metrics_per_column = []
                for i in range(label.shape[1]):
                    res = metric(label[:, i], predict[:, i])
                    metrics_per_column.append(res)
                res_dict[metric_type] = metrics_per_column

            if visualization:
                ##TODO 补充画图功能
                pass
            return res_dict       

    def cal_metric(self, label, predict, threshold = None, visualization = False):
        if self.task in ['regression']:
            return self.cal_reg_metric(label, predict, threshold, visualization)
        elif self.task in ['classification', 'multilabel_classification']:
            return self.cal_classification_metric(label, predict, threshold, visualization)
        elif self.task in ['multiclass']:
            return self.cal_multiclass_metric(label, predict, threshold, visualization)
        elif self.task in ['multilabel_regression']:
            return self.cal_multireg_metric(label, predict, threshold, visualization)
        else:
            raise ValueError("We will add more tasks soon")

    def _early_stop_choice(self, wait, min_score, metric_score, max_score, model, dump_dir, fold, patience, epoch):
        score = list(metric_score.values())[0]
        judge_metric = list(metric_score.keys())[0]
        is_increase = METRICS_REGISTER[self.task][judge_metric][1]
        if is_increase:
            is_early_stop, max_score, wait = self._judge_early_stop_increase(wait, score, max_score, model, dump_dir, fold, patience, epoch)
        else:
            is_early_stop, min_score, wait = self._judge_early_stop_decrease(wait, score, min_score, model, dump_dir, fold, patience, epoch)
        return is_early_stop, min_score, wait, max_score

    def _judge_early_stop_decrease(self, wait, score, min_score, model, dump_dir, fold, patience, epoch):
        is_early_stop = False
        score = np.mean(score)## multilabel
        if score <= min_score :
            min_score = score
            wait = 0
            info = {'model_state_dict': model.state_dict()}
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(info, os.path.join(dump_dir, f'model_{fold}.pth'))
        elif score >= min_score:
            wait += 1
            if wait == patience:
                logger.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, min_score, wait

    def _judge_early_stop_increase(self, wait, score, max_score, model, dump_dir, fold, patience, epoch):
        is_early_stop = False
        score = np.mean(score) ## multilabel
        if score >= max_score :
            max_score = score
            wait = 0
            info = {'model_state_dict': model.state_dict()}
            os.makedirs(dump_dir, exist_ok=True)
            torch.save(info, os.path.join(dump_dir, f'model_{fold}.pth'))
        elif score <= max_score:
            wait += 1
            if wait == patience:
                logger.warning(f'Early stopping at epoch: {epoch+1}')
                is_early_stop = True
        return is_early_stop, max_score, wait

    def calculate_single_classification_threshold(self, target, pred, metrics_key=None, step=20):
        data = copy.deepcopy(pred)
        range_min = np.min(data).item()
        range_max = np.max(data).item()

        for metric_type, metric_value in self.metric_dict.items():
            metric, is_increase, value_type =  metric_value
            if value_type == 'int':
                metrics_key = metric_value
                break
        # default threshold metrics
        if metrics_key is None:
            metrics_key = METRICS_REGISTER['classification']['f1_score']
        logger.info("metrics for threashod: {0}".format(metrics_key))
        metrics = metrics_key[0]
        if metrics_key[1]:
            # increase metric
            best_metric = float('-inf')
            best_threshold = 0.5
            for threshold in np.linspace(range_min, range_max, step):
                pred_label = np.zeros_like(pred)
                pred_label[pred > threshold] = 1
                # print ("threshold: ", threshold, metric(target, pred_label))
                if metric(target, pred_label) > best_metric:
                    best_metric = metric(target, pred_label)
                    best_threshold = threshold
            logger.info("best threshold: {0}, metrics: {1}".format(best_threshold, best_metric))
        else:
            # increase metric
            best_metric = float('inf')
            best_threshold = 0.5
            for threshold in np.linspace(range_min, range_max, step):
                pred_label = np.zeros_like(pred)
                pred_label[pred > threshold] = 1
                if metric(target, pred_label) < best_metric:
                    best_metric = metric(target, pred_label)
                    best_threshold = threshold
            logger.info("best threshold: {0}, metrics: {1}".format(best_threshold, best_metric))

        return best_threshold

    def calculate_classification_threshold(self, target, pred):
        threshold = np.zeros(target.shape[1])
        for idx in range(target.shape[1]):
            threshold[idx] = self.calculate_single_classification_threshold(target[:, idx].reshape(-1,1), \
                pred[:, idx].reshape(-1, 1), metrics_key=None, step=20)
        return threshold
