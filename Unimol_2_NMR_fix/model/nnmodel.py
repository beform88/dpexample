import os
import numpy as np
import torch
import torch.nn as nn
from ..task import Trainer
from ..data import DataHub
from .MLP import MLPModel
from .nmr_re import NMRREModel
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from ..utils import Metrics,logger
import joblib

NNMODEL_REGISTER = {
    'MLPModel': MLPModel,
    'NMRREModel':NMRREModel,
}

LOSS_RREGISTER = {
    'p2p': nn.MSELoss(),
    's2s': None,
    'i2i': None
}


class NNModel(object):
    def __init__(self,trainer,datahub,**kwargs):
        self.trainer = trainer
        self.task_type = kwargs.get('task_type','p2p')
        self.dump_dir = kwargs.get('dump_dir','./results')

        # dataset param
        self.data = datahub.data
        self.train_size = kwargs.get('train_size',0.875)
        self.random_state = kwargs.get('random_state',42)

        # init model
        self.model_name = kwargs.get('model_name','MLPModel')
        self.input_dim = self.data['features'][0].shape[0]
        self.output_dim = self.data['labels'][0].shape[0]
        self.model = self._init_model(**kwargs)

        # init loss_func
        self.loss_func = kwargs.get('loss_func', self.__init_loss_func())

        # init target_scaler
        self.target_scaler = datahub.data['target_scaler']

        # init metrics
        self.cv = dict()
        self.metrics = self.trainer.metrics

    def _init_model(self, **kwargs):
        if self.model_name in NNMODEL_REGISTER:
            model = NNMODEL_REGISTER[self.model_name](input_dim=self.input_dim,output_dim=self.output_dim)
        elif self.model_name == 'mymodel':
            # 保留接口，接入自己的模型
            model = kwargs['mymodel']
        else:
            raise ValueError('Unknown model: {}'.format(self.model_name))
        return model
    
    def __init_loss_func(self,**kwargs):
        loss_func = LOSS_RREGISTER[self.task_type]
        return loss_func

    def run(self,):
        x = torch.tensor(np.asarray(self.data['features']))
        y = torch.tensor(np.float32(np.asarray(self.data['labels'])))

        x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=self.train_size,random_state=self.random_state)

        train_dataset = TorchDataset(x_train,y_train)
        valid_dataset = TorchDataset(x_test,y_test)
        
        try:
            y_pred = self.trainer.fit_predict(train_dataset, valid_dataset, self.model, self.loss_func, self.dump_dir, self.target_scaler)
        except:
            print("NNModel {0} failed...".format(self.model_name))
            return
        
        print("result {0}".format(self.metrics.cal_metric(self.target_scaler.inverse_transform(y_test), self.target_scaler.inverse_transform(y_pred))))

        self.cv['pred'] = y_pred
        self.cv['metric'] = self.metrics.cal_metric(self.target_scaler.inverse_transform(y_test), self.target_scaler.inverse_transform(y_pred))
        self.dump(self.cv['pred'], self.dump_dir, 'cv.data')
        self.dump(self.cv['metric'], self.dump_dir, 'metric.result')
        
        logger.info("{} NN model metrics score: \n{}".format(self.model_name, self.cv['metric']))
        logger.info("{} NN model done!".format(self.model_name))
        logger.info("Metric result saved!")
        logger.info("{} Model saved!".format(self.model_name))

    def dump(self, data, dir, name):
        path = os.path.join(dir, name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        joblib.dump(data, path)
    

class TorchDataset(Dataset):
    def __init__(self, data, label=None):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)