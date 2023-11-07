import os
import numpy as np
import torch
import torch.nn as nn
from ..task import Trainer
from ..data import DataHub
from ..descriptior import Descriptor_Generator2Fintune
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
        self.desc_gen2fintune = datahub.desc_gen2fintune
        self.train_size = datahub.clip_size
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
        self.device = self.trainer.device

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

    def run(self,**kwargs):
        x = torch.tensor(np.asarray(self.data['features']))
        y = torch.tensor(np.float32(np.asarray(self.data['labels'])))
        indices = np.arange(x.shape[0])

        indices_train_idx,indices_test_idx,y_train,y_test = train_test_split(indices,y,train_size=self.train_size,random_state=self.random_state)
        x_train = [x[i] for i in indices_train_idx]
        x_test = [x[i] for i in indices_test_idx]

        if self.desc_gen2fintune == None:
            train_dataset = self.NNDataset(data = x_train, label = y_train)
            valid_dataset = self.NNDataset(data = x_test, label = y_test)
            self.finetune_models =None
            y_pred = self.trainer.fit_predict(train_dataset, valid_dataset, self.model, self.loss_func, self.dump_dir, self.target_scaler)

        else:
            train_structure = {}
            test_structure = {}
            for k in self.data.keys():
                if k in ['features','labels','target_scaler']:continue
                train_structure[k] = [self.data[k][i] for i in indices_train_idx]
                test_structure[k] = [self.data[k][i] for i in indices_test_idx]
            
            train_dataset = self.NNDataset(data = x_train, label = y_train, structure = train_structure, 
                                        des_gen = self.desc_gen2fintune, device = self.device)
            valid_dataset = self.NNDataset(data = x_test, label = y_test, structure = test_structure, 
                                        des_gen = self.desc_gen2fintune, device = self.device)
            # update model's input dim
            input_updates_lens = train_dataset.input_dim_lens()
            if input_updates_lens > self.input_dim:
                self.input_dim = input_updates_lens
                self.model = self._init_model(**kwargs)

            # get models 
            self.finetune_models = train_dataset.finetune_models()
            y_pred = self.trainer.fit_predict_finetune(train_dataset, valid_dataset, self.model, self.finetune_models, self.loss_func, self.dump_dir, self.target_scaler)

        # try:
        #     y_pred = self.trainer.fit_predict(train_dataset, valid_dataset, self.model, self.loss_func, self.dump_dir, self.target_scaler)
        # except:
        #     print("NNModel {0} failed...".format(self.model_name))
        #     return
        # y_pred = self.trainer.fit_predict(train_dataset, valid_dataset, self.model, self.finetune_models, self.loss_func, self.dump_dir, self.target_scaler)
        
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
    
    def NNDataset(self, data, structure = None, atom = None, label = None, des_gen = None, device = None):
        if structure != None:
            # return FinetuneDataset(data = data, structure = structure, label = label, des_gen=  des_gen, device=device )
            return FinetuneDataset_test(data = data, structure = structure, label = label, des_gen=  des_gen, device=device )
        else:
            return TorchDataset(data = data, label = label)

class FinetuneDataset(Dataset):
    def __init__(self, data, structure, label = None, des_gen = None, device = None):
        self.structures = structure
        self.desc_generator = des_gen
        self.device = device
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        structure = {}
        for k in self.structures.keys():
            structure[k] = self.structures[k][idx]
        structure_desc = self.desc_generator.generate(structure)
        data = torch.cat([torch.tensor(self.data[idx]).to(self.device),structure_desc[0]],dim = 0)
        return data, self.label[idx]

    def __len__(self):
        return len(self.data)
    
    def input_dim_lens(self):
        structure = {}
        for k in self.structures.keys():
            structure[k] = self.structures[k][0]
        structure_desc = self.desc_generator.generate(structure)
        data = torch.cat([torch.tensor(self.data[0]).to(self.device),structure_desc[0]],dim = 0)
        return data.shape[0]
    
    def finetune_models(self):
        return self.desc_generator.finetune_models()

class FinetuneDataset_test(Dataset):
    def __init__(self, data, structure, label = None, des_gen = None, device = None):
        self.structures = structure
        self.desc_generator = des_gen
        self.device = device
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        structure = {}
        structure['base'] = torch.tensor(self.data[idx])
        for k in self.structures.keys():
            structure[k] = self.structures[k][idx]
        return structure, self.label[idx]

    def __len__(self):
        return len(self.data)
    
    def input_dim_lens(self):
        structure = {}
        for k in self.structures.keys():
            structure[k] = self.structures[k][0]
        structure_desc = self.desc_generator.generate(structure)
        data = torch.cat([torch.tensor(self.data[0]).to(self.device),structure_desc[0]],dim = 0)
        return data.shape[0]
    
    def finetune_models(self):
        return self.desc_generator.finetune_models()
    
class TorchDataset(Dataset):
    def __init__(self, data, label=None):
        self.data = data
        self.label = label if label is not None else np.zeros((len(data), 1))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]
    
    def __len__(self):
        return len(self.data)