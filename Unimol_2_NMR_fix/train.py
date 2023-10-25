import os
import numpy as np
import pandas as pd
import sys
import torch
import cuda
from collections import defaultdict

from .data import DataHub
from .task import Trainer
from .model import ModelHub


class NMR_Fix_Master(object):
    def __init__(self,**kwargs) :
        # self.data_path = kwargs.get('data_path','data.csv')
        # self.task = kwargs.get('task','c')
        # self.save_dir = kwargs.get('save_dir','')
        self.dump_dir = kwargs.get('dump_dir','./results')
        if not os.path.exists(self.dump_dir):
            os.mkdir(self.dump_dir)

        self.datahub_config = kwargs.get('datahub_config',defaultdict(dict))
        self.trainer_config = kwargs.get('trainer_config', defaultdict(dict))
        self.modelhub_config = kwargs.get('modelhub_config', defaultdict(dict))

        self.__init_datahub__()
        self.__init_trainer__()
        self.__init_modelhub__()
        pass

    def __init_datahub__(self):
        self.datahub = DataHub(**self.datahub_config)
        pass

    def __init_trainer__(self):
        self.trainer = Trainer(**self.trainer_config)
        pass

    def __init_modelhub__(self):
        self.modelhub = ModelHub(datahub = self.datahub, trainer= self.trainer, **self.modelhub_config)
        pass

    def update_datahub(self,**kwargs):
        # 手动更新数据相关参数
        self.datahub.structure_level = kwargs.get('structure_level', 'molecule')
        self.datahub.structure_source = kwargs.get('structure_source','files')
        pass
    
    def update_trainer(self,**kwargs):
        # 手动更新训练参数
        self.trainer.max_epoch = kwargs.get('max_epoch',1)
        self.trainer.learning_rate = kwargs.get('lr',1e-4)
        self.trainer.seed = kwargs.get('seed',42)
        self.trainer.set_seed(kwargs.get('seed',42))
        self.trainer.batch_size = kwargs.get('batch_size',32)
        self.trainer.warmup_ratio = kwargs.get('warmup_ratio',0.1)
        self.trainer.patience = kwargs.get('patience', 20)
        self.trainer.max_norm = kwargs.get('max_norm', 1.0)
        self.trainer.cuda = kwargs.get('cuda', True)
        self.trainer.amp = kwargs.get('amp', True)
        self.trainer.device = torch.device("cuda:0" if torch.cuda.is_available() and self.trainer.cuda else "cpu")
        self.trainer.scaler = torch.cuda.amp.GradScaler() if self.trainer.device.type == 'cuda' and self.trainer.amp==True else None
        pass

    def update_modelhub(self,**kwargs):
        # 手动更新模型设置
        self.modelhub.model.train_size = kwargs.get('train_size',0.875)
        self.modelhub.model.task_type = kwargs.get('task_type','p2p')
        self.modelhub.model.model_name = kwargs.get('model_name','MLPModel')
        self.modelhub.model.model.num_layers = kwargs.get('num_layers',2)
        self.modelhub.model.model.hidden_size = kwargs.get('hidden_size',2048)
        pass


    def run(self):
        self.modelhub.model.run()
        pass

# nfm = NMR_Fix_Master(data_path = '/mnt/vepfs/users/ycjin/Delta-ML Framework/Unimol_2_NMR_fix/example/raw_data/ml_pbe0_pcSseg-2_h.csv',
#                task = 'c',
#                save_dir = '/mnt/vepfs/users/ycjin/Delta-ML Framework/Unimol_2_NMR_fix/example/process_data',
#                )
    
# nfm.run()