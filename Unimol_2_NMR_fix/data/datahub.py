import os
import torch
import numpy as np
import pandas as pd
from .datareader import DataReader
from ..descriptior.descriptior_generator import Descriptor_Generator
from .datascaler import TargetScaler
from sklearn.model_selection import train_test_split

class DataHub(object):

    def __init__(self,**kwargs):
        self.data_path = kwargs.get('data_path','data.csv')
        self.file_name = os.path.splitext(os.path.basename(self.data_path))[0]
        # self.task = kwargs.get('task','all')
        self.save_dir = kwargs.get('save_dir','.')
        self.dump_dir = kwargs.get('dump_dir','.')
        self.is_train = kwargs.get('train',True)
        
        self.if_process = self.if_processed()
        self.if_process = kwargs.get('if_process',self.if_process)
        kwargs['if_process'] = self.if_process

        self.structure_level = kwargs.get('structure_level','atom')
        self.structure_source = kwargs.get('structure_source','files')
        self.desc = kwargs.get('desc',[])

        self.datareader = DataReader(**kwargs)
        self.data = self.datareader.data

        self.save_processed_data2csv = kwargs.get('save_processed_data2csv',False)
        if self.save_processed_data2csv:
            self.data2csv()

        self.__init_data(**kwargs)

    def __init_data(self,**kwargs):
        self.DL_task = 'multilabel_regression' if self.data['labels'].shape[1] > 1 else 'regression'
        self.ss_method = kwargs.get('ss_method','none')
        self.data['target_scaler'] = TargetScaler(ss_method=self.ss_method, task=self.DL_task)
        
        if self.DL_task == 'regression' or self.DL_task == 'multilabel_regression':
            # 转为tensor
            self.data['features'] = torch.tensor(np.asarray(self.data['features']))
            self.data['labels'] = torch.tensor(np.float32(np.asarray(self.data['labels'])))

            if self.is_train:
                self.data['target_scaler'].fit(self.data['labels'],self.dump_dir) 
            self.data['labels'] = self.data['target_scaler'].transform(self.data['labels'])
        else:
            raise ValueError('Unknown task: {}'.format(self.DL_task))
        
    def if_processed(self):
        if os.path.exists(os.path.join(self.save_dir,self.file_name)+'.pkl'):
            return False
        else:
            return True

    def data2csv(self):
        x = np.asarray(self.data['features'])
        y = np.float32(np.asarray(self.data['labels']))

        x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=self.train_size,random_state=42)

        train_csv_data = {
            'features':x_train,
            'labels':y_train
        }
        test_csv_data = {
            'features':x_test,
            'labels':y_test
        }

        train_csv = pd.DataFrame(train_csv_data)
        test_csv = pd.DataFrame(test_csv_data)
        train_csv.to_csv(os.path.join(self.save_dir,'train_data.csv'),index=False)
        test_csv.to_csv(os.path.join(self.save_dir,'test_data.csv'),index=False)