import os
import numpy as np
from .nnmodel import NNModel

class ModelHub(object):
    def __init__(self,datahub,trainer,**kwargs):
        self.datahub = datahub
        self.trainer = trainer
        self.__init_models(**kwargs)

    def __init_models(self,**kwargs):
        self.__init_nnmodel(**kwargs)
        pass

    def __init_nnmodel(self,**kwargs):
        self.model = NNModel(datahub=self.datahub,trainer=self.trainer,**kwargs)
        pass