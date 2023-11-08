import os
import pandas as pd
import numpy as np
from rdkit import Chem
import pickle
from ..descriptior import ACSF_Generator,Descriptor_Generator,Coords2Unimol
from .molreadkits import Mol2Input
from ..utils import xyz2mol
import ase
from ase.io import read 
from rdkit import Chem
from rdkit.Chem import AllChem

class DataReader(object):
    def __init__(self,**kwargs):
        self.data_path = kwargs.get('data_path','data.csv')
        self.file_name = os.path.splitext(os.path.basename(self.data_path))[0]
        self.task = kwargs.get('task','all')
        self.if_process = kwargs.get('if_process',True)
        self.if_train = kwargs.get('train',True)
        
        # desc
        self.structure_level = kwargs.get('structure_level','atom')
        self.structure_source = kwargs.get('structure_source','files')
        self.opt_files = False
        self.desc = kwargs.get('desc',[])
        self.finetune = kwargs.get('finetune',False)

        # cols 
        self.structure_cols = kwargs.get('structure_cols',None)
        self.structure_atom = kwargs.get('structure_atom',None)
        self.feature_cols = kwargs.get('feature_cols',None)
        self.label_cols = kwargs.get('label_cols',['label',])
        self.drop_cols = kwargs.get('drop_cols',['Unnamed: 0',])

        self.save_dir = kwargs.get('save_dir','.')
        self.__init_data__(**kwargs)

    def __init_data__(self,**kwargs):

        print('Init_data in ' + kwargs['data_path'])

        # 数据预处理
        if self.if_process:
            print('Processing Data ...')
            self.process_data()

        # 数据读取
        print('Loading Data ...')
        self.load_data()
        return
    
    def process_data(self,**kwargs):

        df = pd.read_csv(self.data_path)
        # descriptior setting
        if self.structure_cols == None:
            self.structure = None
            self.atom = None
        elif self.structure_level == 'atom' and self.structure_atom != None:
            self.structure = df[self.structure_cols]
            self.atom = df[self.structure_atom]
        else:
            self.structure = df[self.structure_cols]
            self.atom = None

        # get labels
        self.labels = df[self.label_cols].values

        # update drop cols
        self.get_drop_cols()
        
        # get feature cols
        if self.feature_cols == None:
            self.features_cols = [cols for cols in df.columns if cols not in self.drop_cols]
            self.features = df[self.features_cols].values
        else:
            self.features = df[self.feature_cols].values

        # task data
        if self.structure_source == 'smiles':
            mask = self.structure.apply(lambda smi: self.check_smiles(smi, self.if_train))
            self.structure = self.structure[mask]
            self.features = self.features[mask]
            self.labels = self.labels.reshape(-1,1)
            self.labels = self.labels[mask]

        self.data = {
            'structure':self.structure,
            'atom':self.atom,
            'features':self.features,
            'labels':self.labels
        }
        if self.structure_cols != None:
            # make structure to mol
            self.standard_structures2mol()

            # make mols into finetune models inputshape
            self.mol2inputs()

            # add descriptior without fintune
            self.add_desc()

        # save processed data to local
        self.save_csv2pkl(self.data)

    def load_data(self,**kwargs):
        if not os.path.exists(os.path.join(self.save_dir,self.file_name)+'.pkl'):
            print('Error: cant load data from ' + os.path.join(self.save_dir,self.file_name)+'.pkl' + ', path is not exist')
            return False

        with open(os.path.join(self.save_dir,self.file_name)+'.pkl', 'rb') as f:
            self.data = pickle.load(f)
    
    def standard_structures2mol(self,):
        mols = []
        for i in range(len(self.structure)):     
            if self.structure_source == 'files' and self.opt_files == True:
                file_type = os.path.splitext(self.structure[i])[1]
                if file_type == '.mol':
                    mol =  Chem.MolFromMolFile(self.structure[i])
                elif file_type == '.xyz':
                    if xyz2mol(self.structure[i]) == None:
                        print(self.structure[i])
                    mol = xyz2mol(self.structure[i])
                else:
                    raise ValueError('Unknown file type :' + file_type + '  file must be xyz or mol !')
                
            elif self.structure_source == 'files' and self.opt_files == False:
                mol = read(self.structure[i])

            mols.append(mol)

        self.data['structure'] = mols

    def mol2inputs(self):
        mols = self.data['structure']
        mol2input_func = Mol2Input(desc = self.desc, finetune = self.finetune)
        return_inputs = mol2input_func.mol2_inputs(mols)

        for desc in return_inputs.keys():
            self.data[desc] = return_inputs[desc]


    def add_desc(self,):
        
        if self.desc == []:
            return

        for str in self.desc:
            if str not in ['acsf','ACSF','SOAP','soap','Unimol','unimol']:
                raise ValueError('Unknown descriptor name: ' + str + ' . Please use ACSF SOAP or Unimol') 
        
        self.add_descs = []
        des_gen = Descriptor_Generator(self.data,self.structure_level,self.structure_source)
        
        if 'acsf' in self.desc or 'ACSF' in self.desc:
            # self.add_acsf_repr()
            self.add_descs.append(des_gen.add_ACSF_repr())

        if 'soap' in self.desc or 'SOAP' in self.desc:
            self.add_descs.append(des_gen.add_SOAP_repr())
        
        if ('unimol' in self.desc or 'Unimol' in self.desc) and self.finetune == False:
            self.add_descs.append(des_gen.add_Unimol_repr())

        for desc in self.add_descs:
            if self.task == 'all':
                self.data['features'] = np.concatenate((self.data['features'],np.asarray(desc)),axis=1) 
            elif self.task == 'raw':
                self.data['features'] = self.data['features']
            elif self.task == 'desc':
                self.data['features'] = np.asarray(desc)
            # self.data['features'] = self.data['features']
    
    def save_csv2pkl(self, dict):
        
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        with open(os.path.join(self.save_dir,self.file_name)+'.pkl', 'wb') as f:
            pickle.dump(dict, f)

    def get_drop_cols(self,):
        self.drop_cols.append(self.structure_cols)
        self.drop_cols.append(self.structure_atom)
        self.drop_cols.append('Unnamed: 0')
        for col in self.label_cols:
            self.drop_cols.append(col)

    def check_smiles(self,smi, is_train, smi_strict = False):
        if Chem.MolFromSmiles(smi) is None:
            if is_train and not smi_strict:
                print(f'Illegal SMILES clean: {smi}')
                return False
            else:
                raise ValueError(f'SMILES rule is illegal: {smi}')
        return True
