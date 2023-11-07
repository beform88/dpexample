import ase
import os
import pandas as pd
import numpy as np
from ..descriptior import Coords2Unimol
from ..descriptior.unimol_tools.unimol_tools.utils import pad_1d_tokens, pad_2d, pad_coords
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import torch

class Mol2Input(object):
    def __init__(self,**kwargs):
        self.desc = kwargs.get('desc',[])
        self.finetune = kwargs.get('finetune',False)
        self.input_list = []
        self.return_dict = {}

        if ('unimol' in self.desc) or ('Unimol' in self.desc):
            self.__init_unimol_func__()

        if 'new_desc' in self.desc:
            pass

    def __init_unimol_func__(self,):
        self.coords2unimol_input_func = Coords2Unimol()
        self.input_list.append('unimol')

    def mol2_inputs(self,mols):

        if 'unimol' in self.input_list:
            self.return_dict['unimol'] = self.mol2unimol_inputs(mols)

        return self.return_dict


    def mol2unimol_inputs(self,mols):
        # unimol_inputs = {}

        if self.finetune:
            unimol_inputs = []
            for mol in mols:
                atoms = mol.get_chemical_symbols()
                coordinates = mol.get_positions()
                unimol_input = self.coords2unimol_input_func.get(atoms, coordinates)

                for k in unimol_input.keys():
                    if k == 'src_coord':
                        unimol_input[k] = torch.tensor([unimol_input[k]]).float()
                    elif k == 'src_edge_type':
                        unimol_input[k] = torch.tensor([unimol_input[k]]).long()
                    elif k == 'src_distance':
                        unimol_input[k] = torch.tensor([unimol_input[k]]).float()
                    elif k == 'src_tokens':
                        unimol_input[k] = torch.tensor([unimol_input[k]]).long()

                unimol_inputs.append(unimol_input)
        else:
            unimol_inputs = {}
            for mol in mols:
                atoms = mol.get_chemical_symbols()
                coordinates = mol.get_positions()
                unimol_input = self.coords2unimol_input_func.get(atoms, coordinates)
             
                if unimol_inputs == {}:
                    for k in unimol_input.keys():
                        if k == 'src_coord':
                            unimol_inputs[k] = [torch.tensor(unimol_input[k]).float()]
                        elif k == 'src_edge_type':
                            unimol_inputs[k] = [torch.tensor(unimol_input[k]).long()]
                        elif k == 'src_distance':
                            unimol_inputs[k] = [torch.tensor(unimol_input[k]).float()]
                        elif k == 'src_tokens':
                            unimol_inputs[k] = [torch.tensor(unimol_input[k]).long()]
                else:
                    for k in unimol_input.keys():
                        if k == 'src_coord':
                            unimol_inputs[k].append(torch.tensor(unimol_input[k]).float())
                        elif k == 'src_edge_type':
                            unimol_inputs[k].append(torch.tensor(unimol_input[k]).long())
                        elif k == 'src_distance':
                            unimol_inputs[k].append(torch.tensor(unimol_input[k]).float())
                        elif k == 'src_tokens':
                            unimol_inputs[k].append(torch.tensor(unimol_input[k]).long())

            for k in unimol_inputs.keys():
                if k == 'src_coord':
                    unimol_inputs[k] = pad_coords(unimol_inputs[k], pad_idx=0.0)
                elif k == 'src_edge_type':
                    unimol_inputs[k] = pad_2d(unimol_inputs[k], pad_idx=0)
                elif k == 'src_distance':
                    unimol_inputs[k] = pad_2d(unimol_inputs[k], pad_idx=0.0)
                elif k == 'src_tokens':
                    unimol_inputs[k] = pad_1d_tokens(unimol_inputs[k], pad_idx=0)

        return unimol_inputs




        

