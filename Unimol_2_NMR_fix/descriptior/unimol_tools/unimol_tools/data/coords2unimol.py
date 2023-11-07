from __future__ import absolute_import, division, print_function
import ase
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
import warnings
from scipy.spatial import distance_matrix
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings(action='ignore')
from unicore.data import Dictionary
from multiprocessing import Pool
from tqdm import tqdm
import pathlib
from ..utils import logger
from ..config import MODEL_CONFIG

WEIGHT_DIR = os.path.join(pathlib.Path(__file__).resolve().parents[1], 'weights')

class Coords2Unimol(object):
    def __init__(self,**params):
        self.seed = params.get('seed', 42)
        self.max_atoms = params.get('max_atoms', 256)
        self.data_type = params.get('data_type', 'molecule')
        self.method = params.get('method', 'rdkit_random')
        self.mode = params.get('mode', 'fast')
        self.remove_hs = params.get('remove_hs', False)
        self.base_type = params.get('base_type','smiles')
        self.no_optimize = params.get('no_optimize',False)
        if self.data_type == 'molecule':
            name = "no_h" if self.remove_hs else "all_h" 
            name = self.data_type + '_' + name
            self.dict_name = MODEL_CONFIG['dict'][name]
        else:
            self.dict_name = MODEL_CONFIG['dict'][self.data_type]
        self.dictionary = Dictionary.load(os.path.join(WEIGHT_DIR, self.dict_name))
        self.dictionary.add_symbol("[MASK]", is_special=True)
        pass

    def get(self, atoms, coordinates, max_atoms=256, **params):

        atoms = np.array(atoms)
        coordinates = np.array(coordinates).astype(np.float32)
        ### cropping atoms and coordinates
        if len(atoms)>max_atoms:
            idx = np.random.choice(len(atoms), max_atoms, replace=False)
            atoms = atoms[idx]
            coordinates = coordinates[idx]
        ### tokens padding
        src_tokens = np.array([self.dictionary.bos()] + [self.dictionary.index(atom) for atom in atoms] + [self.dictionary.eos()])
        src_distance = np.zeros((len(src_tokens), len(src_tokens)))
        ### coordinates normalize & padding
        src_coord = coordinates - coordinates.mean(axis=0)
        src_coord = np.concatenate([np.zeros((1,3)), src_coord, np.zeros((1,3))], axis=0)
        ### distance matrix
        src_distance = distance_matrix(src_coord, src_coord)
        ### edge type 
        src_edge_type = src_tokens.reshape(-1, 1) * len(self.dictionary) + src_tokens.reshape(1, -1)

        return {
                'src_tokens': src_tokens.astype(int), 
                'src_distance': src_distance.astype(np.float32), 
                'src_coord': src_coord.astype(np.float32), 
                'src_edge_type': src_edge_type.astype(int),
                }