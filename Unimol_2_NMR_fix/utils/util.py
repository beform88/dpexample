from hashlib import md5
import numpy as np
# from openbabel import openbabel
from rdkit import Chem
from ase.io import read, write
import os

SEP = '_TAB_'


def hash_model_name(model_name, params, idx=8):
    model_str = ''
    for key, value in sorted(params.items()):
        if key == 'active':
            continue
        model_str += str(key) + str(value)
    model_str = model_name + '_TAB_' + \
        md5(model_str.encode('utf-8')).hexdigest()[:idx]
    return model_str


def model_name_generation(model_id, model_name, feature_name, task, joiner="_TAB_"):
    return joiner.join([model_id, model_name, feature_name, task])

def recursive_search(key, dictionary):
    if key in dictionary:
        return True
    for value in dictionary.values():
        if isinstance(value, dict):
            if recursive_search(key, value):
                return True
    return False

def recursive_update(key, value, dictionary):
    if key in dictionary:
        dictionary[key] = value
        return
    for sub_dict in dictionary.values():
        if isinstance(sub_dict, dict):
            recursive_update(key, value, sub_dict)

def xyz2mol(xyz = '',save_path=None, method = 'opb'):
    
    if method == 'opb':
        obConversion = openbabel.OBConversion()

        # Read in XYZ file
        obConversion.SetInFormat("xyz")
        mol = openbabel.OBMol()
        obConversion.ReadFile(mol, xyz)

        # Write out MOL file
        obConversion.SetOutFormat("mol")
        if save_path == None:
            save_path = os.path.splitext(xyz)[0]+'.mol'
            obConversion.WriteFile(mol, save_path)
            mol = Chem.MolFromMolFile(save_path, removeHs=False)
            # mol = read(save_path)
            os.remove(save_path)
        else:
            obConversion.WriteFile(mol, save_path)
            mol = Chem.MolFromMolFile(save_path, removeHs=False)
            # mol = read(save_path)

    elif method == 'ase':
        molecules = read(xyz)
        if save_path == None:
            save_path = 'temp_mol.mol'
            write(save_path, molecules)
            mol = Chem.MolFromMolFile(save_path, removeHs=False)
            os.remove(save_path)
        else:
            write(save_path, molecules)
            mol = Chem.MolFromMolFile(save_path, removeHs=False)

    return mol
