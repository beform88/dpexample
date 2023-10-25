import sys
import numpy as np 
from .unimol_tools.unimol_tools import UniMolRepr
from ase.io import read
from rdkit import Chem

class UniRepr_Generator(object):
    def __init__(self,base_type = 'mol'):
        self.generator = UniMolRepr(data_type='molecule', remove_hs=False, base_type = base_type, no_optimize = True)
        pass

    def UniRepr_atom(self, mol, atom_id):
        if len(mol) > 1:
            uni_desc = self.generator.get_repr(mol)
            uni_descs = []
            for i in range(len(uni_desc['cls_repr'])):
                uni_descs.append(uni_desc['cls_repr'][i] + uni_desc['atomic_reprs'][i][atom_id[i]-1].tolist())
                # uni_descs.append(uni_desc['atomic_reprs'][i][atom_id[i]-1].tolist())
            return uni_descs
        else: 
            uni_desc = self.generator.get_repr([mol])
            return uni_desc['cls_repr'][0] + uni_desc['atomic_reprs'][0][atom_id[0]-1].tolist()

    def UniRepr_molecule(self, mol, atom_repr = False):
        # if len(mol) > 1:
        #     uni_desc = self.generator.get_repr(mol)
        # else:
        uni_desc = self.generator.get_repr(mol)

        if atom_repr == False:
            return uni_desc['cls_repr']
        else:
            return uni_desc['cls_repr'], uni_desc['atomic_reprs']

from openbabel import openbabel
from rdkit import Chem
from ase.io import read, write
import os
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
# mol = xyz2mol(xyz = '/mnt/vepfs/users/ycjin/Delta-ML Framework/Unimol_2_NMR_fix/example/structures/orca_xyz_formate/001/001_00.xyz')
# mol = Chem.MolFromMolFile('/mnt/vepfs/users/ycjin/Delta-ML Framework/001_00.mol', removeHs=False)
# a = UniRepr_Generator().UniRepr_atom([mol,mol], [1,2])
# a = UniRepr_Generator('smiles').UniRepr_molecule(['CCC'])
# print(a)