import os
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.io import read
from ..utils import xyz2mol
from .ACSF_generator import ACSF_Generator
from .Uni_generator import UniRepr_Generator

class Descriptor_Generator(object):
    def __init__(self,structure,atom,structure_level,structure_source,**kwargs):
        self.structure = structure
        self.atom = atom
        self.structure_level = structure_level
        self.structure_source = structure_source
        self.opt_files = False
        pass
    
    def add_SOAP_repr(self):
        pass

    def add_ACSF_repr(self,):
        if self.structure_level == 'atom':
            mols = []
            atoms_id = []
            for i in range(len(self.structure)):
                atoms_id.append(self.atom[i])

                if self.structure_source == 'files':
                    file_type = os.path.splitext(self.structure[i])[1]
                    if file_type == '.mol':
                        mols.append(Chem.MolFromMolFile(self.structure[i]))
                    elif file_type == '.xyz':
                        mols.append(xyz2mol(self.structure[i]))
                    else:
                        raise ValueError('Unknown file type :' + file_type + '  file must be xyz or mol !')
                    
                elif self.structure_source == 'smiles':
                    mol = Chem.MolFromSmiles(self.structure[i])
                    if mol == None:
                        mol = AllChem.AddHs(mol)
                    # 补充读取smiles转换为mol的部分
                    pass
            uni_desc = ACSF_Generator().ACSF_atom(mols,atoms_id)

        elif self.structure_level == 'molecule':
            mols = []
            for i in range(len(self.structure)):
                if self.structure_source == 'files':
                    file_type = os.path.splitext(self.structure[i])[1]
                    if file_type == '.mol':
                        mols.append(Chem.MolFromMolFile(self.structure[i]))
                    elif file_type == '.xyz':
                        mols.append(xyz2mol(self.structure[i]))
                    else:
                        raise ValueError('Unknown file type :' + file_type + '  file must be xyz or mol !')
                    
                elif self.structure_source == 'smiles':
                    # 补充读取smiles转换为mol的部分
                    pass
            uni_desc = ACSF_Generator().ACSF_molecule(mols)

        elif self.structure_level == 'system':
            pass
        else:
            raise ValueError('Unknown structure :' + file_type + '  file must be xyz or mol !')
        
        return uni_desc

    def add_Unimol_repr(self):
        # return atom level unirepr
        if self.structure_level == 'atom':
            mols = []
            atoms_id = []
            for i in range(len(self.structure)):
                atoms_id.append(self.atom[i])
                
                if self.structure_source == 'files' and self.opt_files == True:
                    file_type = os.path.splitext(self.structure[i])[1]
                    if file_type == '.mol':
                        mols.append(Chem.MolFromMolFile(self.structure[i]))
                    elif file_type == '.xyz':
                        if xyz2mol(self.structure[i]) == None:
                            print(self.structure[i])
                        mols.append(xyz2mol(self.structure[i]))
                    else:
                        raise ValueError('Unknown file type :' + file_type + '  file must be xyz or mol !')
                    
                elif self.structure_source == 'files' and self.opt_files == False:
                    mols.append([read(self.structure[i])])

                    
                elif self.structure_source == 'smiles':
                    # 补充读取smiles转换为mol的部分
                    pass
            uni_atom_desc = UniRepr_Generator().UniRepr_atom(mols,atoms_id)
            return uni_atom_desc
        
        # return molecule level unirepr
        elif self.structure_level == 'molecule':
            mols = []
            for i in range(len(self.structure)):
                if self.structure_source == 'files':
                    file_type = os.path.splitext(self.structure[i])[1]
                    if file_type == '.mol':
                        mols.append(Chem.MolFromMolFile(self.structure[i]))
                    elif file_type == '.xyz':
                        mols.append(xyz2mol(self.structure[i]))
                    else:
                        raise ValueError('Unknown file type :' + file_type + '  file must be xyz or mol !')
                    
                elif self.structure_source == 'smiles':
                    # 补充读取smiles转换为mol的部分
                    pass
            uni_molecule_desc = UniRepr_Generator().UniRepr_atom(mols, atom_repr = False)
            return uni_molecule_desc

        elif self.structure_level == 'system':
            pass

        else:
            raise ValueError('Unknown structure :' + file_type + '  file must be xyz or mol !')
        