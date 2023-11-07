import os
from rdkit import Chem
from rdkit.Chem import AllChem
from ase.io import read
from ..utils import xyz2mol
from .Uni_generator import UniRepr_Generator

REPR_GENERATORS = {
    'unimol': UniRepr_Generator,
    }


class Descriptor_Generator2Fintune(object):
    # use to generate descs in fintune process
    def __init__(self,**kwargs):
        self.desc_level = kwargs.get('structure_level','atom') # use 'atom' to get desc for atom_level task, or 'molecule' 
        self.atom_repr = kwargs.get('atom_repr',True) # choose 'all' to get both atom_level and molecule_level descs, or use 'base' to get one  
        self.desc_list = kwargs.get('desc', None)
        self.generators = []
        
        if 'unimol' in self.desc_list or 'Unimol' in self.desc_list:
            self.__init_unimol_generators__()
        if 'other_desc' in  self.desc_list:
            pass

    def __init_unimol_generators__(self):
        self.unirepr_Gen = UniRepr_Generator(finetune = True)
        self.generators.append(self.unirepr_Gen)
        pass

    def generate(self, structure):
        desc = []
        for generator in self.generators:
            desc.append(generator.get2fintune(structure,self.desc_level,self.atom_repr))

        return desc
    
    def finetune_models(self):
        models = {}
        for generator in self.generators:
            model = generator.get_models()
            models[model[0]] = model[1]
        return models
