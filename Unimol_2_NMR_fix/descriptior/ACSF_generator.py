from dscribe.descriptors import ACSF, SOAP
from ..utils import xyz2mol

class ACSF_Generator(object):
    def __init__(self,**acsf_config):
        self.species = acsf_config.get('species',('H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl'))
        self.rcut = acsf_config.get('rcut',5.0)
        self.g2_params = acsf_config.get('g2_params',None)
        self.g3_params = acsf_config.get('g3_params',None)
        self.g4_params = acsf_config.get('g4_params',None)
        self.g5_params = acsf_config.get('g5_params',None)
        self.periodic = acsf_config.get('periodic',False)
        self.sparse = acsf_config.get('sparse',False)

        self.acsf = ACSF(
                    rcut = self.rcut,
                    g2_params = self.g2_params,
                    g3_params = self.g3_params,
                    g4_params = self.g4_params,
                    g5_params = self.g5_params,
                    species = self.species,
                    sparse = False
                    )
        pass

    def ACSF_atom(self,mol,atom_id):
        acsf_desc = self.ACSF_molecule(mol)
        if len(mol) > 1:
            acsf_descs = []
            for i in range(len(acsf_desc)):
                acsf_descs.append(acsf_desc[i][atom_id[i]-1])
            return acsf_descs
        else: return acsf_desc[0][atom_id]

    def ACSF_molecule(self,mols):
        # 读入单个或多个分子的mol，生成单个或多个分子的ACSF描述符
        acsf_desc = self.acsf.create(mols)
        return acsf_desc

    def ACSF_system(self,):
        pass
