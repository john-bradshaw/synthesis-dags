
import typing
import enum
import collections
import sys
from os import path

import numpy as np
import torch
from rdkit.Chem import QED

from guacamol import scoring_function
from guacamol import standard_benchmarks
from guacamol import common_scoring_functions
from guacamol import score_modifier
from guacamol.utils import descriptors
import networkx as nx
from ..chem_ops import rdkit_general_ops
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import rdmolops
from rdkit.Chem import Descriptors
from rdkit.Chem.Descriptors import MolLogP
from rdkit.Chem import RDConfig
sys.path.append(path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


from . import dogae_utils
from ..utils import settings


class PropertyEvaluator:
    def __init__(self, property_calculator, dim=1):
        self.seen_molecules = collections.OrderedDict()
        self.property_calculator = property_calculator
        self.dim = dim

    @property
    def num_evaluated(self):
        return len(self.seen_molecules)

    @property
    def best_seen(self):
        seen_molecule_vals = list(self.seen_molecules.items())
        return max(seen_molecule_vals, key=lambda x: x[1])

    def evaluate_molecules(self, list_of_smiles: typing.List[str]):
        out = []
        for smi in list_of_smiles:
            canon_smi = rdkit_general_ops.canconicalize(smi)
            if canon_smi not in self.seen_molecules:
                value = self.property_calculator(canon_smi)
                self.seen_molecules[canon_smi] = value

            out.append(self.seen_molecules[canon_smi])
        return np.array(out)


def get_qed_evaluator():

    def qed(smi):
        mol = rdkit_general_ops.get_molecule(smi, kekulize=False)
        qed = QED.qed(mol)
        return [qed]

    return PropertyEvaluator(qed)


def get_fingerprint_evaluator(nbits=1000):
    def fp(smi):
        mol = rdkit_general_ops.get_molecule(smi, kekulize=False)
        fp = rdkit_general_ops.get_fingerprint_as_array(mol, 2, nbits)
        return fp
    return PropertyEvaluator(fp, dim=nbits)


def get_sascorer():
    smi2score = lambda smiles: [sascorer.calculateScore(Chem.MolFromSmiles(smiles))]
    return PropertyEvaluator(smi2score)


def get_penalized_logp():
    def reward_penalized_log_p_gcpn(smiles):
        """
        Reward that consists of log p penalized by SA and # long cycles,
        as described in (Kusner et al. 2017). Scores are normalized based on the
        statistics of 250k_rndm_zinc_drugs_clean.smi dataset
        :param mol: rdkit mol object
        :return: float
        """
        mol = Chem.MolFromSmiles(smiles)
        # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
        logP_mean = 2.4570953396190123
        logP_std = 1.434324401111988
        SA_mean = -3.0525811293166134
        SA_std = 0.8335207024513095
        cycle_mean = -0.0485696876403053
        cycle_std = 0.2860212110245455

        log_p = MolLogP(mol)
        SA = -sascorer.calculateScore(mol)

        # cycle score
        cycle_list = nx.cycle_basis(nx.Graph(
            Chem.rdmolops.GetAdjacencyMatrix(mol)))
        if len(cycle_list) == 0:
            cycle_length = 0
        else:
            cycle_length = max([len(j) for j in cycle_list])
        if cycle_length <= 6:
            cycle_length = 0
        else:
            cycle_length = cycle_length - 6
        cycle_score = -cycle_length

        normalized_log_p = (log_p - logP_mean) / logP_std
        normalized_SA = (SA - SA_mean) / SA_std
        normalized_cycle = (cycle_score - cycle_mean) / cycle_std

        return [normalized_log_p + normalized_SA + normalized_cycle]

    return PropertyEvaluator(reward_penalized_log_p_gcpn)


def get_logp_eval():
    smi2score = lambda smiles: [MolLogP(Chem.MolFromSmiles(smiles))]
    return PropertyEvaluator(smi2score)


class GuacTask(enum.Enum):
    ARIPIPRAZOLE = "Aripiprazole similarity"
    OSIMERTINIB = "Osimertinib MPO"
    RANOLAZINE = "Ranolazine MPO"
    ZALEPLON = "Zaleplon MPO"
    VALSARTAN = "Valsartan SMARTS"
    DECO = "decoration hop"
    SCAFFOLD = "scaffold hop"

    PERINDOPRIL = "Perindopril MPO"
    AMLODIPINE = "Amlodipine MPO"
    SITAGLIPTIN = "Sitagliptin MPO"

    CELECOXIB = "Celecoxib rediscovery"
    TROGLITAZONE = "Troglitazone rediscovery"
    THIOTHIXENE = "Thiothixene rediscovery"
    ALBUTEROL = "Albuterol similarity"
    MESTRANOL = "Mestranol similarity"
    FEXOFENADINE = "Fexofenadine MPO"


def get_guac_property_eval(task: GuacTask):
    if task is GuacTask.CELECOXIB:
        bench = standard_benchmarks.similarity(smiles='CC1=CC=C(C=C1)C1=CC(=NN1C1=CC=C(C=C1)S(N)(=O)=O)C(F)(F)F', name='Celecoxib', fp_type='ECFP4',
                   threshold=1.0, rediscovery=True)
    elif task is GuacTask.TROGLITAZONE:
        bench = standard_benchmarks.similarity(smiles='Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O', name='Troglitazone', fp_type='ECFP4',
                   threshold=1.0, rediscovery=True)
    elif task is GuacTask.THIOTHIXENE:
        bench = standard_benchmarks.similarity(smiles='CN(C)S(=O)(=O)c1ccc2Sc3ccccc3C(=CCCN4CCN(C)CC4)c2c1', name='Thiothixene', fp_type='ECFP4',
                   threshold=1.0, rediscovery=True)
    elif task is GuacTask.ARIPIPRAZOLE:
        bench = standard_benchmarks.similarity(smiles='Clc4cccc(N3CCN(CCCCOc2ccc1c(NC(=O)CC1)c2)CC3)c4Cl', name='Aripiprazole', fp_type='ECFP4',
                   threshold=0.75)
    elif task is GuacTask.ALBUTEROL:
        bench = standard_benchmarks.similarity(smiles='CC(C)(C)NCC(O)c1ccc(O)c(CO)c1', name='Albuterol', fp_type='FCFP4', threshold=0.75)
    elif task is GuacTask.MESTRANOL:
        bench = standard_benchmarks.similarity(smiles='COc1ccc2[C@H]3CC[C@@]4(C)[C@@H](CC[C@@]4(O)C#C)[C@@H]3CCc2c1', name='Mestranol',
                   fp_type='AP', threshold=0.75)
    elif task is GuacTask.OSIMERTINIB:
        bench = standard_benchmarks.hard_osimertinib()
    elif task is GuacTask.RANOLAZINE:
        bench = standard_benchmarks.ranolazine_mpo()
    elif task is GuacTask.ZALEPLON:
        bench = standard_benchmarks.zaleplon_with_other_formula()
    elif task is GuacTask.VALSARTAN:
        bench = standard_benchmarks.valsartan_smarts()
    elif task is GuacTask.DECO:
        bench = standard_benchmarks.decoration_hop()
    elif task is GuacTask.SCAFFOLD:
        bench = standard_benchmarks.scaffold_hop()
    elif task is GuacTask.PERINDOPRIL:
        bench = standard_benchmarks.perindopril_rings()
    elif task is GuacTask.AMLODIPINE:
        bench = standard_benchmarks.amlodipine_rings()
    elif task is GuacTask.SITAGLIPTIN:
        bench = standard_benchmarks.sitagliptin_replacement()
    elif task is GuacTask.FEXOFENADINE:
        bench = standard_benchmarks.hard_fexofenadine()
    else:
        raise NotImplementedError

    smi2score = lambda smi: [bench.objective.score(smi)]
    return PropertyEvaluator(smi2score)

