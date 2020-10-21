
import itertools
import logging
from os import path
from time import strftime, gmtime
import uuid
import os
import copy
import pickle
import typing
import collections

import requests
import numpy as np
import tabulate
import torch
from torch.utils import data
from torch.utils import tensorboard
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from ignite.engine import Events

from autoencoders import logging_tools

from syn_dags.script_utils import train_utils
from syn_dags.data import synthesis_trees
from syn_dags.data import smiles_to_feats
from syn_dags.model import doggen
from syn_dags.model import reaction_predictors
from syn_dags.model import dog_decoder
from syn_dags.script_utils import finetuning_utils
from syn_dags.script_utils import opt_utils

TB_LOGS_FILE = 'tb_logs'
CHKPT_FOLDER = 'chkpts'
TOTAL_LOSS_TB_STRING = "total-loss"


class Params:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.reactant_vocab_path = "../dataset_creation/reactants_to_reactant_id.json"
        self.weight_path = "path to trained weights"
        self.num_dataloader_workers = 4

        time_run = strftime("%y-%m-%d_%H:%M:%S", gmtime())
        f_name_weights = path.splitext(path.basename(self.weight_path))[0]
        self.run_name = f"doggen_sampling_on_weights_{f_name_weights}_run_at_{time_run}"
        print(f"Run name is {self.run_name}\n\n")

        self.batch_size = 200
        self.num_batches = 100


def main(params: Params):
    torch.manual_seed(2562)
    # torch.backends.cudnn.deterministic = True

    # Data
    starting_reactants = train_utils.load_reactant_vocab(params.reactant_vocab_path)
    collate_func = synthesis_trees.CollateWithLargestFirstReordering(starting_reactants, None)
    def __collate_func(*args, **kwargs):
        pred_out_batch, new_order = collate_func(*args, **kwargs)
        return pred_out_batch, None

    # Model components -- set up individual components
    mol_to_graph_idx_for_reactants = collate_func.base_mol_to_idx_dict.copy()
    reactant_graphs = copy.copy(collate_func.reactant_graphs)
    reactant_graphs.inplace_torch_to(params.device)
    reactant_vocab = dog_decoder.DOGGenerator.ReactantVocab(reactant_graphs, mol_to_graph_idx_for_reactants)

    smi2graph_func = lambda smi: smiles_to_feats.DEFAULT_SMILES_FEATURIZER.smi_to_feats(smi)
    reaction_predictor = reaction_predictors.OpenNMTServerPredictor()

    # Add a logger to the reaction predictor so can find out the reactions it predicts later.
    log_hndlr = logging.FileHandler(path.join("logs", f"reactions-{params.run_name}.log"))
    log_hndlr.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_hndlr.setFormatter(formatter)
    os.makedirs(path.join(CHKPT_FOLDER, params.run_name))
    reaction_predictor.logger.addHandler(log_hndlr)
    reaction_predictor.logger.setLevel(logging.DEBUG)
    reaction_predictor.logger.propagate = False

    # Load weights
    chkpt_loaded = torch.load(params.weight_path, map_location=params.device)

    # Model
    model, hparams = doggen.get_dog_gen(reaction_predictor, smi2graph_func,
                                        reactant_vocab, chkpt_loaded['model_params'])
    model = model.to(params.device)
    model.load_state_dict(chkpt_loaded['model'])

    # Sample
    all_syn_trees = []
    all_log_probs = []
    for i in tqdm(range(params.num_batches)):
        syn_trees, log_probs = model.sample(params.batch_size)
        all_syn_trees.extend(syn_trees)
        all_log_probs.append(log_probs.detach().cpu().numpy().T)
    all_log_probs = np.concatenate(all_log_probs)

    with open(f"{params.run_name}.pick", 'wb') as fo:
        pickle.dump(dict(all_log_probs=all_log_probs, all_syn_trees=all_syn_trees), fo)

    smiles_only = [elem.root_smi for elem in all_syn_trees]
    with open(f"{params.run_name}_smiles.txt", 'w') as fo:
        fo.writelines('\n'.join(smiles_only))


if __name__ == '__main__':
    main(Params())
