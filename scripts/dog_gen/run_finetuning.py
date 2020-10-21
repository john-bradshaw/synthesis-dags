
import logging
from os import path
from time import strftime, gmtime
import uuid
import os
import copy
import pickle
import csv

import numpy as np
import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self, task: opt_utils.GuacTask):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train_tree_path = "../dataset_creation/data/uspto-train-depth_and_tree_tuples.pick"
        self.valid_tree_path = "../dataset_creation/data/uspto-valid-depth_and_tree_tuples.pick"
        self.reactant_vocab_path = "../dataset_creation/reactants_to_reactant_id.json"

        self.weight_path = "add path to learnt weights here"

        self.num_dataloader_workers = 4


        self.opt_name = task.value
        time_run = strftime("%y-%m-%d_%H:%M:%S", gmtime())
        self.exp_uuid = uuid.uuid4()
        self.run_name = f"dog_gen_finetuning_{time_run}_{self.exp_uuid}_{self.opt_name}"
        print(f"Run name is {self.run_name}\n\n")
        self.property_predictor = opt_utils.get_guac_property_eval(task)


def main(params: Params):
    rng = np.random.RandomState(5156)
    torch.manual_seed(5115)
    # torch.backends.cudnn.deterministic = True

    # Data
    train_trees = train_utils.load_tuple_trees(params.train_tree_path, rng)
    val_trees = train_utils.load_tuple_trees(params.valid_tree_path, rng)
    train_trees = train_trees + val_trees
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

    # TB
    tb_summary_writer = SummaryWriter(log_dir='tb_logs')

    # Setup functions needed for runner
    def loss_fn(model: doggen.DogGen, x, y):
        # Outside the model shall compute the embeddings of the graph -- these are needed in both the encoder
        # and decoder so saves compute to just compute them once.
        embedded_graphs = model.mol_embdr(x.molecular_graphs)
        x.molecular_graph_embeddings = embedded_graphs
        new_node_feats_for_dag = x.molecular_graph_embeddings[x.dags_for_inputs.node_features.squeeze(), :]
        x.dags_for_inputs.node_features = new_node_feats_for_dag

        loss = model(x).mean()
        return loss

    def prepare_batch(batch, device):
        x, y = batch
        x.inplace_to(device)
        return x, y

    def create_dataloader(tuple_trees, batch_size):
        return data.DataLoader(tuple_trees, batch_size=batch_size,
                               num_workers=params.num_dataloader_workers, collate_fn=__collate_func,
                               shuffle=True)

    # Now Setup the parts for runner
    hparams = finetuning_utils.DogGenFTParams(30, 7000, 1500)
    parts = finetuning_utils.DogGenFTParts(model, params.property_predictor, set(starting_reactants), rng,
                                           create_dataloader, prepare_batch, loss_fn, params.device)

    # Now setup
    fine_tuner = finetuning_utils.DogGenFT(parts, hparams)

    # Run
    print("Starting hill climber")
    sorted_tts = fine_tuner.run_finetuning(train_trees, tb_summary_writer)

    # Save the molecules that were queried
    print(f"Best molecule found {params.property_predictor.best_seen}")
    out_data = {'seen_molecules': params.property_predictor.seen_molecules,
                    'sorted_tts': sorted_tts,
                    'opt_name': params.opt_name
        }

    with open(f'results_{params.run_name}.pick', 'wb') as fo:
        pickle.dump(out_data, fo)

    best_molecules = sorted(out_data['seen_molecules'].items(), key=lambda x: x[1], reverse=True)
    smiles_score = [(elem[0], elem[1][0]) for elem in best_molecules]
    with open(f'results_{params.run_name}.tsv', 'w') as fo:
        w = csv.writer(fo, dialect=csv.excel_tab)
        w.writerows(smiles_score[:100])

    print("Done!")
    return out_data


if __name__ == '__main__':
    main(Params(opt_utils.GuacTask.SCAFFOLD))

