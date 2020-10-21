
import itertools
import logging
from os import path
from time import strftime, gmtime
import uuid
import os
import copy
import typing

import requests
import numpy as np
import tabulate
import torch
from torch.utils import data
from torch.utils import tensorboard
from torch.nn import functional as F
from torch import optim
from tqdm import tqdm

from ignite.engine import Events

from autoencoders import logging_tools

from syn_dags.script_utils import train_utils
from syn_dags.data import synthesis_trees
from syn_dags.data import smiles_to_feats
from syn_dags.model import dogae
from syn_dags.model import dog_decoder
from syn_dags.model import reaction_predictors
from syn_dags.utils import ignite_utils
from syn_dags.utils import settings
from syn_dags.utils import misc
from syn_dags.script_utils import tensorboard_helper as tb_
from syn_dags.script_utils import dogae_utils
from syn_dags.script_utils import opt_utils

TB_LOGS_FILE = 'tb_logs'
CHKPT_FOLDER = 'chkpts'
TOTAL_LOSS_TB_STRING = "total-loss"


class Params:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.train_tree_path = "../../dataset_creation/data/uspto-train-depth_and_tree_tuples.pick"
        self.val_tree_path = "../../dataset_creation/data/uspto-valid-depth_and_tree_tuples.pick"
        self.reactant_vocab_path = "../../dataset_creation/reactants_to_reactant_id.json"

        self.num_dataloader_workers = 4
        self.batch_size = 64
        self.val_batch_size = 400
        self.learning_rate = 0.001
        self.log_interval_histograms = 100
        self.num_epochs = 400
        self.gamma = 0.1
        self.milestones = [300, 350]
        self.lambda_value = 10.  # see WAE paper, section 4
        self.expensive_ops_freq = 25


        time_run = strftime("%y-%m-%d_%H:%M:%S", gmtime())
        self.exp_uuid = uuid.uuid4()
        self.run_name = f"train_dogae_run_{time_run}_{self.exp_uuid}"
        print(f"Run name is {self.run_name}\n\n")


def main(params: Params):

    rng = np.random.RandomState(4545)
    torch.manual_seed(2562)
    # torch.backends.cudnn.deterministic = True

    # Data
    train_trees = train_utils.load_tuple_trees(params.train_tree_path, rng)

    val_trees = train_utils.load_tuple_trees(params.val_tree_path, rng)
    print(f"Number of trees in valid set: {len(val_trees)}")

    starting_reactants = train_utils.load_reactant_vocab(params.reactant_vocab_path)

    collate_func = synthesis_trees.CollateWithLargestFirstReordering(starting_reactants, None)
    def __collate_func(*args, **kwargs) -> synthesis_trees.PredOutBatch:
        pred_out_batch, new_order = collate_func(*args, **kwargs)
        return pred_out_batch, None

    train_dataloader = data.DataLoader(train_trees, batch_size=params.batch_size,
                                       num_workers=params.num_dataloader_workers, collate_fn=__collate_func,
                                       shuffle=True)
    val_dataloader = data.DataLoader(val_trees, batch_size=params.val_batch_size, num_workers=params.num_dataloader_workers,
                                     collate_fn=__collate_func, shuffle=False)


    # Model -- set up individual componenets
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

    # Model
    _dogae_params = {'latent_dim': 25,
                     'mol_graph_embedder_params': {'hidden_layer_size': 80,
                                                   'edge_names': ['single', 'double', 'triple', 'aromatic'],
                                                   'embedding_dim': 50,
                                                   'num_layers': 4},
                     'dag_graph_embedder_gnn_params': {'hlayer_size': 50,
                                                       'edge_names': ['reactions'],
                                                       'num_layers': 7},
                     'dag_embedder_aggr_type_s': 'FINAL_NODE',
                     'decoder_params': {'gru_insize': 50,
                                        'gru_hsize': 200,
                                        'num_layers': 3,
                                        'gru_dropout': 0.1,
                                        'max_steps': 100},
                     }

    model, hparams = dogae.get_model(reaction_predictor, smi2graph_func, reactant_vocab, params=_dogae_params)
    model = model.to(params.device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=params.milestones, gamma=params.gamma)

    # Tensorboard:
    # Set up some loggers
    tb_writer_train = tb_.get_tb_writer(f"{TB_LOGS_FILE}/{params.run_name}_train")
    tb_writer_train.global_step = 0
    tb_writer_train.add_hparams({**misc.unpack_class_into_params_dict(hparams, prepender="model:"),
                                **misc.unpack_class_into_params_dict(params, prepender="train:")}, {})
    tb_writer_val = tb_.get_tb_writer(f"{TB_LOGS_FILE}/{params.run_name}_val")
    def add_details_to_train(dict_to_add):
        for name, value in dict_to_add.items():
            tb_writer_train.add_scalar(name, value)
    train_log_helper = logging_tools.LogHelper([add_details_to_train])

    # Create ignite parts
    def loss_fn(model, x, y):
        # Outside the model shall compute the embeddings of the graph -- these are needed in both the encoder
        # and decoder so saves compute to just compute them once.
        embedded_graphs = model.mol_embdr(x.molecular_graphs)
        x.molecular_graph_embeddings = embedded_graphs
        new_node_feats_for_dag = x.molecular_graph_embeddings[x.dags_for_inputs.node_features.squeeze(),:]
        x.dags_for_inputs.node_features = new_node_feats_for_dag

        loss = -model(x, lambda_=params.lambda_value).mean()

        loss_parts = {'ae': loss.item()}

        return loss

    def prepare_batch(batch, device):
        x, y = batch
        x.inplace_to(device)
        return x, y

    def setup_for_val():
        tb_writer_val.global_step = tb_writer_train.global_step  # match the steps
        model._logger_manager = None  # turn off the more precise logging for when we go through validation set/sample.

    trainer, timers = ignite_utils.create_supervised_trainer_timers(
        model, optimizer, loss_fn, device=params.device, prepare_batch=prepare_batch
    )

    print("Beginning Training!")
    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(initial=0, leave=False, total=len(train_dataloader), desc=desc.format(0))

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        pbar.desc = desc.format(engine.state.output)

        tb_writer_train.global_step += 1
        tb_writer_train.add_scalar(TOTAL_LOSS_TB_STRING, engine.state.output)
        if engine.state.iteration % params.log_interval_histograms == 0:
            # Every 100 steps we store the histograms of our sampled z's to ensure not getting posterior collapse
            model.encoder.shallow_dist._tb_logger = tb_writer_train  # turn it on for this step
        else:
            model.encoder.shallow_dist._tb_logger = None

        pbar.update()

    @trainer.on(Events.EPOCH_STARTED)
    def setup_trainer(engine):
        timers.reset()
        model._logger_manager = train_log_helper
        tb_writer_train.add_scalar("epoch_num", engine.state.epoch)
        tqdm.write(f"\n\n# Epoch {engine.state.epoch} starting!")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        tqdm.write(f"\n\n# Epoch {engine.state.epoch} finished")
        tqdm.write(f"## Timings:\n{str(timers)}")
        tqdm.write(f"## Validation")

        # Switch the logger for validation:
        setup_for_val()

        # Validate via teach forced loss and reconstruction stats.
        run_expensive_ops_flag = (engine.state.epoch % params.expensive_ops_freq) == 0
        # ^ we will only do the ops that involve sampling infrequently to save constantly bombarding the server.
        try:
            dogae_utils.validation(val_dataloader, model, tb_writer_val, params.device, {'lambda_': params.lambda_value},
                                   TOTAL_LOSS_TB_STRING, run_expensive_ops_flag)

            # Create some samples!
            if run_expensive_ops_flag:
                out_tuple_trees = dogae_utils.sample_n_from_prior(model, 10, rng)
                tuple_trees_as_text = ' ;\n'.join(map(str, out_tuple_trees))
                tb_writer_train.add_text("tuple_trees_sampled", f"```{tuple_trees_as_text}```")

        except requests.exceptions.Timeout as ex:
            # can have problems with reaction server so continue
            # training for now and ignore validation (can do it from the checkpoints later).
            pass

        # Save a checkpoint
        time_chkpt = strftime("%y-%m-%d_%H:%M:%S", gmtime())
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'mol_to_graph_idx_for_reactants': mol_to_graph_idx_for_reactants,
            'run_name': params.run_name,
            'iter': engine.state.iteration,
            'epoch': engine.state.epoch,
            'lambda':params.lambda_value,
            'dogae_params':_dogae_params,
        },
        path.join(CHKPT_FOLDER, params.run_name, f'time-{time_chkpt}_epoch-{engine.state.epoch}.pth.pick'))

        # Reset the progress bar and run the LR scheduler.
        pbar.n = pbar.last_print_n = 0
        pbar.reset()
        lr_scheduler.step()

    @trainer.on(Events.STARTED)
    def initial_validation(engine):
        tqdm.write(f"# Initial Validation")

        # Switch the logger for validation:
        setup_for_val()

        dogae_utils.validation(val_dataloader, model, tb_writer_val, params.device,
                               {'lambda_': params.lambda_value}, TOTAL_LOSS_TB_STRING)  # run before start training.

    trainer.run(train_dataloader, max_epochs=params.num_epochs)

    pbar.close()


if __name__ == '__main__':
    main(Params())
    print("Done!")
