
import typing
import copy
import logging
import itertools

import numpy as np
import tabulate
import torch
from torch.utils import tensorboard
from torch.nn import functional as F
from tqdm import tqdm


from ..model import dog_decoder
from ..model import dogae
from ..model import reaction_predictors
from ..data import smiles_to_feats
from ..data import synthesis_trees
from ..utils import ignite_utils
from ..utils import settings

from autoencoders import base_ae


def load_model(weight_path, device, server_address, log_path_for_react_predict):

    # Checkpoint
    chkpt = torch.load(weight_path, device)

    # Data
    starting_reactants = chkpt['mol_to_graph_idx_for_reactants']
    collate_func = synthesis_trees.CollateWithLargestFirstReordering(starting_reactants)

    # Model -- set up individual componenets
    mol_to_graph_idx_for_reactants = collate_func.base_mol_to_idx_dict.copy()
    reactant_graphs = copy.copy(collate_func.reactant_graphs)
    reactant_graphs.inplace_torch_to(device)
    reactant_vocab = dog_decoder.DOGGenerator.ReactantVocab(reactant_graphs, mol_to_graph_idx_for_reactants)

    smi2graph_func = lambda smi: smiles_to_feats.DEFAULT_SMILES_FEATURIZER.smi_to_feats(smi)
    reaction_predictor = reaction_predictors.OpenNMTServerPredictor()

    # Add a logger to the reaction predictor so can find out the reactions it predicts later.
    log_hndlr = logging.FileHandler(log_path_for_react_predict)
    log_hndlr.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_hndlr.setFormatter(formatter)
    reaction_predictor.logger.addHandler(log_hndlr)
    reaction_predictor.logger.setLevel(logging.DEBUG)
    reaction_predictor.logger.propagate = False

    # Model
    _dogae_params = chkpt['dogae_params']
    model, hparams = dogae.get_model(reaction_predictor, smi2graph_func, reactant_vocab, params=_dogae_params)
    model = model.to(device)

    # Load weights into model.
    model.load_state_dict(chkpt['model'])

    property_loss = None

    other_parts = dict(
        log_hndlr=log_hndlr, hparams=hparams, chkpt=chkpt, property_loss=property_loss
    )

    return model, collate_func, other_parts


@torch.no_grad()
def sample_n_from_prior(ae, n, rng: np.random.RandomState, return_extras=False):
    ae.eval()
    latent_dim = ae.latent_prior.mean_log_var[0].shape[1]
    z = torch.tensor(rng.randn(n, latent_dim), dtype=settings.TORCH_FLT).to(next(iter(ae.parameters())).device)
    x, log_probs = ae.decode_from_z_no_grad(z)
    out = [tree.nx_to_tuple_tree(tree.tree, tree.root_smi) for tree in x]
    if return_extras:
        return out, z, log_probs
    else:
        return out

@torch.no_grad()
def sample_ntimes_using_z_and_sort(ae: base_ae.SingleLatentWithPriorAE, n: int,
                                   z: torch.Tensor, return_top_only=False):
    """
    Samples n times from autoencoder, then revaluate the likelihood of each of the samples and sorts based on this.

    :param z: [b, ...]

    List[List], [batch_size, num_samples]
    """
    # Get the samples!
    sample_log_prob_tuples = [ae.decode_from_z_no_grad(z, sample_x=True) for _ in tqdm(range(n), desc="sampling...")]

    # Then seperate them into the trees and the log probs
    syn_tree_samples, log_probs = zip(*sample_log_prob_tuples)

    # Rearrange so that the samples of the same z location are together
    log_probs = torch.stack(list(log_probs), dim=0)  # [num_samples, seq_size, batch_size]
    samples_grouped = list(zip(*syn_tree_samples))

    # we sum the log probs over the sequence and sort over the different samples
    log_probs = torch.sum(log_probs, dim=1).transpose(0,1)  # [batch_size, num_samples]
    sorted_log_probs, indices = torch.sort(log_probs, dim=1, descending=True)

    # we then sort the synthesis trees the same way
    synthesis_trees_sorted = []
    for trees_in_old_order, new_indices in zip(samples_grouped, indices):
        synthesis_trees_sorted.append([trees_in_old_order[i] for i in new_indices])

    # if return_top_only flag is set then we will only return the top synthesis tree from each element.
    if return_top_only:
        out = [elem[0] for elem in synthesis_trees_sorted]
        return out

    return synthesis_trees_sorted, sorted_log_probs


def evaluate_reconstruction_accuracy(x_in: synthesis_trees.PredOutBatch,
                                     syn_trees_out: typing.List[synthesis_trees.SynthesisTree],
                                     tb_write_to_pass: tensorboard.SummaryWriter=None):
    root_node_match = 0
    ismorphic = 0
    graph_edit_distance = 0.
    jacard_similarity_nodes = 0.
    jacard_similarity_reactants = 0.

    for i, (tree_in, tree_out) in enumerate(zip(x_in.syn_trees, syn_trees_out)):
        root_node_match += int(tree_in.root_smi == tree_out.root_smi)
        ismorphic += int(tree_in.compare_with_other_ismorphism(tree_out))
        graph_edit_distance += tree_in.compare_with_other_graph_edit(tree_out)
        j_reactants, j_nodes = tree_in.compare_with_other_jacard(tree_out)
        jacard_similarity_nodes += j_nodes
        jacard_similarity_reactants += j_reactants

        if i < 30 and tb_write_to_pass is not None:
            try:
                tb_write_to_pass.add_text(f"reconstructed-tuple-trees-{i}",
                                      f"in: `{str(tree_in.tuple_tree_repr())}`   ; out: `{str(tree_out.tuple_tree_repr())}`")
            except Exception as ex:
                print(ex)
                pass
            try:
                tb_write_to_pass.add_text(f"reconstructed-seqs-{i}",
                                          f"in: `{tree_in.text_for_construction(strict_mode=True)}`   ; "
                                          f"out: `{tree_out.text_for_construction()}`")
            except Exception as ex:
                print(ex)
                pass

    total_num = float(len(syn_trees_out))
    return tuple([x/total_num for x in [root_node_match, ismorphic, graph_edit_distance,
                                        jacard_similarity_nodes, jacard_similarity_reactants]])




@torch.no_grad()
def validation(val_dataloader, ae, tb_writer_val, device, kw_args_for_ae,
               loss_string="total-loss", run_expensive_ops=True, return_zs=False):
    ae.eval()

    # We are going to record a series of measurements that we will average after we have gone through the whole data:
    meters = {loss_string: ignite_utils.AverageMeter()}
    if run_expensive_ops:
        meters.update({k:ignite_utils.AverageMeter() for k in [
        "root-node-match", 'tree-match',
        "graph-edit-distance", "jacard-similarity-nodes",
        "jacard-similarity-reactants",
        ]})

    in_trees_out_trees = []

    if return_zs:
        zs_out = []

    # Now we will iterate through the minibatches
    with tqdm(val_dataloader, total=len(val_dataloader)) as t:
        for i, (x, _) in enumerate(t):
            # Set up the data
            x: synthesis_trees.PredOutBatch
            x.inplace_to(device)
            batch_size = x.batch_size

            # get graph embeddings
            embedded_graphs = ae.mol_embdr(x.molecular_graphs)
            x.molecular_graph_embeddings = embedded_graphs
            new_node_feats_for_dag = x.molecular_graph_embeddings[x.dags_for_inputs.node_features.squeeze(), :]
            x.dags_for_inputs.node_features = new_node_feats_for_dag

            # Evaluate reconstruction accuracy -- if the flag is set
            meter_name_update_values = []
            if run_expensive_ops:
                reconstruction_trees, _ = ae.reconstruct_no_grad(x)
                tb_write_to_pass = tb_writer_val if i == 0 else None
                (root_node_match, ismorphic_match, graph_edit_distance,
                 jacard_similarity_nodes, jacard_similarity_reactants) = \
                    evaluate_reconstruction_accuracy(x, reconstruction_trees, tb_write_to_pass)
                in_trees_out_trees.extend(list(zip(x.syn_trees, reconstruction_trees)))
                if return_zs:
                    zs_out.append(ae._reconstruction_z.detach().cpu().numpy())

                meter_name_update_values.extend([("root-node-match", root_node_match),
                                                 ("tree-match", ismorphic_match),
                                                 ("graph-edit-distance", graph_edit_distance),
                                                 ("jacard-similarity-nodes", jacard_similarity_nodes),
                                                 ("jacard-similarity-reactants", jacard_similarity_reactants)])

            # Compute the loss (include property loss)
            loss = -ae(x, **kw_args_for_ae).mean()

            meter_name_update_values = [(loss_string, loss)] + meter_name_update_values
            # Update the meters that record the various statistics.
            for meter_name, value in meter_name_update_values:
                meters[meter_name].update(value, n=batch_size)

            # Update the stats in the progress bar
            t.set_postfix(**{k: f"{v.avg:.4E}" for k,v in meters.items()})

    # Print out the final averages and add them to TB:
    print(f"\n## Validation finished over a total of {meters[loss_string].count} items.")
    print(f"The final scores are (AE kw args: {kw_args_for_ae})")
    out = {k:v.avg for k,v in meters.items()}
    if tb_writer_val is not None:
        for k, v in out.items():
            tb_writer_val.add_scalar(k, v)
    print(tabulate.tabulate(list(out.items()), tablefmt="simple", floatfmt=".4f"))
    print("===============================================")

    out = [meters, in_trees_out_trees]
    if return_zs:
        extra_out = dict()
        if return_zs:
            extra_out['all_zs'] = np.concatenate(zs_out, axis=0)
        out.append(extra_out)
    return tuple(out)
