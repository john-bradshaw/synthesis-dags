
import typing
import copy

import torch
from torch import optim
from torch import nn
from dataclasses import dataclass, field
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


from ..model import doggen
from . import opt_utils
from ..data import synthesis_trees


@dataclass(order=True, frozen=True)
class ScoredTupleTree:
    tuple_tree: tuple = field(compare=False)
    score_to_maximize: float

    @property
    def root_smi(self):
        return self.tuple_tree[0]

@dataclass
class DogGenFTParams:
    n_rounds: int = 30
    n_samples_per_round: int = 7000
    n_samples_to_keep_per_round: int = 1500
    n_epochs_for_finetuning: int = 2
    batch_size: int = 64
    break_down_tuple_tree_into_parts: bool = False # this is if you want to break down the tuple tree into
    sample_batch_size: int = 200
    learning_rate: float = 1e-3
    clip_gradients: bool = True


@dataclass
class DogGenFTParts:
    model: doggen.DogGen
    scorer: opt_utils.PropertyEvaluator
    reactant_vocab_set: typing.Set[str]
    rng: np.random.RandomState
    dataloader_factory: typing.Callable
    prepare_batch: typing.Callable
    loss_fn: typing.Callable
    device: str


class DogGenFT:
    def __init__(self, parts: DogGenFTParts, params: DogGenFTParams):
        self.parts = parts
        self.hparams = params

        self.optimizer = None
        self._num_total_train_steps_for_hc = None


    def run_finetuning(self, initial_tuple_trees, tb_logger: SummaryWriter):
        """
        :param initial_tuple_trees:  all SMILES in canoncical form already.
        :param tb_logger:
        :return:
        """

        seen_tts = self.filter_out_uninteresting_trees_and_clean(initial_tuple_trees, set())
        sorted_tts = self.score_new_trees_and_sort(seen_tts, [])
        self._report_best(sorted_tts, tb_logger, 0)

        self.optimizer = optim.Adam(self.parts.model.parameters(), lr=self.hparams.learning_rate)
        self._num_total_train_steps_for_hc = 0

        print('## Sampling before tuning...')
        sampled_dirty_tts = self.sample_from_model()
        sampled_clean_tts = self.filter_out_uninteresting_trees_and_clean(sampled_dirty_tts, sorted_tts)
        sorted_tts = self.score_new_trees_and_sort(sampled_clean_tts, sorted_tts)
        self._report_best(sorted_tts, tb_logger, 0)

        for round in range(self.hparams.n_rounds):
            print(f"# Starting round {round}")
            print('## Setting up new batch for training...')
            new_batch_for_fine_tuning = [e.tuple_tree for e in sorted_tts[:self.hparams.n_samples_to_keep_per_round]]

            print('## Starting dog_gen on new batch...')
            self.train_one_round(new_batch_for_fine_tuning, tb_logger)

            print('## Sampling...')
            sampled_dirty_tts = self.sample_from_model()
            sampled_clean_tts = self.filter_out_uninteresting_trees_and_clean(sampled_dirty_tts, sorted_tts)
            sorted_tts = self.score_new_trees_and_sort(sampled_clean_tts, sorted_tts)
            self._report_best(sorted_tts, tb_logger, round)
        return sorted_tts

    def train_one_round(self, tuple_trees_to_train_on, tb_logger: SummaryWriter):
        self.parts.model.train()
        train_dataloader = self.parts.dataloader_factory(tuple_trees=tuple_trees_to_train_on, batch_size=self.hparams.batch_size)
        for epoch in range(self.hparams.n_epochs_for_finetuning):
            print(f"### Training epoch {epoch}")
            loss = 0.
            for data in tqdm(train_dataloader, desc="training"):
                self.optimizer.zero_grad()
                batch = self.parts.prepare_batch(data, self.parts.device)
                loss = self.parts.loss_fn(self.parts.model, *batch)
                loss.backward()
                tb_logger.add_scalar("train_one_round_loss", loss.item(), self._num_total_train_steps_for_hc)
                if self.hparams.clip_gradients:
                    nn.utils.clip_grad_norm_(self.parts.model.parameters(), 1.0)
                self.optimizer.step()
                self._num_total_train_steps_for_hc += 1
            print(f"loss, last batch: {loss.item()}")

    def _report_best(self, sorted_list: typing.List[ScoredTupleTree],
                     tb_logger: SummaryWriter, step_num):
        print(f"Step {step_num}, Best 3 TTs so far are {sorted_list[:3]}")
        tb_logger.add_scalar("Best score So Far", sorted_list[0].score_to_maximize,
                             global_step=step_num)
        tb_logger.add_scalar("Second best score So Far", sorted_list[1].score_to_maximize,
                             global_step=step_num)
        tb_logger.add_scalar("Third best score So Far", sorted_list[2].score_to_maximize,
                             global_step=step_num)
        tb_logger.add_scalar("Mean of top 50 scores", np.mean([sorted_list[i].score_to_maximize for
                                                              i in range(50)]), global_step=step_num)

    def sample_from_model(self):
        self.parts.model.eval()
        out_list: typing.List[synthesis_trees.SynthesisTree] = []
        for _ in tqdm(range(int(np.ceil(self.hparams.n_samples_per_round / self.hparams.sample_batch_size))),
                      desc="sampling from model"):
            syn_trees, _ = self.parts.model.sample(batch_size=self.hparams.sample_batch_size)
            out_list.extend(syn_trees)
        out_tts = [e.tuple_tree_repr() for e in out_list]
        return out_tts

    def score_new_trees_and_sort(self, new_tts, existing_tree_scores):
        existing_tree_scores = copy.copy(existing_tree_scores)
        scores = self.parts.scorer.evaluate_molecules([e[0] for e in new_tts])
        new_scored_tts = [ScoredTupleTree(tt, score) for tt, score in zip(new_tts, scores)]
        existing_tree_scores.extend(new_scored_tts)
        return sorted(existing_tree_scores, reverse=True)

    def filter_out_uninteresting_trees_and_clean(self, list_of_tuple_trees, seen_tt_scores):
        out = []
        invariant_seen_tts = set([synthesis_trees.SynthesisTree.make_tuple_tree_invariant(elem.tuple_tree)
                                  for elem in seen_tt_scores])
        for tt in tqdm(list_of_tuple_trees, desc="cleaning trees"):

            # Filter out reactants, at top as these we already know how to make -- just buy them!
            if tt[0] in self.parts.reactant_vocab_set:
                continue

            # if already exists then remove as already seen it
            invariant_rep = synthesis_trees.SynthesisTree.make_tuple_tree_invariant(tt)
            if invariant_rep in invariant_seen_tts:
                continue
            else:
                invariant_seen_tts.add(invariant_rep)

            # Clean out any tree parts that lead to an already reactant
            clean_tt = synthesis_trees.SynthesisTree.clean_dirty_tuple_tree(tt, self.parts.reactant_vocab_set)
            out.append(clean_tt)
        return out

