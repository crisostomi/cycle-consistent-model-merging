import copy
import logging
from typing import Dict, Optional

import torch
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from ccmm.matching.frank_wolfe_sync_matching import frank_wolfe_synchronized_matching
from ccmm.matching.repair import repair_model
from ccmm.matching.utils import (
    apply_permutation_to_statedict,
    get_all_symbols_combinations,
    perm_indices_to_perm_matrix,
    perm_matrix_to_perm_indices,
    unfactor_permutations,
)
from ccmm.matching.weight_matching import PermutationSpec, weight_matching
from ccmm.utils.utils import average_models, l2_norm_models, timeit

pylogger = logging.getLogger(__name__)


class Merger:
    def __init__(self, name, permutation_spec: PermutationSpec):
        self.name = name
        self.permutation_spec = permutation_spec

    def __call__(self, *args, **kwargs):
        pass


class DummyMerger(Merger):
    """
    Return a state dict that has parameters equal to the mean of all the parameters of the different models.
    """

    def __init__(self, name, permutation_spec: PermutationSpec):
        super().__init__(name, permutation_spec)

    def __call__(self, models: Dict[str, LightningModule], train_loader: Optional[DataLoader] = None):
        model_list = list(models.values())
        merged_params = copy.deepcopy(model_list[0].model.state_dict())
        for model in model_list[1:]:
            for key in merged_params.keys():
                merged_params[key] += model.model.state_dict()[key]

        for key in merged_params.keys():
            merged_params[key] /= len(model_list)

        merged_model = model_list[0]
        merged_model.model.load_state_dict(merged_params)

        repaired_model = repair_model(merged_model, models, train_loader)

        return merged_model, repaired_model


class GitRebasinMerger(Merger):
    def __init__(self, name, permutation_spec: PermutationSpec, max_iter=100):
        super().__init__(name, permutation_spec)
        self.max_iter = max_iter

    def __call__(self, models: Dict[str, LightningModule], train_loader: Optional[DataLoader] = None):

        merged_model = self.merge_models(models)

        repaired_model = repair_model(merged_model, models, train_loader)

        return merged_model, repaired_model

    @timeit
    def merge_models(self, models):

        model_params = [model.model.state_dict() for model in models.values()]
        num_models = len(model_params)

        for iteration in range(self.max_iter):
            progress = False

            for model_idx in torch.randperm(num_models):
                model = model_params[model_idx]

                other_models = [model_params[i] for i in range(num_models) if i != model_idx]
                other_models_mean = average_models(other_models)

                l2_before = l2_norm_models(other_models_mean, model)

                permutation = weight_matching(
                    self.permutation_spec,
                    fixed=other_models_mean,
                    permutee=model,
                )

                model = apply_permutation_to_statedict(self.permutation_spec, permutation, model)

                l2_after = l2_norm_models(other_models_mean, model)

                model_params[model_idx] = model

                progress = progress or l2_after < l2_before - 1e-12

                pylogger.info(f"iteration {iteration}/model {model_idx}: l2 diff {l2_after - l2_before:.4f}")

            if not progress:
                break

        mean_params = average_models(model_params)
        merged_model = models[list(models.keys())[0]]
        merged_model.model.load_state_dict(mean_params)

        return merged_model


class FrankWolfeSynchronizedMerger(Merger):
    def __init__(
        self,
        name,
        permutation_spec: PermutationSpec,
        initialization_method,
        average_in_universe=False,
        keep_soft_perms=False,
        max_iter=100,
    ):
        super().__init__(name, permutation_spec)
        self.max_iter = max_iter
        self.average_in_universe = average_in_universe
        self.initialization_method = initialization_method
        self.keep_soft_perms = keep_soft_perms

    def __call__(self, models, train_loader: Optional[DataLoader] = None):

        merged_model, models_permuted_to_universe = self.merge_models(models)
        repaired_model = repair_model(merged_model, models_permuted_to_universe, train_loader)

        return merged_model, repaired_model

    @timeit
    def merge_models(self, models):
        symbols = list(models.keys())

        merged_model = copy.deepcopy(models[symbols[0]])

        combinations = get_all_symbols_combinations(symbols)
        canonical_combinations = [(source, target) for (source, target) in combinations if source < target]  # NOQA

        models_permuted_to_universe = {symbol: copy.deepcopy(model) for symbol, model in models.items()}

        perm_indices, _ = frank_wolfe_synchronized_matching(
            models=models,
            perm_spec=self.permutation_spec,
            symbols=symbols,
            combinations=canonical_combinations,
            max_iter=self.max_iter,
            initialization_method=self.initialization_method,
            keep_soft_perms=self.keep_soft_perms,
        )

        for symbol in symbols:
            perms_to_apply = {}

            for perm_name in perm_indices[symbol].keys():
                perm = perm_indices[symbol][perm_name]

                if self.keep_soft_perms:
                    perm = perm.T
                    perm_to_apply = perm
                else:
                    perm = perm_indices_to_perm_matrix(perm).T
                    perm_to_apply = perm_matrix_to_perm_indices(perm)

                perms_to_apply[perm_name] = perm_to_apply

            updated_params = apply_permutation_to_statedict(
                self.permutation_spec, perms_to_apply, models[symbol].model.state_dict()
            )
            models_permuted_to_universe[symbol].model.load_state_dict(updated_params)

        if self.average_in_universe:
            merged_params = average_models([model.model.state_dict() for model in models_permuted_to_universe.values()])
        else:
            merged_params = models_permuted_to_universe[symbols[0]]

        merged_model.model.load_state_dict(merged_params)

        return merged_model, models_permuted_to_universe


class FrankWolfeToReferenceMerger(Merger):
    def __init__(
        self,
        name,
        permutation_spec: PermutationSpec,
        initialization_method,
        max_iter=100,
    ):
        super().__init__(name, permutation_spec)
        self.max_iter = max_iter
        self.initialization_method = initialization_method

    def __call__(self, models, train_loader: Optional[DataLoader] = None):
        merged_model = models[list(models.keys())[0]]
        num_models = len(models)

        symbols = list(models.keys())

        combinations = get_all_symbols_combinations(symbols)
        canonical_combinations = [(source, target) for (source, target) in combinations if source < target]

        model_params = {symbol: model.model.state_dict() for symbol, model in models.items()}

        perm_indices, _ = frank_wolfe_synchronized_matching(
            models=models,
            perm_spec=self.permutation_spec,
            symbols=symbols,
            combinations=canonical_combinations,
            max_iter=self.max_iter,
            initialization_method=self.initialization_method,
        )

        perm_indices = unfactor_permutations(perm_indices)

        ref_model_id = 0

        other_model_ids = [i for i in range(num_models) if i != ref_model_id]

        for other_model_id in other_model_ids:

            other_model_params = model_params[other_model_id]

            other_model_params = apply_permutation_to_statedict(
                self.permutation_spec, perm_indices[ref_model_id][other_model_id], other_model_params
            )

            model_params[other_model_id] = other_model_params

        mean_params = average_models(model_params)
        merged_model.model.load_state_dict(mean_params)

        return merged_model


class GitRebasinPairwiseMerger(Merger):
    def __init__(self, name, permutation_spec: PermutationSpec, max_iter=100, ref_model_symbol="a"):
        super().__init__(name, permutation_spec)
        self.max_iter = max_iter
        self.ref_model_symbol = ref_model_symbol

    def __call__(self, models: Dict[str, LightningModule], train_loader: Optional[DataLoader] = None, repair=True):

        model_params = {symb: model.model.state_dict() for symb, model in models.items()}

        symbols = sorted(list(models.keys()))

        ref_model_params = model_params[self.ref_model_symbol]

        other_model_symb = [symb for symb in symbols if symb != self.ref_model_symbol]
        models_permuted_to_ref = {symbol: copy.deepcopy(model) for symbol, model in models.items()}

        for other_model_symb in other_model_symb:

            other_model_params = model_params[other_model_symb]

            permutation = weight_matching(
                self.permutation_spec, fixed=ref_model_params, permutee=other_model_params, verbose=True
            )

            other_model_params = apply_permutation_to_statedict(self.permutation_spec, permutation, other_model_params)

            models_permuted_to_ref[other_model_symb].model.load_state_dict(other_model_params)

            model_params[other_model_symb] = other_model_params

        mean_params = average_models(model_params)
        merged_model = models[list(models.keys())[0]]
        merged_model.model.load_state_dict(mean_params)

        if repair:
            repaired_model = repair_model(merged_model, models_permuted_to_ref, train_loader)

            return merged_model, repaired_model

        return merged_model
