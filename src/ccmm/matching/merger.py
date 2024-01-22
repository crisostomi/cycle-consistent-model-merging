import copy
import logging
from typing import Dict

import torch
from pytorch_lightning import LightningModule

from ccmm.matching.frank_wolfe_sync_matching import frank_wolfe_synchronized_matching
from ccmm.matching.utils import (
    apply_permutation_to_statedict,
    get_all_symbols_combinations,
    perm_indices_to_perm_matrix,
    perm_matrix_to_perm_indices,
)
from ccmm.matching.weight_matching import PermutationSpec, weight_matching
from ccmm.utils.utils import average_models, l2_norm_models, unfactor_permutations

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

    def __call__(self, models: Dict[str, LightningModule]):
        model_list = list(models.values())
        merged_params = copy.deepcopy(model_list[0].model.state_dict())
        for model in model_list[1:]:
            for key in merged_params.keys():
                merged_params[key] += model.model.state_dict()[key]

        for key in merged_params.keys():
            merged_params[key] /= len(model_list)

        merged_model = model_list[0]
        merged_model.model.load_state_dict(merged_params)
        return merged_model


class GitRebasinMerger(Merger):
    def __init__(self, name, permutation_spec: PermutationSpec, max_iter=100):
        super().__init__(name, permutation_spec)
        self.max_iter = max_iter

    def __call__(self, models: Dict[str, LightningModule]):
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
        max_iter=100,
    ):
        super().__init__(name, permutation_spec)
        self.max_iter = max_iter
        self.average_in_universe = average_in_universe
        self.initialization_method = initialization_method

    def __call__(self, models):
        symbols = list(models.keys())

        merged_model = models[symbols[0]]

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

        for symbol in symbols:
            perms_to_apply = {}
            for perm_name in perm_indices[symbol].keys():
                perm = perm_indices_to_perm_matrix(perm_indices[symbol][perm_name]).T
                perms_to_apply[perm_name] = perm_matrix_to_perm_indices(perm)
            updated_params = apply_permutation_to_statedict(self.permutation_spec, perms_to_apply, model_params[symbol])
            model_params[symbol] = updated_params

        if self.average_in_universe:
            merged_params = average_models(model_params)
        else:
            merged_params = model_params[symbols[0]]

        merged_model.model.load_state_dict(merged_params)

        return merged_model


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

    def __call__(self, models):
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
    def __init__(self, name, permutation_spec: PermutationSpec, max_iter=100):
        super().__init__(name, permutation_spec)
        self.max_iter = max_iter

    def __call__(self, models: Dict[str, LightningModule]):
        model_params = [model.model.state_dict() for model in models.values()]
        num_models = len(model_params)
        ref_model_id = 0
        ref_model_params = model_params[ref_model_id]

        other_model_ids = [i for i in range(num_models) if i != ref_model_id]

        for other_model_id in other_model_ids:

            other_model_params = model_params[other_model_id]

            permutation = weight_matching(
                self.permutation_spec,
                fixed=ref_model_params,
                permutee=other_model_params,
            )

            other_model_params = apply_permutation_to_statedict(self.permutation_spec, permutation, other_model_params)

            model_params[other_model_id] = other_model_params

        mean_params = average_models(model_params)
        merged_model = models[list(models.keys())[0]]
        merged_model.model.load_state_dict(mean_params)

        return merged_model
