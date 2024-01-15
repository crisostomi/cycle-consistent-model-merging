import copy
import logging
from typing import Dict

import torch
from pytorch_lightning import LightningModule

from ccmm.matching.weight_matching import PermutationSpec, apply_permutation, weight_matching
from ccmm.utils.utils import average_models, l2_norm_models

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

                model = apply_permutation(self.permutation_spec, permutation, model)

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
