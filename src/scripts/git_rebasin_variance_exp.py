import copy
import logging
from typing import Dict

import hydra
import numpy as np
import omegaconf
import torch  # noqa
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from scripts.evaluate_matched_models import evaluate_pair_of_models

import ccmm  # noqa
from ccmm.matching.utils import apply_permutation_to_statedict, get_inverse_permutations, restore_original_weights
from ccmm.utils.utils import load_model_from_artifact, map_model_seed_to_symbol

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    core_cfg = cfg  # NOQA
    cfg = cfg.matching

    seed_index_everything(cfg)

    run = wandb.init(project=core_cfg.core.project_name, entity=core_cfg.core.entity, job_type="matching")

    # {a: 1, b: 2, c: 3, ..}
    symbols_to_seed: Dict[int, str] = {map_model_seed_to_symbol(seed): seed for seed in cfg.model_seeds}

    artifact_path = (
        lambda seed: f"{core_cfg.core.entity}/{core_cfg.core.project_name}/{core_cfg.dataset.name}_{core_cfg.model.model_identifier}_{seed}:latest"
    )

    # {a: model_a, b: model_b, c: model_c, ..}
    models: Dict[str, LightningModule] = {
        map_model_seed_to_symbol(seed): load_model_from_artifact(run, artifact_path(seed)) for seed in cfg.model_seeds
    }

    # data structure that specifies the permutations acting on each layer and on what axis
    permutation_spec_builder = instantiate(core_cfg.model.permutation_spec_builder)
    permutation_spec = permutation_spec_builder.create_permutation()

    ref_model = list(models.values())[0]
    assert set(permutation_spec.layer_and_axes_to_perm.keys()) == set(ref_model.model.state_dict().keys())

    # always permute the model having larger character order, i.e. c -> b, b -> a and so on ...
    symbols = set(symbols_to_seed.keys())
    sorted_symbols = sorted(symbols, reverse=False)
    fixed_symbol, permutee_symbol = sorted_symbols
    fixed_model, permutee_model = models[fixed_symbol], models[permutee_symbol]

    # dicts for permutations and permuted params, D[a][b] refers to the permutation/params to map b -> a
    permutations = {symb: {other_symb: None for other_symb in symbols.difference(symb)} for symb in symbols}

    matcher = instantiate(cfg.matcher, permutation_spec=permutation_spec)

    updated_params = {symb: {other_symb: None for other_symb in symbols.difference(symb)} for symb in symbols}
    model_orig_weights = {symbol: copy.deepcopy(model.model.state_dict()) for symbol, model in models.items()}

    all_train_accs, all_test_accs, all_train_losses, all_test_losses = [], [], [], []

    for seed in range(1, 2):

        cfg.seed_index = seed
        seed_index_everything(cfg)

        restore_original_weights(models, model_orig_weights)

        permutations[fixed_symbol][permutee_symbol], _ = matcher(fixed=fixed_model.model, permutee=permutee_model.model)

        permutations[permutee_symbol][fixed_symbol] = get_inverse_permutations(
            permutations[fixed_symbol][permutee_symbol]
        )

        # perms[a, b] maps b -> a
        updated_params[fixed_symbol][permutee_symbol] = apply_permutation_to_statedict(
            permutation_spec, permutations[fixed_symbol][permutee_symbol], models[permutee_symbol].model.state_dict()
        )
        restore_original_weights(models, model_orig_weights)

        transform = instantiate(core_cfg.dataset.test.transform)

        train_dataset = instantiate(core_cfg.dataset.train, transform=transform)
        test_dataset = instantiate(core_cfg.dataset.test, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

        lambdas = np.linspace(0, 1, num=cfg.num_interpolation_steps)

        restore_original_weights(models, model_orig_weights)
        results = evaluate_pair_of_models(
            models,
            fixed_symbol,
            permutee_symbol,
            updated_params,
            train_loader,
            test_loader,
            lambdas,
            core_cfg,
        )

        for model in models.values():
            model = model.to("cpu")

        midpoint_train_acc = results["train_acc"][1]
        midpoint_test_acc = results["test_acc"][1]

        midpoint_train_loss = results["train_loss"][1]
        midpoint_test_loss = results["test_loss"][1]

        all_train_accs.append(midpoint_train_acc)
        all_test_accs.append(midpoint_test_acc)
        all_train_losses.append(midpoint_train_loss)
        all_test_losses.append(midpoint_test_loss)

    print(all_train_accs)
    print(all_test_accs)
    print(all_train_losses)
    print(all_test_losses)


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="matching", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
