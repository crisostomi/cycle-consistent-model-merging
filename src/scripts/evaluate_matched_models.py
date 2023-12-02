import copy
import logging
from pathlib import Path
from typing import Dict

import hydra
import omegaconf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything

from ccmm.matching.weight_matching import apply_permutation
from ccmm.utils.matching_utils import get_all_symbols_combinations, restore_original_weights
from ccmm.utils.plot import plot_interpolation_results
from ccmm.utils.utils import linear_interpolation, load_model_from_info, load_permutations, map_model_seed_to_symbol

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    seed_index_everything(cfg)

    # [1, 2, 3, ..]
    model_seeds = cfg.model_seeds
    cfg.results_path = Path(cfg.results_path) / f"{len(model_seeds)}"

    if not cfg.sync_method:
        pylogger.warning("Only using naive and git re-basin interpolation.")

    # {a: 1, b: 2, c: 3, ..}
    symbols_to_seed = {map_model_seed_to_symbol(seed): seed for seed in model_seeds}

    # {a, b, c, ..}
    symbols = set(symbols_to_seed.keys())

    # (a, b), (a, c), (b, c), ...
    all_combinations = get_all_symbols_combinations(symbols)

    models: Dict[str, LightningModule] = {
        map_model_seed_to_symbol(seed): load_model_from_info(cfg.model_info_path, seed) for seed in model_seeds
    }

    model_orig_weights = {symbol: copy.deepcopy(model.model.state_dict()) for symbol, model in models.items()}

    updated_params = {symb: {other_symb: None for other_symb in symbols.difference(symb)} for symb in symbols}
    sync_updated_params = {symb: {other_symb: None for other_symb in symbols.difference(symb)} for symb in symbols}

    permutation_spec_builder = instantiate(cfg.permutation_spec_builder)
    permutation_spec = permutation_spec_builder.create_permutation()

    permutations = load_permutations(cfg.permutations_path / "permutations.json")
    sync_permutations = (
        load_permutations(cfg.permutations_path / "sync_permutations.json") if cfg.sync_method else permutations
    )

    for fixed, permutee in all_combinations:
        # sync_perms[a, b] maps b -> a
        sync_updated_params[fixed][permutee] = apply_permutation(
            permutation_spec, sync_permutations[fixed][permutee], models[permutee].model.state_dict()
        )
        restore_original_weights(models, model_orig_weights)
        updated_params[fixed][permutee] = apply_permutation(
            permutation_spec, permutations[fixed][permutee], models[permutee].model.state_dict()
        )
        restore_original_weights(models, model_orig_weights)

    transform = instantiate(cfg.transform)

    train_dataset = instantiate(cfg.datasets.train, transform=transform)
    test_dataset = instantiate(cfg.datasets.test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    lambdas = torch.linspace(0, 1, steps=cfg.num_interpolation_steps)

    # combinations of the form (a, b), (a, c), (b, c), .. and not (b, a), (c, a) etc
    canonical_combinations = [(fixed, permutee) for (fixed, permutee) in all_combinations if fixed < permutee]

    for fixed, permutee in canonical_combinations:
        restore_original_weights(models, model_orig_weights)
        evaluate_pair_of_models(
            models, fixed, permutee, updated_params, sync_updated_params, train_loader, test_loader, lambdas, cfg
        )


def evaluate_pair_of_models(
    models, fixed_id, permutee_id, updated_params, sync_updated_params, train_loader, test_loader, lambdas, cfg
):
    fixed_model = models[fixed_id]
    permutee_model = models[permutee_id]

    # synchronized interpolation
    permutee_model.model.load_state_dict(sync_updated_params[fixed_id][permutee_id])

    results_sync = evaluate_interpolated_models(fixed_model, permutee_model, train_loader, test_loader, lambdas, cfg)

    # naive interpolation
    # results_naive = evaluate_interpolated_models(model_a, model_b, train_loader, test_loader, lambdas)
    results_naive = {
        "train_acc": [0 for i in lambdas],
        "train_loss": [0 for i in lambdas],
        "test_acc": [0 for i in lambdas],
        "test_loss": [0 for i in lambdas],
    }

    # clever interpolation
    permutee_model.model.load_state_dict(updated_params[fixed_id][permutee_id])

    results_clever = evaluate_interpolated_models(fixed_model, permutee_model, train_loader, test_loader, lambdas, cfg)

    acc_plot_path = Path(
        f"{cfg.results_path}/{cfg.model_identifier}_acc_{fixed_id}_{permutee_id}_{cfg.sync_method}.png"
    )
    loss_plot_path = Path(
        f"{cfg.results_path}/{cfg.model_identifier}_loss_{fixed_id}_{permutee_id}_{cfg.sync_method}.png"
    )

    acc_plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_interpolation_results(lambdas, results_naive, results_clever, results_sync, metric_to_plot="acc")
    fig.savefig(acc_plot_path)

    fig = plot_interpolation_results(lambdas, results_naive, results_clever, results_sync, metric_to_plot="loss")
    fig.savefig(loss_plot_path)


def evaluate_interpolated_models(fixed, permutee, train_loader, test_loader, lambdas, cfg):
    fixed = fixed.cuda()
    permutee = permutee.cuda()

    fixed_dict = copy.deepcopy(fixed.model.state_dict())
    permutee_dict = copy.deepcopy(permutee.model.state_dict())

    results = {
        "train_acc": [],
        "test_acc": [],
        "train_loss": [],
        "test_loss": [],
    }
    trainer = instantiate(cfg.trainer)

    for lam in tqdm(lambdas):
        interpolated_params = linear_interpolation(lam, fixed_dict, permutee_dict)
        permutee.model.load_state_dict(interpolated_params)

        train_results = trainer.test(permutee, train_loader)

        results["train_acc"].append(train_results[0]["acc/test"])
        results["train_loss"].append(train_results[0]["loss/test"])

        test_results = trainer.test(permutee, test_loader)

        results["test_acc"].append(test_results[0]["acc/test"])
        results["test_loss"].append(test_results[0]["loss/test"])

    return results


@hydra.main(config_path=str(PROJECT_ROOT / "conf/matching"), config_name="git_rebasin")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
