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

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.model_logging import NNLogger

from ccmm.matching.utils import apply_permutation_to_statedict, get_all_symbols_combinations, restore_original_weights
from ccmm.utils.utils import linear_interpolation, load_model_from_info, load_permutations, map_model_seed_to_symbol

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    core_cfg = cfg
    cfg = cfg.matching

    seed_index_everything(cfg)

    # [1, 2, 3, ..]
    model_seeds = cfg.model_seeds
    cfg.results_path = Path(cfg.results_path) / f"{len(model_seeds)}"

    # {a: 1, b: 2, c: 3, ..}
    symbols_to_seed = {map_model_seed_to_symbol(seed): seed for seed in model_seeds}

    # {a, b, c, ..}
    symbols = set(symbols_to_seed.keys())

    # (a, b), (a, c), (b, c), ...
    all_combinations = get_all_symbols_combinations(symbols)

    models: Dict[str, LightningModule] = {
        map_model_seed_to_symbol(seed): load_model_from_info(cfg.model_info_path, seed) for seed in model_seeds
    }

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=None,
    )

    logger: NNLogger = NNLogger(logging_cfg=core_cfg.logging, cfg=core_cfg, resume_id=template_core.resume_id)

    model_orig_weights = {symbol: copy.deepcopy(model.model.state_dict()) for symbol, model in models.items()}

    updated_params = {symb: {other_symb: None for other_symb in symbols.difference(symb)} for symb in symbols}

    permutation_spec_builder = instantiate(core_cfg.model.permutation_spec_builder)
    permutation_spec = permutation_spec_builder.create_permutation()

    permutations = load_permutations(cfg.permutations_path / "permutations.json")

    for fixed, permutee in all_combinations:
        # perms[a, b] maps b -> a
        updated_params[fixed][permutee] = apply_permutation_to_statedict(
            permutation_spec, permutations[fixed][permutee], models[permutee].model.state_dict()
        )
        restore_original_weights(models, model_orig_weights)

    transform = instantiate(core_cfg.dataset.test.transform)

    train_dataset = instantiate(core_cfg.dataset.train, transform=transform)
    test_dataset = instantiate(core_cfg.dataset.test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    lambdas = torch.linspace(0, 1, steps=cfg.num_interpolation_steps)

    # combinations of the form (a, b), (a, c), (b, c), .. and not (b, a), (c, a) etc
    canonical_combinations = [(fixed, permutee) for (fixed, permutee) in all_combinations if fixed < permutee]

    for fixed, permutee in canonical_combinations:
        restore_original_weights(models, model_orig_weights)
        evaluate_pair_of_models(
            models,
            fixed,
            permutee,
            updated_params,
            train_loader,
            test_loader,
            lambdas,
            core_cfg,
            logger,
        )

    if logger is not None:
        logger.log_configuration(model=list(models.values())[0], cfg=core_cfg)
        logger.experiment.finish()

    return logger.run_dir


def evaluate_pair_of_models(
    models, fixed_id, permutee_id, updated_params, train_loader, test_loader, lambdas, cfg, logger
):
    fixed_model = models[fixed_id]
    permutee_model = models[permutee_id]

    permutee_model.model.load_state_dict(updated_params[fixed_id][permutee_id])

    results = evaluate_interpolated_models(
        fixed_model, permutee_model, train_loader, test_loader, lambdas, cfg.matching
    )

    metrics = ["acc", "loss"]
    for step, lambd in enumerate(lambdas):
        for metric in metrics:
            logger.experiment.log({f"{metric}/test": results[f"test_{metric}"][step]}, step=step)
            logger.experiment.log({f"{metric}/train": results[f"train_{metric}"][step]}, step=step)

    for mode in ["train", "test"]:
        logger.experiment.log({f"loss_barrier/{mode}": results[f"{mode}_loss_barrier"]})


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

    train_loss_barrier = compute_loss_barrier(results["train_loss"])
    test_loss_barrier = compute_loss_barrier(results["test_loss"])

    results["train_loss_barrier"] = train_loss_barrier
    results["test_loss_barrier"] = test_loss_barrier

    return results


def compute_loss_barrier(losses):
    """
    max_{lambda in [0,1]} loss(alpha * model_a + (1 - alpha) * model_b) - 0.5 * (loss(model_a) + loss(model_b))
    """
    model_a_loss = losses[0]
    model_b_loss = losses[-1]

    avg_loss = (model_a_loss + model_b_loss) / 2

    return max(losses) - avg_loss


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="matching", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
