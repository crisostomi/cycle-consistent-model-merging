import copy
import logging
from pathlib import Path
from typing import Dict

import hydra
import numpy as np
import omegaconf
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import EMNIST
from tqdm import tqdm

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.model_logging import NNLogger

from ccmm.matching.repair import repair_model
from ccmm.matching.utils import (
    apply_permutation_to_statedict,
    get_all_symbols_combinations,
    load_permutations,
    restore_original_weights,
)
from ccmm.utils.utils import get_model, linear_interpolate, load_model_from_artifact, map_model_seed_to_symbol

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    core_cfg = copy.deepcopy(cfg)
    cfg = cfg.matching

    seed_index_everything(cfg)

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=None,
    )

    # [1, 2, 3, ..]
    model_seeds = cfg.model_seeds
    cfg.results_path = Path(cfg.results_path) / f"{len(model_seeds)}"

    # {a: 1, b: 2, c: 3, ..}
    symbols_to_seed = {map_model_seed_to_symbol(seed): seed for seed in model_seeds}

    # {a, b, c, ..}
    symbols = set(symbols_to_seed.keys())

    # (a, b), (a, c), (b, c), ...
    all_combinations = get_all_symbols_combinations(symbols)

    artifact_path = (
        lambda seed: f"{core_cfg.core.entity}/{core_cfg.core.project_name}/{core_cfg.dataset.name}_{core_cfg.model.model_identifier}_{seed}:latest"
    )

    updated_params = {symb: {other_symb: None for other_symb in symbols.difference(symb)} for symb in symbols}

    permutation_spec_builder = instantiate(core_cfg.model.permutation_spec_builder)
    permutation_spec = permutation_spec_builder.create_permutation_spec()

    permutations = load_permutations(
        cfg.permutations_path / "permutations.json", factored=cfg.use_factored_permutations
    )

    # combinations of the form (a, b), (a, c), (b, c), .. and not (b, a), (c, a) etc

    canonical_combinations = [(fixed, permutee) for (fixed, permutee) in all_combinations if fixed < permutee]

    for fixed, permutee in canonical_combinations:
        pylogger.info(f"Permuting model {permutee} into {fixed}.")

        cfg.model_seeds = [symbols_to_seed[fixed], symbols_to_seed[permutee]]

        logger: NNLogger = NNLogger(logging_cfg=core_cfg.logging, cfg=core_cfg, resume_id=template_core.resume_id)

        # {a: model_a, b: model_b, c: model_c, ..}
        models: Dict[str, LightningModule] = {
            map_model_seed_to_symbol(seed): load_model_from_artifact(logger.experiment, artifact_path(seed))
            for seed in cfg.model_seeds
        }

        model_orig_weights = {symbol: copy.deepcopy(model.model.state_dict()) for symbol, model in models.items()}

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

        lambdas = np.linspace(0, 1, num=cfg.num_interpolation_steps)

        restore_original_weights(models, model_orig_weights)
        results = evaluate_pair_of_models(
            models,
            fixed,
            permutee,
            updated_params,
            train_loader,
            test_loader,
            lambdas,
            core_cfg,
        )

        log_results(results, lambdas)

        if logger is not None:
            logger.log_configuration(model=list(models.values())[0], cfg=core_cfg)
            logger.experiment.finish()

    return logger.run_dir


def evaluate_pair_of_models(models, fixed_id, permutee_id, updated_params, train_loader, test_loader, lambdas, cfg):
    fixed_model = models[fixed_id]
    permutee_model = models[permutee_id]

    get_model(permutee_model).load_state_dict(updated_params[fixed_id][permutee_id])

    results = evaluate_interpolated_models(
        fixed_model, permutee_model, train_loader, test_loader, lambdas, cfg.matching, repair=False
    )

    return results


def evaluate_interpolated_models(fixed, permutee, train_loader, test_loader, lambdas, cfg, repair=False):
    fixed = fixed.cuda()
    permutee = permutee.cuda()

    fixed_dict = copy.deepcopy(get_model(fixed).state_dict())
    permutee_dict = copy.deepcopy(get_model(permutee).state_dict())

    results = {
        "train_acc": [],
        "test_acc": [],
        "train_loss": [],
        "test_loss": [],
    }
    trainer = instantiate(cfg.trainer, enable_progress_bar=False, enable_model_summary=False)

    for lam in tqdm(lambdas):

        interpolated_model = copy.deepcopy(permutee)

        interpolated_params = linear_interpolate(lam, fixed_dict, permutee_dict)
        get_model(interpolated_model).load_state_dict(interpolated_params)

        if repair and lam > 1e-5 and lam < 1 - 1e-5:
            print("Repairing model")
            interpolated_model = repair_model(interpolated_model, {"a": fixed, "c": permutee}, train_loader)

        train_results = trainer.test(interpolated_model, train_loader)

        results["train_acc"].append(train_results[0]["acc/test"])
        results["train_loss"].append(train_results[0]["loss/test"])

        if test_loader:
            test_results = trainer.test(interpolated_model, test_loader)

            results["test_acc"].append(test_results[0]["acc/test"])
            results["test_loss"].append(test_results[0]["loss/test"])

    train_loss_barrier = compute_loss_barrier(results["train_loss"])
    results["train_loss_barrier"] = train_loss_barrier

    if test_loader:
        test_loss_barrier = compute_loss_barrier(results["test_loss"])
        results["test_loss_barrier"] = test_loss_barrier

    return results


def log_results(results, lambdas):

    for metric in ["acc", "loss"]:
        for mode in ["train", "test"]:
            for step, lam in enumerate(lambdas):
                wandb.log({f"{mode}_{metric}": results[f"{mode}_{metric}"][step], "lambda": lam})

    for mode in ["train", "test"]:
        wandb.log({f"{mode}_loss_barrier": results[f"{mode}_loss_barrier"]})


def compute_loss_barrier(losses):
    """
    max_{lambda in [0,1]} loss(alpha * model_a + (1 - alpha) * model_b) - 0.5 * (loss(model_a) + loss(model_b))
    """
    model_a_loss = losses[0]
    model_b_loss = losses[-1]

    avg_loss = (model_a_loss + model_b_loss) / 2

    return max(losses) - avg_loss


# matching_n_models, matching
@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="matching", version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
