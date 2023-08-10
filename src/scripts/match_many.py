import copy
import json
import logging
from pathlib import Path
from pydoc import locate

import hydra
import omegaconf
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.serialization import load_model

import ccmm  # noqa
from ccmm.matching.weight_matching import apply_permutation, weight_matching
from ccmm.utils.plot import plot_interpolation_results
from ccmm.utils.training import test
from ccmm.utils.utils import flatten_params, linear_interpolation

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    seed_index_everything(cfg)

    model_a = load_model_from_info(cfg.model_a_info_path)
    model_b = load_model_from_info(cfg.model_b_info_path)
    model_c = load_model_from_info(cfg.model_c_info_path)

    model_a_orig_weights = copy.deepcopy(flatten_params(model_a.model))
    model_b_orig_weights = copy.deepcopy(flatten_params(model_b.model))
    model_c_orig_weights = copy.deepcopy(flatten_params(model_c.model))

    permutation_spec_builder = instantiate(cfg.permutation_spec_builder)
    permutation_spec = permutation_spec_builder.create_permutation()

    permutations = {"a": {"b": None, "c": None}, "b": {"c": None, "a": None}, "c": {"a": None, "b": None}}
    updated_params = {"b": None, "c": None}
    assert set(permutation_spec.axes_to_perm.keys()) == set(model_a.model.state_dict().keys())

    permutations["b"]["a"] = weight_matching(
        permutation_spec, flatten_params(model_a.model), flatten_params(model_b.model)
    )

    permutations["a"]["b"] = get_inverse_permutations(permutations["b"]["a"])

    updated_params["b"] = apply_permutation(permutation_spec, permutations["b"]["a"], flatten_params(model_b.model))

    permutations["c"]["a"] = weight_matching(
        permutation_spec, flatten_params(model_a.model), flatten_params(model_c.model)
    )

    updated_params["c"] = apply_permutation(permutation_spec, permutations["c"]["a"], flatten_params(model_c.model))

    permutations["a"]["c"] = get_inverse_permutations(permutations["c"]["a"])

    permutations["b"]["c"] = weight_matching(
        permutation_spec, flatten_params(model_c.model), flatten_params(model_b.model)
    )

    permutations["c"]["b"] = get_inverse_permutations(permutations["b"]["c"])

    model_a.model.load_state_dict(model_a_orig_weights)
    model_b.model.load_state_dict(model_b_orig_weights)
    model_c.model.load_state_dict(model_c_orig_weights)

    transform = instantiate(cfg.transform)

    train_dataset = instantiate(cfg.datasets.train, transform=transform)
    test_dataset = instantiate(cfg.datasets.test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    lambdas = torch.linspace(0, 1, steps=cfg.num_interpolation_steps)

    # naive interpolation
    # results_naive = evaluate_interpolated_models(model_a, model_b, train_loader, test_loader, lambdas)
    results_naive = {
        "train_acc": [0 for i in lambdas],
        "train_loss": [0 for i in lambdas],
        "test_acc": [0 for i in lambdas],
        "test_loss": [0 for i in lambdas],
    }

    # clever interpolation
    model_b.model.load_state_dict(updated_params["b"])

    results_clever = evaluate_interpolated_models(model_a, model_b, train_loader, test_loader, lambdas)

    acc_plot_path = Path(f"{cfg.results_path}/{cfg.model_identifier}_acc.png")
    loss_plot_path = Path(f"{cfg.results_path}/{cfg.model_identifier}_loss.png")

    acc_plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_interpolation_results(lambdas, results_naive, results_clever, metric_to_plot="acc")
    fig.savefig(acc_plot_path)

    fig = plot_interpolation_results(lambdas, results_naive, results_clever, metric_to_plot="loss")
    fig.savefig(loss_plot_path)


def get_inverse_permutations(permutations):
    # TODO
    return permutations


def load_model_from_info(model_info_path):
    model_info = json.load(open(model_info_path))
    model_class = locate(model_info["class"])

    model = load_model(model_class, checkpoint_path=Path(model_info["path"] + ".zip"))
    model.eval()

    return model


def evaluate_interpolated_models(model_a, model_b, train_loader, test_loader, lambdas):
    model_a = model_a.cuda()
    model_b = model_b.cuda()

    model_a_dict = copy.deepcopy(model_a.model.state_dict())
    model_b_dict = copy.deepcopy(model_b.model.state_dict())

    results = {
        "train_acc": [],
        "test_acc": [],
        "train_loss": [],
        "test_loss": [],
    }

    for lam in tqdm(lambdas):
        naive_p = linear_interpolation(lam, model_a_dict, model_b_dict)
        model_b.model.load_state_dict(naive_p)

        train_loss, train_acc = test(model_b.cuda(), "cuda", train_loader)
        results["train_acc"].append(train_acc)
        results["train_loss"].append(train_loss)

        test_loss, test_acc = test(model_b.cuda(), "cuda", test_loader)
        results["test_acc"].append(test_acc)
        results["test_loss"].append(test_loss)

    return results


@hydra.main(config_path=str(PROJECT_ROOT / "conf/matching"), config_name="match_many")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
