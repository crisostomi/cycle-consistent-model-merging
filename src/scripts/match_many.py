import copy
import itertools
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
from ccmm.matching.weight_matching import apply_permutation, optimize_synchronization, weight_matching
from ccmm.utils.plot import plot_interpolation_results
from ccmm.utils.training import test
from ccmm.utils.utils import flatten_params, is_valid_permutation_matrix, linear_interpolation

pylogger = logging.getLogger(__name__)


def run(cfg: DictConfig) -> str:
    seed_index_everything(cfg)

    model_symbols = cfg.model_symbols

    models = {symbol: load_model_from_info(cfg.model_info_path[symbol]) for symbol in model_symbols}

    model_orig_weights = {symbol: copy.deepcopy(flatten_params(model.model)) for symbol, model in models.items()}

    permutation_spec_builder = instantiate(cfg.permutation_spec_builder)
    permutation_spec = permutation_spec_builder.create_permutation()

    permutations = {"a": {"b": None, "c": None}, "b": {"c": None, "a": None}, "c": {"a": None, "b": None}}
    updated_params = {"a": {"b": None, "c": None}, "b": {"c": None, "a": None}, "c": {"a": None, "b": None}}
    sync_updated_params = {"a": {"b": None, "c": None}, "b": {"c": None, "a": None}, "c": {"a": None, "b": None}}

    assert set(permutation_spec.axes_to_perm.keys()) == set(models["a"].model.state_dict().keys())

    permutations["b"]["a"] = weight_matching(
        permutation_spec, target=flatten_params(models["a"].model), to_permute=flatten_params(models["b"].model)
    )

    permutations["a"]["b"] = get_inverse_permutations(permutations["b"]["a"])

    updated_params["b"]["a"] = apply_permutation(
        permutation_spec, permutations["b"]["a"], flatten_params(models["b"].model)
    )

    permutations["c"]["a"] = weight_matching(
        permutation_spec, target=flatten_params(models["a"].model), to_permute=flatten_params(models["c"].model)
    )

    permutations["a"]["c"] = get_inverse_permutations(permutations["c"]["a"])

    updated_params["c"]["a"] = apply_permutation(
        permutation_spec, permutations["c"]["a"], flatten_params(models["c"].model)
    )

    permutations["c"]["b"] = weight_matching(
        permutation_spec, target=flatten_params(models["b"].model), to_permute=flatten_params(models["c"].model)
    )

    permutations["b"]["c"] = get_inverse_permutations(permutations["c"]["b"])

    updated_params["c"]["b"] = apply_permutation(
        permutation_spec, permutations["c"]["b"], flatten_params(models["c"].model)
    )

    sync_permutations = synchronize_permutations(permutations)

    restore_original_weights(models, model_orig_weights)

    sync_updated_params["b"]["a"] = apply_permutation(
        permutation_spec, sync_permutations["b"]["a"], flatten_params(models["b"].model)
    )
    sync_updated_params["c"]["a"] = apply_permutation(
        permutation_spec, sync_permutations["c"]["a"], flatten_params(models["c"].model)
    )
    sync_updated_params["c"]["b"] = apply_permutation(
        permutation_spec, sync_permutations["c"]["b"], flatten_params(models["c"].model)
    )

    transform = instantiate(cfg.transform)

    train_dataset = instantiate(cfg.datasets.train, transform=transform)
    test_dataset = instantiate(cfg.datasets.test, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size)

    lambdas = torch.linspace(0, 1, steps=cfg.num_interpolation_steps)

    restore_original_weights(models, model_orig_weights)
    evaluate_pair_of_models(
        models, "a", "b", updated_params, sync_updated_params, train_loader, test_loader, lambdas, cfg
    )

    restore_original_weights(models, model_orig_weights)
    evaluate_pair_of_models(
        models, "a", "c", updated_params, sync_updated_params, train_loader, test_loader, lambdas, cfg
    )

    restore_original_weights(models, model_orig_weights)
    evaluate_pair_of_models(
        models, "b", "c", updated_params, sync_updated_params, train_loader, test_loader, lambdas, cfg
    )


def restore_original_weights(models, original_weights):
    for model_id, model in models.items():
        model.model.load_state_dict(original_weights[model_id])


def evaluate_pair_of_models(
    models, model_a_id, model_b_id, updated_params, sync_updated_params, train_loader, test_loader, lambdas, cfg
):
    model_a = models[model_a_id]
    model_b = models[model_b_id]

    # naive interpolation
    results_naive = evaluate_interpolated_models(model_a, model_b, train_loader, test_loader, lambdas)
    # results_naive = {
    #     "train_acc": [0 for i in lambdas],
    #     "train_loss": [0 for i in lambdas],
    #     "test_acc": [0 for i in lambdas],
    #     "test_loss": [0 for i in lambdas],
    # }

    # clever interpolation
    model_b.model.load_state_dict(updated_params[model_b_id][model_a_id])

    results_clever = evaluate_interpolated_models(model_a, model_b, train_loader, test_loader, lambdas)

    # cleverer interpolation
    model_b.model.load_state_dict(sync_updated_params[model_b_id][model_a_id])

    results_cleverer = evaluate_interpolated_models(model_a, model_b, train_loader, test_loader, lambdas)

    acc_plot_path = Path(f"{cfg.results_path}/{cfg.model_identifier}_acc_{model_a_id}_{model_b_id}.png")
    loss_plot_path = Path(f"{cfg.results_path}/{cfg.model_identifier}_loss_{model_a_id}_{model_b_id}.png")

    acc_plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plot_interpolation_results(lambdas, results_naive, results_clever, results_cleverer, metric_to_plot="acc")
    fig.savefig(acc_plot_path)

    fig = plot_interpolation_results(lambdas, results_naive, results_clever, results_cleverer, metric_to_plot="loss")
    fig.savefig(loss_plot_path)


def get_inverse_permutations(permutations):
    inv_permutations = {}
    for perm_name, perm in permutations.items():
        perm_matrix = perm_list_to_perm_matrix(perm)
        inv_permutations[perm_name] = perm_matrix_to_perm_list(perm_matrix)
    return inv_permutations


def perm_list_to_perm_matrix(perm_list):
    n = len(perm_list)
    perm_matrix = torch.eye(n)[perm_list.long()]
    return perm_matrix


def perm_matrix_to_perm_list(perm_matrix):
    return perm_matrix.nonzero()[:, 1]


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


def synchronize_permutations(permutations):
    ref_perms = permutations["a"]["b"]
    perm_names = list(ref_perms.keys())

    sync_permutations = {
        "a": {"b": {perm_name: None for perm_name in perm_names}, "c": {perm_name: None for perm_name in perm_names}},
        "b": {"c": {perm_name: None for perm_name in perm_names}, "a": {perm_name: None for perm_name in perm_names}},
        "c": {"a": {perm_name: None for perm_name in perm_names}, "b": {perm_name: None for perm_name in perm_names}},
    }

    # (a, b), (a, c), (b, c), ...
    combinations = list(itertools.permutations(permutations.keys(), 2))

    for perm_name, perm in ref_perms.items():
        perm_matrices = {
            comb: perm_list_to_perm_matrix(permutations[comb[0]][comb[1]][perm_name]) for comb in combinations
        }
        sync_perm_matrices = {comb: None for comb in combinations}

        n = len(perm)
        uber_matrix = torch.zeros((n * 3, n * 3))

        uber_matrix[:n, :n] = torch.eye(n)
        uber_matrix[n : n * 2, n : n * 2] = torch.eye(n)
        uber_matrix[n * 2 :, n * 2 :] = torch.eye(n)

        order = {"a": 0, "b": 1, "c": 2}
        for comb, perm_matrix in perm_matrices.items():
            uber_matrix[
                n * order[comb[0]] : n * (order[comb[0]] + 1), n * order[comb[1]] : n * (order[comb[1]] + 1)
            ] = perm_matrix

        sync_matrix = optimize_synchronization(uber_matrix, n)

        for comb in combinations:
            new_perm_matrix_comb = sync_matrix[
                n * order[comb[0]] : n * (order[comb[0]] + 1), n * order[comb[1]] : n * (order[comb[1]] + 1)
            ]
            assert is_valid_permutation_matrix(new_perm_matrix_comb)
            sync_perm_matrices[comb] = new_perm_matrix_comb

            sync_permutations[comb[0]][comb[1]][perm_name] = perm_matrix_to_perm_list(new_perm_matrix_comb)

        # permutations are orthogonal, hence their inverse is their transpose
        for comb in combinations:
            inv_comb = (comb[1], comb[0])
            assert torch.all(sync_perm_matrices[comb] == sync_perm_matrices[inv_comb].T)

    return sync_permutations


@hydra.main(config_path=str(PROJECT_ROOT / "conf/matching"), config_name="match_many")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
