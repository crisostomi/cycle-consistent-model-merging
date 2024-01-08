import copy
import json
import logging
from pathlib import Path
from pydoc import locate
from typing import Any, Dict, List, Tuple

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from omegaconf import ListConfig
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from scipy.interpolate import interp1d
from scipy.misc import derivative

from nn_core.serialization import load_model

ModelParams = Dict[str, torch.Tensor]


pylogger = logging.getLogger(__name__)

MODEL_SEED_TO_SYMBOL = {
    1: "a",
    2: "b",
    3: "c",
    4: "d",
    5: "e",
    "dummy_a": "x",
    "dummy_b": "y",
    "dummy_c": "z",
}


def map_model_seed_to_symbol(seed):
    return MODEL_SEED_TO_SYMBOL[seed]


def flatten_params(model):
    return model.state_dict()


def calculate_global_radius(dist_matrix, k=5, target_percentage=0.8):
    """
    Calculate a global radius such that a target percentage of neurons have at least K neighbors.
    """
    num_neurons = dist_matrix.shape[0]
    max_radius = torch.max(dist_matrix)
    step = max_radius / 100  # Increment step for radius
    radius = 0

    while radius <= max_radius:
        count = 0
        for i in range(num_neurons):
            # Calculate distances
            neuron_distances = dist_matrix[i]

            # Count how many neurons are within the current radius
            neighbors_within_radius = torch.sum(neuron_distances <= radius).item() - 1  # Exclude the neuron itself

            # Check if the current neuron has at least K neighbors
            if neighbors_within_radius >= k:
                count += 1

        # Check if the current radius satisfies the condition for the target percentage of neurons
        if (count / num_neurons) >= target_percentage:
            return radius
        radius += step

    return radius


def plot_cumulative_density(neuron_dists, min_value, num_radii):
    max_dist = torch.max(neuron_dists)
    radii = torch.linspace(min_value, max_dist, num_radii)

    cumulative_counts = [torch.sum(neuron_dists <= radius) for radius in radii]

    # Normalize cumulative counts to get the fraction (percentage) of neighbors
    cumulative_counts_normalized = torch.tensor(cumulative_counts) / len(neuron_dists)

    # Interpolate the cumulative counts to compute derivative
    interpolated_counts = interp1d(radii, cumulative_counts_normalized, kind="cubic")

    print(interpolated_counts)
    # Compute the derivative (approximate local density)
    # Compute the derivative within the bounds of the interpolated function
    # Using a slightly reduced range to avoid boundary issues
    radius_min = radii[1] + 1e-1  # start slightly above the minimum to avoid boundary issue
    radius_max = radii[-2] - 1e-1  # end slightly below the maximum to avoid boundary issue
    fine_radii = torch.linspace(radius_min, radius_max, num_radii)
    density = [derivative(interpolated_counts, r, dx=1e-2) for r in fine_radii]

    # Plotting
    plt.figure(figsize=(12, 6))

    # Plot cumulative counts
    plt.subplot(1, 2, 1)
    plt.plot(radii, cumulative_counts_normalized, label="Cumulative Counts")
    plt.xlabel("Radius")
    plt.ylabel("Fraction of Neighbors")
    plt.title("Cumulative Counts vs Radius")
    plt.grid(True)
    plt.legend()

    # Plot derivative (density)
    plt.subplot(1, 2, 2)
    plt.plot(fine_radii, density, label="Density (Derivative)", color="orange")
    plt.xlabel("Radius")
    plt.ylabel("Density")
    plt.title("Density vs Radius")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def linear_interpolation(lam, t1, t2):
    t3 = copy.deepcopy(t2)
    for p in t1:
        t3[p] = (1 - lam) * t1[p] + lam * t2[p]
    return t3


def l2_norm_models(state_dict1, state_dict2):
    """Calculate the L2 norm of the difference between two state dictionaries."""
    diff_squared_sum = sum(torch.sum((state_dict1[key] - state_dict2[key]) ** 2) for key in state_dict1)
    return torch.sqrt(diff_squared_sum)


def average_models(model_params):
    return {k: torch.mean(torch.stack([p[k] for p in model_params]), dim=0) for k in model_params[0].keys()}


def get_checkpoint_callback(callbacks):
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            return callback


def build_callbacks(cfg: ListConfig, *args: Callback) -> List[Callback]:
    """Instantiate the callbacks given their configuration.

    Args:
        cfg: a list of callbacks instantiable configuration
        *args: a list of extra callbacks already instantiated

    Returns:
        the complete list of callbacks to use
    """
    callbacks: List[Callback] = list(args)

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def block(i, j, n):
    return slice(i * n, (i + 1) * n), slice(j * n, (j + 1) * n)


def load_model_from_info(model_info_path, seed=None, zipped=True):
    suffix = "" if seed is None else f"_{seed}.json"
    model_info_path_seed = model_info_path + suffix
    model_info = json.load(open(model_info_path_seed))
    model_class = locate(model_info["class"])

    suffix = ".zip" if zipped else ""
    model = load_model(model_class, checkpoint_path=Path(model_info["path"] + suffix))
    model.eval()

    return model


def save_permutations(permutations: Dict[str, Dict], path: str):
    """ """
    for source, targets in permutations.items():
        if targets is None:
            continue
        for target, source_target_perms in targets.items():
            if source_target_perms is None:
                continue
            for perm_name, perm in source_target_perms.items():
                if perm is None:
                    continue
                permutations[source][target][perm_name] = perm.tolist()

    with open(path, "w+") as f:
        json.dump(permutations, f)


def load_permutations(path):
    with open(path, "r") as f:
        permutations = json.load(f)

    for source, targets in permutations.items():
        for target, source_target_perms in targets.items():
            for perm_name, perm in source_target_perms.items():
                if perm is not None:
                    permutations[source][target][perm_name] = torch.tensor(perm)

    return permutations


class OnSaveCheckpointCallback(Callback):
    def on_save_checkpoint(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, checkpoint: Dict[str, Any]
    ) -> None:
        metadata = getattr(pl_module, "metadata", None)
        if metadata is not None:
            checkpoint["metadata"] = metadata


def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


def unravel_index(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> Tuple[torch.LongTensor, ...]:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (N,).
        shape: The targeted shape, (D,).

    Returns:
        A tuple of unraveled coordinate tensors of shape (D,).
    """

    coord = unravel_indices(indices, shape)
    return tuple(coord)
