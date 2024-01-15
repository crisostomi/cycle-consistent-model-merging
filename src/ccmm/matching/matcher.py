import torch

from ccmm.matching.frank_wolfe_matching import frank_wolfe_weight_matching
from ccmm.matching.permutation_spec import PermutationSpec
from ccmm.matching.quadratic_matching import quadratic_weight_matching
from ccmm.matching.synchronized_matching import synchronized_weight_matching
from ccmm.matching.weight_matching import LayerIterationOrder, weight_matching


class Matcher:
    def __init__(self, name, permutation_spec: PermutationSpec):
        self.name = name
        self.permutation_spec = permutation_spec

    def __call__(self, *args, **kwargs):
        pass


class DummyMatcher(Matcher):
    def __init__(self, name, permutation_spec: PermutationSpec):
        super().__init__(name, permutation_spec)

    def __call__(self, fixed, permutee):
        perm_sizes = {
            p: fixed[params_and_axes[0][0]].shape[params_and_axes[0][1]]
            for p, params_and_axes in self.permutation_spec.perm_to_axes.items()
        }

        permutation_indices = {p: torch.arange(n) for p, n in perm_sizes.items()}

        return permutation_indices


class GitRebasinMatcher(Matcher):
    def __init__(
        self,
        name,
        permutation_spec: PermutationSpec,
        max_iter=100,
        layer_iteration_order: LayerIterationOrder = LayerIterationOrder.RANDOM,
    ):
        super().__init__(name, permutation_spec)
        self.max_iter = max_iter
        self.layer_iteration_order = layer_iteration_order

    def __call__(self, fixed, permutee):
        permutation_indices = weight_matching(
            ps=self.permutation_spec,
            fixed=fixed,
            permutee=permutee,
            max_iter=self.max_iter,
            layer_iteration_order=self.layer_iteration_order,
        )

        return permutation_indices


class QuadraticMatcher(Matcher):
    def __init__(self, name, permutation_spec: PermutationSpec, max_iter=100, alternate_diffusion_params=None):
        super().__init__(name, permutation_spec)
        self.max_iter = max_iter
        self.alternate_diffusion_params = alternate_diffusion_params

    def __call__(self, fixed, permutee):
        permutation_indices = quadratic_weight_matching(
            ps=self.permutation_spec,
            fixed=fixed,
            permutee=permutee,
            max_iter=self.max_iter,
            alternate_diffusion_params=self.alternate_diffusion_params,
        )

        return permutation_indices


class AlternatingDiffusionMatcher(Matcher):
    def __init__(self, name, permutation_spec: PermutationSpec, max_iter=100, alternate_diffusion_params=None):
        super().__init__(name, permutation_spec)
        self.max_iter = max_iter
        self.alternate_diffusion_params = alternate_diffusion_params

    def __call__(self, fixed, permutee):
        permutation_indices = weight_matching(
            ps=self.permutation_spec,
            fixed=fixed,
            permutee=permutee,
            max_iter=self.max_iter,
            alternate_diffusion_params=self.alternate_diffusion_params,
        )

        return permutation_indices


class SynchronizedMatcher(Matcher):
    def __init__(self, name, permutation_spec, max_iter=100, sync_method="stiefel"):
        super().__init__(name, permutation_spec)
        self.max_iter = max_iter
        self.sync_method = sync_method

    def __call__(self, models, symbols, combinations):
        permutation_indices = synchronized_weight_matching(
            models=models,
            ps=self.permutation_spec,
            method=self.sync_method,
            symbols=symbols,
            combinations=combinations,
            max_iter=self.max_iter,
        )

        return permutation_indices


class FrankWolfeMatcher(Matcher):
    def __init__(self, name, permutation_spec: PermutationSpec, max_iter=100):
        super().__init__(name, permutation_spec)
        self.max_iter = max_iter

    def __call__(self, fixed, permutee):
        permutation_indices = frank_wolfe_weight_matching(
            ps=self.permutation_spec, fixed=fixed, permutee=permutee, max_iter=self.max_iter
        )

        return permutation_indices
