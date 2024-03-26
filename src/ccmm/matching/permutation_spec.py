import logging
from collections import defaultdict
from functools import partial
from typing import NamedTuple

pylogger = logging.getLogger(__name__)


def conv_axes(name, p_rows, p_cols):
    return {
        f"{name}.weight": (
            p_rows,
            p_cols,
            None,
            None,
        )
    }


def layernorm_axes(name, p):
    return {f"{name}.weight": (p,), f"{name}.bias": (p,)}


def batchnorm_axes(name, p):
    return {
        f"{name}.weight": (p,),
        f"{name}.bias": (p,),
        f"{name}.running_mean": (p,),
        f"{name}.running_var": (p,),
        f"{name}.num_batches_tracked": (None,),
    }


def dense_layer_axes(name, p_rows, p_cols, bias=True):
    return {f"{name}.weight": (p_rows, p_cols), f"{name}.bias": (p_rows,)}


def easyblock_axes(name, p, norm_layer="ln"):
    """Easy blocks that use a residual connection, without any change in the number of channels."""
    norm_axes = layernorm_axes if norm_layer == "ln" else batchnorm_axes

    return {
        **conv_axes(f"{name}.conv1", p_rows=f"P_{name}_inner", p_cols=p),
        **norm_axes(f"{name}.bn1.layer_norm", f"P_{name}_inner"),
        **conv_axes(f"{name}.conv2", p_rows=p, p_cols=f"P_{name}_inner"),
        **norm_axes(f"{name}.bn2.layer_norm", p),
    }


def shortcut_block_axes(name, p_rows, p_cols, norm_layer="ln"):
    """This is for blocks that use a residual connection, but change the number of channels via a Conv."""
    norm_axes = layernorm_axes if norm_layer == "ln" else batchnorm_axes

    return {
        **conv_axes(f"{name}.conv1", p_rows=f"P_{name}_inner", p_cols=p_cols),
        **norm_axes(f"{name}.bn1.layer_norm", f"P_{name}_inner"),
        **conv_axes(f"{name}.conv2", p_rows=p_rows, p_cols=f"P_{name}_inner"),
        **norm_axes(f"{name}.bn2.layer_norm", p_rows),
        **conv_axes(f"{name}.shortcut.0", p_rows=p_rows, p_cols=p_cols),
        **norm_axes(f"{name}.shortcut.1.layer_norm", p_rows),
    }


class PermutationSpec(NamedTuple):
    # maps permutation matrices to the layers they permute, expliciting the axis they act on
    perm_to_layers_and_axes: dict

    # maps layers to permutations: if a layer has k dimensions, it maps to a permutation matrix (or None) for each dimension
    layer_and_axes_to_perm: dict


class PermutationSpecBuilder:
    def __init__(self) -> None:
        pass

    def create_permutation_spec(self) -> list:
        pass

    def permutation_spec_from_axes_to_perm(self, axes_to_perm: dict) -> PermutationSpec:
        perm_to_axes = defaultdict(list)

        for wk, axis_perms in axes_to_perm.items():
            for axis, perm in enumerate(axis_perms):
                if perm is not None:
                    perm_to_axes[perm].append((wk, axis))

        return PermutationSpec(perm_to_layers_and_axes=dict(perm_to_axes), layer_and_axes_to_perm=axes_to_perm)


class MLPPermutationSpecBuilder(PermutationSpecBuilder):
    def __init__(self, num_hidden_layers: int):
        self.num_hidden_layers = num_hidden_layers

    def create_permutation_spec(self) -> PermutationSpec:
        L = self.num_hidden_layers
        assert L >= 1

        axes_to_perm = {
            "layer0.weight": ("P_0", None),
            **{f"layer{i}.weight": (f"P_{i}", f"P_{i-1}") for i in range(1, L)},
            **{f"layer{i}.bias": (f"P_{i}",) for i in range(L)},
            f"layer{L}.weight": (None, f"P_{L-1}"),
            f"layer{L}.bias": (None,),
        }

        return self.permutation_spec_from_axes_to_perm(axes_to_perm)


class ResNet20PermutationSpecBuilder(PermutationSpecBuilder):
    def __init__(self, norm_layer="ln") -> None:
        self.norm_layer = norm_layer

    def create_permutation(self) -> PermutationSpec:
        norm_axes_fn = layernorm_axes if self.norm_layer == "ln" else batchnorm_axes
        easyblock_fn = partial(easyblock_axes, norm_layer=self.norm_layer)
        shortcut_block_fn = partial(shortcut_block_axes, norm_layer=self.norm_layer)

        axes_to_perm = {
            # initial conv, only permute its rows as the columns are the input channels
            **conv_axes("conv1", p_rows="P_bg0", p_cols=None),
            # batch norm after initial conv
            **norm_axes_fn("bn1.layer_norm", p="P_bg0"),
            ##########
            # layer 1
            **easyblock_fn("blockgroup1.block1", p="P_bg0"),
            **easyblock_fn(
                "blockgroup1.block2",
                p="P_bg0",
            ),
            **easyblock_fn("blockgroup1.block3", p="P_bg0"),
            ##########
            # layer 2
            **shortcut_block_fn("blockgroup2.block1", p_rows="P_bg1", p_cols="P_bg0"),
            **easyblock_fn(
                "blockgroup2.block2",
                p="P_bg1",
            ),
            **easyblock_fn("blockgroup2.block3", p="P_bg1"),
            ##########
            # layer 3
            **shortcut_block_fn("blockgroup3.block1", p_rows="P_bg2", p_cols="P_bg1"),
            **easyblock_fn(
                "blockgroup3.block2",
                p="P_bg2",
            ),
            **easyblock_fn("blockgroup3.block3", p="P_bg2"),
            ###########
            # output layer, only permute its columns as the rows are the output channels
            **dense_layer_axes("linear", p_rows=None, p_cols="P_bg2"),
        }

        return self.permutation_spec_from_axes_to_perm(axes_to_perm)


class ResNet50PermutationSpecBuilder(PermutationSpecBuilder):
    def __init__(self) -> None:
        pass

    def create_permutation(self) -> PermutationSpec:
        # TODO: invert conv and batch norm as in ResNet20

        return self.permutation_spec_from_axes_to_perm(
            {
                **conv_axes("conv1", p_rows="P_bg0", p_cols=None),
                **layernorm_axes("bn1", "P_bg0"),
                **shortcut_block_axes("layer1.0", "P_bg1", "P_bg0"),
                **easyblock_axes(
                    "layer1.1",
                    "P_bg1",
                ),
                **easyblock_axes("layer1.2", "P_bg1"),
                **easyblock_axes("layer1.3", "P_bg1"),
                **easyblock_axes("layer1.4", "P_bg1"),
                **easyblock_axes("layer1.5", "P_bg1"),
                **easyblock_axes("layer1.6", "P_bg1"),
                **easyblock_axes("layer1.7", "P_bg1"),
                **shortcut_block_axes("layer2.0", "P_bg2", "P_bg1"),
                **easyblock_axes(
                    "layer2.1",
                    "P_bg2",
                ),
                **easyblock_axes("layer2.2", "P_bg2"),
                **easyblock_axes("layer2.3", "P_bg2"),
                **easyblock_axes("layer2.4", "P_bg2"),
                **easyblock_axes("layer2.5", "P_bg2"),
                **easyblock_axes("layer2.6", "P_bg2"),
                **easyblock_axes("layer2.7", "P_bg2"),
                **shortcut_block_axes("layer3.0", "P_bg3", "P_bg2"),
                **easyblock_axes(
                    "layer3.1",
                    "P_bg3",
                ),
                **easyblock_axes("layer3.2", "P_bg3"),
                **easyblock_axes("layer3.3", "P_bg3"),
                **easyblock_axes("layer3.4", "P_bg3"),
                **easyblock_axes("layer3.5", "P_bg3"),
                **easyblock_axes("layer3.6", "P_bg3"),
                **easyblock_axes("layer3.7", "P_bg3"),
                **layernorm_axes("out_bn", "P_bg3"),
                **dense_layer_axes("linear", p_rows=None, p_cols="P_bg3"),
            }
        )


class VGG16PermutationSpecBuilder(PermutationSpecBuilder):
    def __init__(self) -> None:
        pass

    def create_permutation(self, model=None) -> PermutationSpec:
        layers_with_conv = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        layers_with_conv_b4 = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]
        layers_with_bn = [4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]

        axes_to_perm = {
            # first features
            "embedder.0.weight": ("P_Conv_0", None, None, None),
            "embedder.0.bias": ("P_Conv_0", None),
            "embedder.1.layer_norm.weight": ("P_Conv_0", None),
            "embedder.1.layer_norm.bias": ("P_Conv_0", None),
            **{
                f"embedder.{layers_with_conv[i]}.weight": (
                    f"P_Conv_{layers_with_conv[i]}",
                    f"P_Conv_{layers_with_conv_b4[i]}",
                    None,
                    None,
                )
                for i in range(len(layers_with_conv))
            },
            **{f"embedder.{i}.bias": (f"P_Conv_{i}",) for i in layers_with_conv + [0]},
            # bn
            **{
                f"embedder.{layers_with_bn[i]}.layer_norm.weight": (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_bn))
            },
            **{
                f"embedder.{layers_with_bn[i]}.layer_norm.bias": (f"P_Conv_{layers_with_conv[i]}", None)
                for i in range(len(layers_with_bn))
            },
            **dense_layer_axes("classifier.0", p_rows="P_Dense_0", p_cols="P_Conv_40", bias=True),
            # skipping 1 and 3 as they are ReLUs
            **dense_layer_axes("classifier.2", p_rows="P_Dense_1", p_cols="P_Dense_0", bias=True),
            **dense_layer_axes("classifier.4", p_rows=None, p_cols="P_Dense_1", bias=True),
        }

        return self.permutation_spec_from_axes_to_perm(axes_to_perm)
