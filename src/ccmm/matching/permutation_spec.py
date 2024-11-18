import copy
import logging
from collections import defaultdict
from functools import partial
from typing import NamedTuple

import torch

from ccmm.utils.perm_graph import get_perm_dict, graph_permutations_to_layer_and_axes_to_perm

pylogger = logging.getLogger(__name__)


def conv_axes(name, p_rows, p_cols, bias=False):
    axes = {
        f"{name}.weight": (
            p_rows,
            p_cols,
            None,
            None,
        )
    }
    if bias:
        axes[f"{name}.bias"] = (p_rows,)

    return axes


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

    def create_permutation_spec(self, **kwargs) -> list:
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

    def create_permutation_spec(self, *args, **kwargs) -> PermutationSpec:
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

    def create_permutation_spec(self, *args, **kwargs) -> PermutationSpec:
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

    def create_permutation(self, **kwargs) -> PermutationSpec:
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


class ViTPermutationSpecBuilder(PermutationSpecBuilder):
    def __init__(self, depth) -> None:
        self.depth = depth

    def create_permutation_spec(self, **kwargs) -> PermutationSpec:

        axes_to_perm = {
            # layer norm
            "to_patch_embedding.to_patch_tokens.1.weight": (None,),  # (3*c*16)
            "to_patch_embedding.to_patch_tokens.1.bias": (None,),  # (3*c*16)
            # linear
            "to_patch_embedding.to_patch_tokens.2.weight": ("P_in", None),  # (dim, 3*c*16)
            "to_patch_embedding.to_patch_tokens.2.bias": ("P_in",),  # dim
            "pos_embedding": (None, None, "P_in"),  # (1, p+1, dim) probably P_transf_in or its own P
            "cls_token": (None, None, "P_in"),  # (1, 1, dim) probably P_transf_in or its own P
            **transformer_block_axes(self.depth, p_in="P_in", p_out="P_last"),
            # layer norm
            "mlp_head.0.weight": ("P_last",),  # (dim, )
            "mlp_head.0.bias": ("P_last",),  # (dim,)
            # linear
            "mlp_head.1.bias": (None,),  # (num_classes)
            "mlp_head.1.weight": (None, "P_last"),  # (num_classes, dim)
        }

        return self.permutation_spec_from_axes_to_perm(axes_to_perm)


def transformer_block_axes(depth, p_in, p_out):

    all_axes = {}

    for block_ind in range(depth):

        block_out = p_out if block_ind == depth - 1 else f"P{block_ind}_out"
        block_in = p_in if block_ind == 0 else f"P{block_ind-1}_out"

        block_axes = {
            # Attention
            ## layer norm
            f"transformer.layers.{block_ind}.0.norm.weight": (block_in,),  # (dim,)
            f"transformer.layers.{block_ind}.0.norm.bias": (block_in,),  # (dim,)
            f"transformer.layers.{block_ind}.0.temperature": (None,),  # (,)
            # HEADS
            f"transformer.layers.{block_ind}.0.to_q.weight": (f"P{block_ind}_attn_QK", block_in),  # (head_dim, dim)
            f"transformer.layers.{block_ind}.0.to_k.weight": (f"P{block_ind}_attn_QK", block_in),  # (head_dim, dim)
            f"transformer.layers.{block_ind}.0.to_v.weight": (None, block_in),  # (head_dim, dim)
            # Attention out
            f"transformer.layers.{block_ind}.0.to_out.0.weight": (
                None,
                None,  # f"P{block_ind}_attn_V",
            ),  # (dim, dim)
            f"transformer.layers.{block_ind}.0.to_out.0.bias": (None,),  # (dim,)
            # shortcut
            f"transformer.layers.{block_ind}.1.identity": (block_in, None),  # (dim, dim) # WORKS
            # MLP
            ## layer norm
            f"transformer.layers.{block_ind}.2.net.0.weight": (None,),  # (dim, )
            f"transformer.layers.{block_ind}.2.net.0.bias": (None,),  # (dim,)
            ## linear 1
            f"transformer.layers.{block_ind}.2.net.1.weight": (
                f"P{block_ind}_mlp_out",
                None,
            ),  # (mlp_dim, dim)
            f"transformer.layers.{block_ind}.2.net.1.bias": (f"P{block_ind}_mlp_out",),  # (mlp_dim,)
            ## linear 2
            f"transformer.layers.{block_ind}.2.net.4.weight": (
                block_out,
                f"P{block_ind}_mlp_out",
            ),  # (dim, mlp_dim) # WORKS
            f"transformer.layers.{block_ind}.2.net.4.bias": (block_out,),  # (dim,) # WORKS
            # shortcut 2
            f"transformer.layers.{block_ind}.3.identity": (None, block_out),  # (dim, dim) # WORKS
        }

        all_axes.update(block_axes)

    return all_axes


class CNNPermutationSpecBuilder(PermutationSpecBuilder):
    def __init__(self) -> None:
        super().__init__()

    def create_permutation_spec(self, **kwargs) -> PermutationSpec:
        axes_to_perm = {
            **conv_axes("conv1", p_rows="P_conv1", p_cols=None, bias=True),
            **conv_axes("conv2", p_rows="P_conv2", p_cols="P_conv1", bias=True),
            "shortcut.identity": (None, "P_conv2"),
            **dense_layer_axes("fc1", p_rows="P_fc1", p_cols=None),
            **dense_layer_axes("fc2", p_rows=None, p_cols="P_fc1"),
        }

        return self.permutation_spec_from_axes_to_perm(axes_to_perm)


class AutoPermutationSpecBuilder(PermutationSpecBuilder):
    def create_permutation_spec(self, ref_model, **kwargs) -> list:
        x = torch.randn(1, 3, 256, 256)

        while hasattr(ref_model, "model"):
            ref_model = ref_model.model

        perm_dict, map_param_index, map_prev_param_index = get_perm_dict(ref_model, input=x)

        layer_and_axes_to_perm = graph_permutations_to_layer_and_axes_to_perm(
            ref_model, perm_dict, map_param_index, map_prev_param_index
        )

        return self.permutation_spec_from_axes_to_perm(layer_and_axes_to_perm)
