import logging
from collections import defaultdict
from typing import NamedTuple

pylogger = logging.getLogger(__name__)


def conv_axes(name, p_in, p_out):
    return {
        f"{name}.weight": (
            p_out,
            p_in,
            None,
            None,
        )
    }


def norm_layer_axes(name, p):
    return {f"{name}.weight": (p,), f"{name}.bias": (p,)}


def dense_layer_axes(name, p_in, p_out, bias=True):
    # it's (p_in , p_out) in git-rebasin (due to jax)
    return {f"{name}.weight": (p_out, p_in), f"{name}.bias": (p_out,)}


def easyblock_axes(name, p):
    """Easy blocks that use a residual connection, without any change in the number of channels."""
    return {
        **conv_axes(f"{name}.conv1", p, f"P_{name}_inner"),
        **norm_layer_axes(f"{name}.bn1", p),
        **conv_axes(f"{name}.conv2", f"P_{name}_inner", p),
        **norm_layer_axes(f"{name}.bn2", p),
    }


def shortcut_block_axes(name, p_in, p_out):
    """This is for blocks that use a residual connection, but change the number of channels via a Conv."""
    return {
        **conv_axes(f"{name}.conv1", p_in, f"P_{name}_inner"),
        **norm_layer_axes(f"{name}.bn1", f"P_{name}_inner"),
        **conv_axes(f"{name}.conv2", f"P_{name}_inner", p_out),
        **norm_layer_axes(f"{name}.bn2", p_out),
        **conv_axes(f"{name}.shortcut.0", p_in, p_out),
        **norm_layer_axes(f"{name}.shortcut.1", p_out),
    }


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


class PermutationSpecBuilder:
    def __init__(self) -> None:
        pass

    def create_permutation(self) -> list:
        pass

    def permutation_spec_from_axes_to_perm(self, axes_to_perm: dict) -> PermutationSpec:
        perm_to_axes = defaultdict(list)

        for wk, axis_perms in axes_to_perm.items():
            for axis, perm in enumerate(axis_perms):
                if perm is not None:
                    perm_to_axes[perm].append((wk, axis))

        return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


class MLPPermutationSpecBuilder(PermutationSpecBuilder):
    def __init__(self, num_hidden_layers: int):
        self.num_hidden_layers = num_hidden_layers

    def create_permutation(self) -> PermutationSpec:
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
    def __init__(self) -> None:
        pass

    def create_permutation(self) -> PermutationSpec:
        axes_to_perm = {
            # initial conv
            **conv_axes("conv1", None, "P_bg0"),
            # batch norm after initial conv
            **norm_layer_axes("bn1", "P_bg0"),
            # layer 1
            **shortcut_block_axes("layer1.0", "P_bg0", "P_bg1"),
            **easyblock_axes(
                "layer1.1",
                "P_bg1",
            ),
            **easyblock_axes("layer1.2", "P_bg1"),
            # layer 2
            **shortcut_block_axes("layer2.0", "P_bg1", "P_bg2"),
            **easyblock_axes(
                "layer2.1",
                "P_bg2",
            ),
            **easyblock_axes("layer2.2", "P_bg2"),
            # layer 3
            **shortcut_block_axes("layer3.0", "P_bg2", "P_bg3"),
            **easyblock_axes(
                "layer3.1",
                "P_bg3",
            ),
            **easyblock_axes("layer3.2", "P_bg3"),
            **norm_layer_axes("out_bn", "P_bg3"),
            # output layer
            **dense_layer_axes("linear", "P_bg3", None),
        }

        return self.permutation_spec_from_axes_to_perm(axes_to_perm)


class ResNet50PermutationSpecBuilder(PermutationSpecBuilder):
    def __init__(self) -> None:
        pass

    def create_permutation(self) -> PermutationSpec:
        # TODO: invert conv and batch norm as in ResNet20

        return self.permutation_spec_from_axes_to_perm(
            {
                **conv_axes("conv1", None, "P_bg0"),
                **norm_layer_axes("bn1", "P_bg0"),
                **shortcut_block_axes("layer1.0", "P_bg0", "P_bg1"),
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
                **shortcut_block_axes("layer2.0", "P_bg1", "P_bg2"),
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
                **shortcut_block_axes("layer3.0", "P_bg2", "P_bg3"),
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
                **norm_layer_axes("out_bn", "P_bg3"),
                **dense_layer_axes("linear", "P_bg3", None),
            }
        )


class VGG16PermutationSpecBuilder(PermutationSpecBuilder):
    def __init__(self) -> None:
        pass

    def create_permutation(self) -> PermutationSpec:
        layers_with_conv = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37, 40]
        layers_with_conv_b4 = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]
        layers_with_bn = [4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]
        return self.permutation_spec_from_axes_to_perm(
            {
                # first features
                "features.0.weight": ("P_Conv_0", None, None, None),
                "features.1.weight": ("P_Conv_0", None),
                "features.1.bias": ("P_Conv_0", None),
                "features.1.running_mean": ("P_Conv_0", None),
                "features.1.running_var": ("P_Conv_0", None),
                "features.1.num_batches_tracked": (),
                **{
                    f"features.{layers_with_conv[i]}.weight": (
                        f"P_Conv_{layers_with_conv[i]}",
                        f"P_Conv_{layers_with_conv_b4[i]}",
                        None,
                        None,
                    )
                    for i in range(len(layers_with_conv))
                },
                **{f"features.{i}.bias": (f"P_Conv_{i}",) for i in layers_with_conv + [0]},
                # bn
                **{
                    f"features.{layers_with_bn[i]}.weight": (f"P_Conv_{layers_with_conv[i]}", None)
                    for i in range(len(layers_with_bn))
                },
                **{
                    f"features.{layers_with_bn[i]}.bias": (f"P_Conv_{layers_with_conv[i]}", None)
                    for i in range(len(layers_with_bn))
                },
                **{
                    f"features.{layers_with_bn[i]}.running_mean": (f"P_Conv_{layers_with_conv[i]}", None)
                    for i in range(len(layers_with_bn))
                },
                **{
                    f"features.{layers_with_bn[i]}.running_var": (f"P_Conv_{layers_with_conv[i]}", None)
                    for i in range(len(layers_with_bn))
                },
                **{f"features.{layers_with_bn[i]}.num_batches_tracked": () for i in range(len(layers_with_bn))},
                **dense_layer_axes("classifier", "P_Conv_40", "P_Dense_0", False),
            }
        )