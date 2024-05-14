import json

import matplotlib.pyplot as plt
import numpy as np


class Palette(dict):
    def __init__(self, path):

        with open(path, "r") as f:
            color_palette = json.load(f)

        # mapping = {
        #     "Burnt sienna": "red",
        #     "Cambridge blue": "green",
        #     "Delft Blue": "blue",
        #     "Eggshell": "white",
        #     "Sunset": "yellow",
        # }

        mapping = {
            "Bittersweet shimmer": "light red",
            "Persian green": "green",
            "Saffron": "yellow",
            "Charcoal": "dark blue",
            "Burgundy": "red",
            "Burnt sienna": "orange",
            "Eggplant": "violet",
            "Sandy brown": "light orange",
        }

        for color, hex_code in color_palette.items():
            self[mapping[color]] = hex_code

    def get_colors(self, n):
        return list(self.values())[:n]


def plot_interpolation_results(
    lambdas,
    results_naive,
    results_clever,
    results_cleverer=None,
    metric_to_plot="acc",
):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(
        lambdas,
        results_naive[f"train_{metric_to_plot}"],
        linestyle="dashed",
        color="tab:blue",
        alpha=0.5,
        linewidth=2,
        label="Train, naïve interp.",
    )
    ax.plot(
        lambdas,
        results_naive[f"test_{metric_to_plot}"],
        linestyle="dashed",
        color="tab:orange",
        alpha=0.5,
        linewidth=2,
        label="Test, naïve interp.",
    )
    ax.plot(
        lambdas,
        results_clever[f"train_{metric_to_plot}"],
        linestyle="solid",
        color="tab:blue",
        linewidth=2,
        label="Train, permuted interp.",
    )
    ax.plot(
        lambdas,
        results_clever[f"test_{metric_to_plot}"],
        linestyle="solid",
        color="tab:orange",
        linewidth=2,
        label="Test, permuted interp.",
    )

    if results_cleverer is not None:
        ax.plot(
            lambdas,
            results_cleverer[f"train_{metric_to_plot}"],
            linestyle="dotted",
            color="tab:blue",
            linewidth=2,
            label="Train, synchronized perm interp.",
        )
        ax.plot(
            lambdas,
            results_cleverer[f"test_{metric_to_plot}"],
            linestyle="dotted",
            color="tab:orange",
            linewidth=2,
            label="Test, synchronized perm interp.",
        )
    ax.set_xlabel(r"$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel(metric_to_plot)
    # TODO label x=0 tick as \theta_1, and x=1 tick as \theta_2
    ax.set_title(f"{metric_to_plot} between the two models")
    ax.legend(loc="lower right", framealpha=0.5)
    fig.tight_layout()
    return fig


def decimal_to_rgb_color(decimal_value, cmap="viridis"):
    """
    Convert a decimal value (between 0 and 1) to the corresponding RGB color in the given colormap.
    """
    if not (0 <= decimal_value <= 1):
        raise ValueError("decimal_value should be between 0 and 1 inclusive.")

    colormap = plt.get_cmap(cmap)
    color = colormap(decimal_value)[:3]

    color = [round(c, 2) for c in color]
    return tuple(color)


def adjust_cmap_alpha(cmap, alpha=1.0):
    # Get the colormap colors
    colors = cmap(np.arange(cmap.N))

    # Set the alpha value
    colors[:, -1] = alpha

    # Create a new colormap with the modified colors
    new_cmap = plt.matplotlib.colors.ListedColormap(colors)
    return new_cmap


def rgba_to_rgb(rgba, background=(1, 1, 1)):
    print(rgba)
    """Convert an RGBA color to an RGB color, blending over a specified background color."""
    return [rgba[i] * rgba[3] + background[i] * (1 - rgba[3]) for i in range(3)]
