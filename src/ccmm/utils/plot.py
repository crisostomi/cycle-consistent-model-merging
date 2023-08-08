import matplotlib.pyplot as plt


def plot_interpolation_results(
    lambdas,
    results_naive,
    results_clever,
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
    ax.set_xlabel(r"$\lambda$")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Model $A$", "Model $B$"])
    ax.set_ylabel("Accuracy")
    # TODO label x=0 tick as \theta_1, and x=1 tick as \theta_2
    ax.set_title(f"{metric_to_plot} between the two models")
    ax.legend(loc="lower right", framealpha=0.5)
    fig.tight_layout()
    return fig
