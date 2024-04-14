import matplotlib as mpl
import matplotlib.pyplot as plt

from evolutionary_algorithms.evolutionary.mutation import Mutation

mpl.use("TkAgg")
x = [num / 100 for num in range(101)]


def plot(ax, ylabel="", xlabel="", title=""):
    ax.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.show()


def prepare_plot(width=8, height=6):
    fig, ax = plt.subplots()
    fig.set_figwidth(width)
    fig.set_figheight(height)
    return ax


def visualize(
    means,
    shaping_scaling_factors_1,
    shaping_scaling_factors_2,
    colors,
    linestyles,
    labels,
    ylabel="",
    xlabel="",
    title="",
):
    ax = prepare_plot()
    for mean, factor_1, factor_2, color, label, linestyle in zip(
        means,
        shaping_scaling_factors_1,
        shaping_scaling_factors_2,
        colors,
        labels,
        linestyles,
    ):
        ax.plot(
            x,
            [
                Mutation.transformation(random_xi, mean, si1=factor_1, si2=factor_2)
                for random_xi in x
            ],
            color,
            label=label,
            linestyle=linestyle,
        )

    plot(ax, ylabel, xlabel, title)


def plot_transformation():

    means = [0, 0.25, 0.5, 0.75, 1.0]
    consts_means = [0.5] * 5

    shaping_scaling_factors = [0, 5, 10, 20, 50]
    consts_shaping_scaling_factors = [10] * 5

    consts_linestyles = ["solid"] * 5
    linestyles = ["dashed", "solid"]

    colors = ["cyan", "purple", "green", "red", "blue"]

    indexes = slice(2, 4)

    means_labels = [f"x` = {mean}" for mean in means]
    shaping_scaling_factors_labels = [
        f"s = {factor}" for factor in shaping_scaling_factors
    ]
    same_shaping_scaling_factors_labels = (
        f"si1 = si2 = {consts_shaping_scaling_factors[0]}"
    )
    different_shaping_scaling_factors_1_labels = f"si1 = {shaping_scaling_factors[indexes][0]}, si2 = {shaping_scaling_factors[indexes][1]}"
    different_shaping_scaling_factors_2_labels = f"si1 = {shaping_scaling_factors[indexes][1]}, si2 = {shaping_scaling_factors[indexes][0]}"

    # Effects of mean of dynamic population on the transformation function h
    visualize(
        means,
        consts_shaping_scaling_factors,
        consts_shaping_scaling_factors,
        colors,
        consts_linestyles,
        means_labels,
        "x",
        "x'",
        "Wpływ średniej wartości genu na funkcję mapującą\ndla s = 10",
    )

    # Effects of shaping scaling factor on the transformation function h
    visualize(
        consts_means,
        shaping_scaling_factors,
        shaping_scaling_factors,
        colors,
        consts_linestyles,
        shaping_scaling_factors_labels,
        "x",
        "x'",
        "Wpływ współczynnika skalującego na funkcję mapującą\ndla x` = 0.5",
    )

    # Effects of different shape factors si1 =/= si2
    visualize(
        consts_means[indexes],
        consts_shaping_scaling_factors[indexes],
        shaping_scaling_factors[indexes],
        colors[indexes],
        linestyles,
        [
            same_shaping_scaling_factors_labels,
            different_shaping_scaling_factors_1_labels,
        ],
        "xi",
        "random x",
        "effects of different shape factors si1 =/= si2 \nfor mean xi = 0.5",
    )
    visualize(
        consts_means[indexes],
        shaping_scaling_factors[indexes],
        consts_shaping_scaling_factors[indexes],
        colors[indexes],
        linestyles,
        [
            same_shaping_scaling_factors_labels,
            different_shaping_scaling_factors_2_labels,
        ],
        "xi",
        "random x",
        "effects of different shape factors si1 =/= si2 \nfor mean xi = 0.5",
    )


if __name__ == "__main__":
    plot_transformation()
