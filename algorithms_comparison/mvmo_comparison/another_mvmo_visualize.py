import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.use("TkAgg")
x = [num / 100 for num in range(101)]


def another_implementation_mapping_transformation(xi_star, x_bar, s_old):
    if xi_star < 0.5:
        s_new = s_old / (1 - x_bar)
        hm = x_bar - x_bar / (0.5 * s_new + 1)
        hf = x_bar * (1 - np.exp(-xi_star * s_new))
        hc = (x_bar - hm) * 5 * xi_star
        xi_new = hf + hc
    else:
        s_new = s_old / x_bar
        hm = (1 - x_bar) / (0.5 * s_new + 1)
        hb = (1 - x_bar) / ((1 - xi_star) * s_new + 1) + x_bar
        hc = hm * 5 * (1 - xi_star)
        xi_new = hb - hc
    return xi_new


def another_implementation_mapping_transformation_fixed(xi_star, x_bar, s_old):
    hx = x_bar * (1 - np.exp(-1 * xi_star * s_old)) + (1 - x_bar) * np.exp(
        (xi_star - 1) * s_old
    )
    h0 = x_bar * (1 - np.exp(-1 * 0 * s_old)) + (1 - x_bar) * np.exp((0 - 1) * s_old)
    h1 = x_bar * (1 - np.exp(-1 * 1 * s_old)) + (1 - x_bar) * np.exp((1 - 1) * s_old)
    xi_new = hx + (1 - h1 + h0) * xi_star - h0
    return xi_new


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
                another_implementation_mapping_transformation(
                    random_xi, mean, factor_1
                )
                # another_implementation_mapping_transformation(random_xi, mean, factor_1)
                for random_xi in x
            ],
            color,
            label=label,
            linestyle=linestyle,
        )

    plot(ax, ylabel, xlabel, title)


def plot_transformation():

    means = [0.001, 0.25, 0.5, 0.75, 0.999]
    consts_means = [0.5] * 5

    shaping_scaling_factors = [0, 5, 10, 20, 50]
    consts_shaping_scaling_factors = [10] * 5

    consts_linestyles = ["solid"] * 5

    colors = ["cyan", "purple", "green", "red", "blue"]

    x_overline = r'$\bar{x}$'
    means_labels = [f"{x_overline} = {mean}" for mean in means]
    shaping_scaling_factors_labels = [
        f"s = {factor}" for factor in shaping_scaling_factors
    ]

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


if __name__ == "__main__":
    plot_transformation()
