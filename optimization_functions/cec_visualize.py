import matplotlib.pyplot as plt
import numpy as np
from cec.CEC2022 import cec2022_func

title = "bazowa"


def plot(func_num, option="surface"):
    cec_function = cec2022_func(func_num=func_num).values
    X = np.arange(-100, 100, 1)
    Y = np.arange(-100, 100, 1)
    X, Y = np.meshgrid(X, Y)
    Z = cec_function(np.vstack((X.flatten(), Y.flatten())))
    Z = Z.reshape(len(X), len(Y))

    if option == "surface":
        ax = plt.axes(projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x)")
        plt.title(title)
        # ax.view_init(elev=18, azim=-109)
        plt.show()
    elif option == "contourf":
        ax = plt.axes()
        # levels = np.logspace(math.log10(Z.min()), math.log10(Z.max()), 10)
        levels = np.linspace(Z.min(), Z.max(), 15)
        cs = ax.contourf(X, Y, Z, cmap="viridis", levels=levels)
        ax.contour(cs, colors="k", linewidths=0.5)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(title)
        ax.plot()
        plt.show()


if __name__ == "__main__":

    plot(4, "surface")
    plot(4, "contourf")
