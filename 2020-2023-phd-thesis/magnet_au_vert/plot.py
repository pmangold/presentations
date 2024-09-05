import os
import pickle
import seaborn as sns
import numpy as np

from pcoptim import LeastSquares, Logistic

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D

from pydaha.plot import plot_from_file, plot_markers_from_file

final=True

palette=sns.color_palette("rocket")



matplotlib.rcParams['figure.figsize'] = (4, 3)
matplotlib.rcParams['lines.markersize'] = 16 #12
matplotlib.rcParams['lines.linewidth'] = 6 #6
matplotlib.rcParams['legend.fontsize'] = 13 #20
matplotlib.rcParams['xtick.major.size'] = 15
matplotlib.rcParams['ytick.major.size'] = 15
matplotlib.rcParams['axes.labelsize'] = 22
matplotlib.rcParams['xtick.labelsize'] = 15
matplotlib.rcParams['ytick.labelsize'] = 15
matplotlib.rcParams['lines.markeredgewidth'] = 2

base_dir = "./results_to_plot/"

from problems import losses_desc, datasets_size

with open("optimals.pickle", "rb") as f:
    optimals = pickle.load(f)


plot_legend = list(datasets_size.keys()) # { "houses_norm", "lasso",  "correlated_classif_1000_100_0.5_raw" }

for dataset in datasets_size:
    n, p = datasets_size[dataset][0], datasets_size[dataset][1]

    reg = losses_desc[dataset][3]

    fig, ax = plt.subplots()

    ref = optimals[dataset][1]
    ref_norm = np.linalg.norm(optimals[dataset][0], ord=1)

    plot_from_file(base_dir + dataset + "/cd/full/",
                   ref=ref, ax=ax, color=palette[0])#, marker="o")
    plot_from_file(base_dir + dataset + "/sgd/full/",
                   ref=ref, ax=ax, color=palette[1])#, marker="+")
    plot_from_file(base_dir + dataset + "/gcd_gsr/full/",
                   ref=ref, ax=ax, color=palette[2])#, marker="^")

    xticks = ax.get_xticks()
    xlim1, xlim2 = ax.get_xlim()
    xlim1 = xlim1 if xlim1 > 0 else 0

    if xlim2 < 10:
        xticks = np.arange(int(xlim1), int(xlim2)+1)
    else:
        xticks = np.arange(int(xlim1), int(xlim2)+1, 3)
    ax.set_xticks(xticks)

    plot_markers_from_file(base_dir + dataset + "/cd/full/",
                                  xticks=xticks,
                                  ref=ref, ax=ax, color=palette[0], marker="o")
    plot_markers_from_file(base_dir + dataset + "/sgd/full/",
                                  xticks=xticks,
                                  ref=ref, ax=ax, color=palette[1], marker="^")
    plot_markers_from_file(base_dir + dataset + "/gcd_gsr/full/",
                                   xticks=xticks,
                                   ref=ref, ax=ax, color=palette[2], marker="X")

    hdls = [Line2D([0],[0],marker='o',color=palette[0]),
            Line2D([0],[0],marker='^',color=palette[1]),
            Line2D([0],[0],marker='X',color=palette[2])]


    plt.ylabel("Relative Error", fontsize=13)
    plt.xlabel("Passes on data", fontsize=13)
    plt.yscale("log")

    # else:
    plt.legend(hdls, ["DP-CD", "DP-SGD", "DP-GCD"], loc="upper right")

    if final:
        plt.savefig("plots_final/" + dataset + ".pdf", bbox_inches="tight")
    else:
        plt.savefig("plots/" + dataset + ".pdf", bbox_inches="tight")
