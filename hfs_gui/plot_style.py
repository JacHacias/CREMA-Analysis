import matplotlib as mpl


PUBLICATION_STYLE = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.4,
    "axes.grid": False,
    "font.family": "serif",
    "font.size": 13,
    "axes.labelsize": 17,
    "axes.titlesize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 1.3,
    "ytick.major.width": 1.3,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "legend.frameon": False,
    "mathtext.fontset": "stix",
}


def apply_publication_style():
    mpl.rcParams.update(PUBLICATION_STYLE)


def style_axes(ax):
    ax.grid(False)
    ax.tick_params(direction="out", length=5, width=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
