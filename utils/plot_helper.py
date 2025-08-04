import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'legend.frameon': False,
    "font.size": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    'axes.titlesize': 9,
    'legend.fontsize': 8
})

fig_dir = 'figures'
os.system(f"mkdir -p {fig_dir}")

fig_size_default = {"width": 6.9}

snp_branch_args = {"markevery": 0.1, "marker": 'o',
                   "markersize": 3.5, "color": "k", "linewidth": 1}
hopf_branch_args = {"markevery": 0.1, "marker": '^',
                    "markersize": 4, "color": "grey", "linewidth": 1}

color_mut = "#d78e90"
color_mut_2 = "#a3c6e1"
color_wt = "#aed8c5"
color_mut_dark = "#d20026"
color_mut_2_dark = "#006b8d"
color_wt_dark = "#00b566"
current_color = "#6a3e2f"


def my_legend(ax, **kwargs):
    handles, labels = [
        *zip(*{l: h for h, l in zip(*ax.get_legend_handles_labels())}.items())][::-1]
    ax.legend(handles, labels, **kwargs)


def annotate_blended(ax, text, xy, xtrans=None, ytrans=None, **kwargs):
    from matplotlib.transforms import blended_transform_factory
    if (xtrans is None):
        xtrans = ax.transAxes
    if (ytrans is None):
        ytrans = ax.transAxes
    trans = blended_transform_factory(x_transform=xtrans, y_transform=ytrans)
    args = dict()
    args.update(**kwargs)
    return ax.annotate(text, xy, xycoords=trans, **args)


def share_axis(axes, x=False, y=False):
    if (np.ndim(axes) == 2):
        for k in range(axes.shape[0]):
            share_axis(axes[k], x, y)
        for k in range(axes.shape[1]):
            share_axis(axes[:, k], x, y)
        return
    if (y):
        for k in range(1, len(axes)):
            axes[k].sharey(axes[0])
    if (x):
        for k in range(1, len(axes)):
            axes[k].sharex(axes[0])


def align_labels(fig, shift=-0.08, x=0, y=0, axs=None):
    if axs is None:
        axs = fig.axes
    ax = axs[0]
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_ref = bbox.width
    height_ref = bbox.height
    for ax in axs:
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width = bbox.width
        height = bbox.height
        shift_cor_y = shift * width_ref / width
        shift_cor_x = shift * height_ref / height
        if y:
            ax.yaxis.set_label_coords(x=shift_cor_y, y=0.5)
        if x:
            ax.xaxis.set_label_coords(x=0.5, y=shift_cor_x)


def lettering(fig, pos, labels, subscript=False, fontsize=10, bold=True, **kwargs):
    axs = fig.axes
    x_ref, y_ref = pos
    ax = axs[0]
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_ref, height_ref = bbox.width, bbox.height
    bbox_props = dict(boxstyle="round", fc="w", ec="w", **kwargs)
    for idx, ax in enumerate(axs):
        if labels[idx]:
            bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width, height = bbox.width, bbox.height
            if x_ref < 0.5:
                x = x_ref * width_ref / width
            else:
                x = 1 - ((1 - x_ref) * width_ref / width)

            if y_ref < 0.5:
                y = y_ref * height_ref / height
            else:
                y = 1 - ((1 - y_ref) * height_ref / height)

            label = labels[idx]
            if subscript == 2:
                if bold:
                    label = fr'$\rm \mathbf{{{label[0]}_{label[1]}}}$'
                else:
                    label = fr'$\rm {{{label[0]}_{label[1]}}}$'
            else:
                if bold:
                    label = fr'$\rm \mathbf{{{label}}}$'
                else:
                    label = f'{label}'
            ax.text(x, y, label, transform=ax.transAxes,
                    bbox=bbox_props, fontsize=fontsize)
