# coding: utf-8
#
# TARGET:
#   Oracle bounds regarding fairness for majority voting
#


# import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib import cm as mpl_cm
# from matplotlib.collections import LineCollection
# import seaborn as sns
import numpy as np


# =====================================
# Preliminaries


# -------------------------------------
# Parameters


# mpl.use('Agg')  # set the 'backend'
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.family'] = 'Helvetica'  # 'Arial'
plt.rcParams['font.family'] = 'stixgeneral'

# plt.rc('text', usetex=True)
# plt.rcParams['font.sans-serif'] = ['SimHei']  # show chinese lbl
plt.rcParams['axes.unicode_minus'] = False  # show minus notations
plt.rcParams['pdf.fonttype'] = 42           # to avoid type 3 font
# plt.rcParams['ps.fonttype'] = 42


PLT_SHOW_OFF = False
PLT_SPINEOFF = True
PLT_LOCATION = 'best'
PLT_FRAMEBOX = False
# PLT_FRAMEBOX = True

PLT_T_LAYOUT = True
PLT_AX_STYLE = True
DTY_PLT = '.pdf'
PLT_T_LAYOUT = False


# For code evaluation
'''
DTY_PLT = '.png'
PLT_FRAMEBOX = True
PLT_SPINEOFF = False
# PLT_SHOW_OFF = True
# PLT_T_LAYOUT = False
# PLT_AX_STYLE = False
'''


# mpl.rcParams.update(
#     {
#         'text.usetex': True,
#         'font.family': 'stixgeneral',
#         'mathtext.fontset': 'stix',
#     }
# )


# -------------------------------------
# Help(er) functions


def _style_set_axis(ax, invt=False):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in')
    if invt:
        ax.invert_yaxis()  # x-y,矩阵模式
    return ax


def _style_set_fig(fig, siz=(2.85, 2.36)):
    fig.set_size_inches(*siz)
    # 裁剪后 (2.5, 1.9), instead of (5, 4)
    return fig


_setup_config = {
    'S-WS': (2.85, 2.36),
    'S-NT': (2.95, 2.44),  # nice
    'M-WS': (3.05, 2.52),  # nice
    'M-NT': (3.15, 2.61),  # default
    'L-WS': (3.45, 2.85),
    'L-NT': (5, 4),
    'extra': (7, 5.467),
    'L-ET': (6, 4.7)
    # 'extra': (9, 5.4)
}  # 'S/M/L' 'Wide/Narrow' 'Tall/Short'


def _setup_figsize(fig, figsize='S-NT', invt=False):
    figsize = _setup_config[figsize]
    fig.set_size_inches(*figsize)
    if PLT_AX_STYLE:
        _style_set_axis(fig.gca(), invt)
    return fig


def _setup_figshow(fig, figname):
    if PLT_SHOW_OFF:
        fig.show()
        return
    if PLT_T_LAYOUT:
        fig.tight_layout()
    fig.savefig("{}{}".format(figname, DTY_PLT),
                dpi=300, bbox_inches='tight')
    # fig.savefig(figname, dpi=300, bbox_inches='tight')
    # plt.close(fig)
    return


def _setup_locater(fig, base=0.05):
    # 把轴的刻度间隔设置为1，并存在变量里
    x_major_locator = plt.MultipleLocator(base)
    y_major_locator = plt.MultipleLocator(base)
    # ax为两条坐标轴的实例
    ax = fig.gca()
    # 把轴的主刻度设置为1的倍数
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    # 把轴的刻度范围设置为-到-
    return fig


def _set_quantile(p, low, high):
    # aka. fractile, percentile
    q_frac = (high - low) / abs(p)
    if p > 0:
        return high - q_frac
    elif p < 0:
        return low + q_frac
    return low + (high - low) / 2


# -------------------------------------


# =====================================
# draw_graph.py


_barh_kwargs = {
    'error_kw': {
        "ecolor": "0.12",
        "capsize": 5.6,
        'elinewidth': 1.1},
    'edgecolor': 'black',
    'lw': .7, }

_barh_fcolor = [
    tuple([i / 255 for i in [153, 50, 204]]),
    tuple([i / 255 for i in [204, 204, 254]]),
    tuple([0, 0.4470, 0.7410]),
    tuple([i / 255 for i in [135, 206, 250]]),
    tuple([0, 0.5019, 0.5020]),
    tuple([i / 255 for i in [64, 224, 208]]),
    tuple([i / 255 for i in [255, 250, 205]]),
    tuple([i / 255 for i in [240, 230, 140]]),
    tuple([i / 255 for i in [255, 127, 80]]),
    tuple([i / 255 for i in [255, 165, 0]]),
    tuple([i / 255 for i in [255, 20, 147]]),
]


def _setup_rgb_color(N, cname='GnBu_r'):
    # cmap = mpl_cm.get_cmap(cname)
    cmap = plt.colormaps.get_cmap(cname)

    norm = plt.Normalize(0, N)
    norm_y = norm(np.arange(N + 1))

    colors = cmap(norm_y)
    return colors, len(colors)


# https://stackoverflow.com/questions/76901874/userwarning-the-figure-layout-has-changed-to-tight-self-figure-tight-layouta
# https://github.com/mwaskom/seaborn/issues/3431
# https://stackoverflow.com/questions/47253462/matplotlib-mathtext-glyph-errors-in-tick-labels
