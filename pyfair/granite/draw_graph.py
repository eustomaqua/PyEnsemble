# coding: utf-8
#
# TARGET:
#   Oracle bounds regarding fairness for weighted voting
#


import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix

from pyfair.facil.utils_saver import elegant_print
from pyfair.marble.draw_hypos import (
    Friedman_test, Nememyi_posthoc_test)
from pyfair.facil.draw_prelim import (
    PLT_LOCATION, PLT_FRAMEBOX, _style_set_fig, _setup_config,
    _setup_figsize, _setup_figshow, _setup_locater,
    _set_quantile, _barh_kwargs, _barh_fcolor, _setup_rgb_color)


# =====================================
# Matlab plot


# ----------------------------------
# 散点图和拟合曲线，相关系数


def scatter_and_corr(X, Y, figname, figsize='M-WS',
                     annots=('X', 'Y')):
    # x-axis: objective    /propose, Risk_esti/div_x
    # y-axis: groundtruth  /compare, Risk_real

    # Correlation
    R = np.corrcoef(X, Y)[1, 0]
    key = "Correlation = %.4f" % R
    fig = plt.figure(figsize=_setup_config['L-NT'])
    plt.scatter(X, Y, label=key)          # fp1 =
    plt.xlabel(annots[0])
    plt.ylabel(annots[1])

    # Regression
    regr = np.polyfit(X, Y, deg=1)
    # slope, intercept = regr
    # return R, slope, intercept
    estimated = np.polyval(regr, X)
    plt.plot(X, estimated, "k-", lw=1.5)  # fp2 =

    plt.legend(loc=PLT_LOCATION, frameon=PLT_FRAMEBOX)
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# seaborn/regression.py
#     def lineplot(self, ax, kws):
# seaborn/algorithms.py
# seaborn/utils.py

def _sns_algo_bootstrap(*args, **kwargs):
    n_boots = 1000
    # seed = None  # units = seed = None
    # # args = (X, Y)
    # # Default keyword arguments
    func = kwargs.get("func", np.mean)
    n, boot_dist = len(args[0]), []

    for _ in range(int(n_boots)):  # for i in
        resampler = np.random.randint(0, n, n, dtype=np.intp)
        sample = [a.take(resampler, axis=0) for a in args]
        boot_dist.append(func(*sample))
    return np.array(boot_dist)


def _sns_sub_fit_poly(X, Y, grid, order=1):
    def reg_func(_x, _y):
        return np.polyval(np.polyfit(_x, _y, order), grid)

    yhat = reg_func(X, Y)
    yhat_boots = _sns_algo_bootstrap(X, Y, func=reg_func)
    return yhat, yhat_boots


def _sns_sub_fit_fast(X, Y, grid):
    def reg_func(_x, _y):
        return np.linalg.pinv(_x).dot(_y)

    X, Y = np.c_[np.ones(len(X)), X], Y
    grid = np.c_[np.ones(len(grid)), grid]  # (100,2)
    yhat = grid.dot(reg_func(X, Y))

    beta_boots = _sns_algo_bootstrap(X, Y, func=reg_func).T
    yhat_boots = grid.dot(beta_boots).T
    return yhat, yhat_boots


def _sns_line_fit_regs(ax, X, Y, order=1):
    x_min, x_max = ax.get_xlim()
    grid = np.linspace(x_min, x_max, 100)

    if order > 1:
        yhat, yhat_boots = _sns_sub_fit_poly(X, Y, grid, order)
    else:
        yhat, yhat_boots = _sns_sub_fit_fast(X, Y, grid)

    def _utils_ci(a, which=95, axis=None):
        # Return a percentile range from an array of values.
        p = 50 - which / 2, 50 + which / 2
        return np.nanpercentile(a, p, axis)

    ci = 95
    err_bands = _utils_ci(yhat_boots, ci, axis=0)
    return grid, yhat, err_bands


def _sns_line_err_bars(ax, kws, X, Y, order=1):
    # yhat, yhat_boots = _sns_line_fit_regs(grid, X, Y, order)
    grid, yhat, err_bands = _sns_line_fit_regs(ax, X, Y, order)
    edges = grid[0], grid[-1]

    # Get set default aesthetics
    fill_color = kws["color"]
    lw = kws.pop("lw", mpl.rcParams["lines.linewidth"] * 1.5)
    kws.setdefault("linewidth", lw)

    # Draw the regression line and confidence interval
    line, = ax.plot(grid, yhat, **kws)
    line.sticky_edges.x[:] = edges  # Prevent mpl from adding margin
    if err_bands is not None:
        ax.fill_between(grid, *err_bands,
                        facecolor=fill_color, alpha=.15)
    return ax


def scatter_id_chart(X, Y, figname, figsize='M-WS',
                     annots=('X', 'Y'), identity=False,
                     base=None, diff=0.05,
                     locate=PLT_LOCATION):
    # Correlation
    R = np.corrcoef(X, Y)[1, 0]
    key = "Correlation = %.4f" % R
    # Regression
    regr = np.polyfit(X, Y, deg=1)
    estimated = np.polyval(regr, X)

    fig = plt.figure(figsize=_setup_config['L-NT'])
    if identity:
        plt.scatter(X, Y)
        Z = sorted(X)
        plt.plot(Z, Z, "k--", lw=1, label='$f(x)=x$')
        plt.plot(X, estimated, "-", lw=1, label=key,
                 # color="dodgerblue"
                 # color="royalblue"
                 color="navy")  # cm
        # Note that 部分可能是实线，因为X没排序/不是增序排列
        # plt.plot(X, estimated, "b--", lw=1, label=key)
    else:
        plt.scatter(X, Y, label=key)
        plt.plot(X, estimated, "k-", lw=1.5)

    kws = {"color": "navy", "lw": 1}
    _sns_line_err_bars(fig.gca(), kws, X, Y)

    ax = fig.gca()
    x_min, x_max = ax.get_xlim()
    _, y_max = ax.get_ylim()  # y_min,
    if y_max - x_max >= abs(diff):
        # ax.set_xlim([x_min, x_max + 0.05])
        if diff > 0:
            ax.set_xlim(x_min, x_max + abs(diff))
        elif diff < 0:
            ax.set_ylim(0, y_max - 2 * abs(diff))

    if base is not None:
        _setup_locater(fig, base)
    fig.gca().set_aspect(1)  # plt.axis("equal")
    # 但是这个没有高赞的ax.set_aspect(1)通用，因为
    # plt.axis("equal")直接让xlim等设置失效了

    plt.xlabel(annots[0])
    plt.ylabel(annots[1])
    plt.legend(loc=locate, frameon=PLT_FRAMEBOX,
               labelspacing=.07, prop={'size': 9})  # 10
    if figsize is not None:
        fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


_scat_mcolor = [
    'steelblue', 'royalblue', 'slateblue',
    'lightslategray', 'slategray',
]


def scatter_parl_chart(data, picked_keys, annot,
                       figname, figsize='M-WS'):
    num_expt, num_core = data.shape
    ind = np.arange(num_expt)

    # core = picked_keys[0]
    # core = core.replace('$', '')
    # core = int(core[-1])
    core = int(figname[-1])

    fig = plt.figure(figsize=_setup_config['L-NT'])
    for k in range(num_core):
        plt.scatter(ind, data[:, k], s=9,
                    # c=_scat_mcolor[-k],
                    c=_scat_mcolor[k + core - 1],
                    label=picked_keys[k])

    plt.xlabel('Experiment')
    plt.ylabel(annot)
    plt.legend(loc=PLT_LOCATION,  # boxoff=
               labelspacing=.05, frameon=False)

    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def sns_scatter_corr(X, Y, figname, figsize='M-WS',
                     annots=('X', 'Y'),
                     identity=False, base=None):
    df = pd.DataFrame({'X': X, 'Y': Y})
    # df.corr()  # 默认是Pearson相关性分析

    sns.lmplot(y='Y', x='X', data=df, palette="tab20")
    # sns.lmplot(y='Y', x='X', data=df, legend=False)
    fig = plt.gcf()
    if figsize is not None:
        fig = _setup_figsize(fig, figsize)
    ax = fig.gca()  # ax = plt.gca()

    loc_x = ax.get_xlim()
    loc_y = ax.get_ylim()
    # loc_x = loc_x[1] - loc_x[0]
    # loc_y = loc_y[1] - loc_y[0]
    loc_x = _set_quantile(2.35, *loc_x)
    loc_y = _set_quantile(-4.5, *loc_y)

    if identity:
        loc_y = _set_quantile(-3.4, *ax.get_ylim())
        loc_x = _set_quantile(2.27, *ax.get_xlim())
        # loc_x: 2.19, 2.21,

    # '''
    # ax.text(loc_x, loc_y,
    #       "Pearson: {:.2f}".format(df.corr().iloc[1, 0]),
    #       transform=ax.transAxes)  # 添加相关性
    # '''
    key = df.corr().iloc[1, 0]
    if not identity:
        plt.text(loc_x, loc_y,
                 # "Pearson: {:.4f}".format(df.corr().iloc[1, 0])
                 "Correlation = {:.4f}".format(key))
        # fig = plt.gcf()

    if identity:
        Z = sorted(X)
        plt.plot(Z, Z, "k--", lw=1.2, label='$f(x)=x$')
        # plt.plot(X, X, "k--", lw=1, label='$f(x)=x$')
        # plt.legend(loc="lower right", frameon=False)
        plt.legend(loc=PLT_LOCATION, frameon=False,
                   labels=[
                       "Correlation = {:.4f}".format(key),
                       "$f(x)=x$"],
                   labelspacing=.07)

    if base is not None:
        _setup_locater(fig, base)
    fig.gca().set_aspect(1)
    plt.xlabel(annots[0])
    plt.ylabel(annots[1])
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def sns_corr_chart(X, Y, sens, figname, figsize='M-WS'):
    attr = "Sensitive"  # "sensitive_attributes"
    df = pd.DataFrame({'X': X, 'Y': Y, attr: sens})
    priv, keys = df[attr].unique(), {}
    for k, v in enumerate(priv):
        tmp = df[df[attr] == v].drop(columns=attr)
        tmp = tmp.corr().iloc[1, 0]
        # tmp = df[df[attr] == v].corr().iloc[1, 0]
        tmp = "{:6s} correlation = {:.4f}".format(v, tmp)
        keys[v] = tmp

    sns.lmplot(x='X', y='Y', data=df, ci=95,
               hue=attr, palette="husl",
               # legend_out=True,  # legend=keys
               # palette=dict(Yes="g", No="m")
               legend=False)  # markers=["o", "x"], ci=0.95,
    fig = plt.gcf()
    plt.legend(title=attr, loc=PLT_LOCATION, labels=list(priv))

    ax = fig.gca()
    loc_x = ax.get_xlim()
    loc_y = ax.get_ylim()
    loc_x = _set_quantile(1.40, *loc_x)
    loc_y = _set_quantile(-5.7, *loc_y)
    for k, v in enumerate(priv):
        plt.text(loc_x, loc_y - k * 13, "{:27s}".format(keys[v]))

    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# ----------------------------------
# Friedman 统计检验图


def _Friedman_plt(fig, k, ep1, ep2, anotCD, CD, offset):
    axs = plt.gca()
    # axs.set_yticks(np.arange(k), picked_keys)
    for t in range(k):
        plt.plot([ep1[t], ep2[t]], [t, t], '-b', lw=1.5)
    plt.ylim(-0.7, k - 0.3)

    xt = axs.get_xticks()
    axs.set_xlim(np.floor(xt[0]), np.ceil(xt[-1]))
    if anotCD:
        # plt.text(xt[-2], k - .7, r'CD = {:.4f}'.format(CD))
        plt.text(xt[-3] + .2, k - 1.3, r'CD = {:.4f}'.format(CD))
        axs.set_xlim(np.floor(xt[0]), np.ceil(xt[-1]) + 1.2)
    # plt.plot([ep2[k - 2], ep2[k - 2]], [-1.2, k + 1], 'r--', lw=1)
    # '''
    # plt.plot([ep1[k - 1], ep1[k - 1]], [-1.2, k + 1], 'r--', lw=1)
    # plt.plot([ep2[k - 1], ep2[k - 1]], [-1.2, k + 1], 'r--', lw=1)
    # '''
    if offset in [-1, -2]:
        plt.plot([ep1[k + offset], ] * 2, [-1.2, k + 1],
                 'r--', lw=1)
        plt.plot([ep2[k + offset], ] * 2, [-1.2, k + 1],
                 'r--', lw=1)
    elif offset in [-3]:
        plt.plot([ep2[k - 1], ] * 2, [-1.2, k + 1], 'r--', lw=1)
        plt.plot([ep2[k - 2], ] * 2, [-1.2, k + 1], 'r--', lw=1)
    elif offset in [-4]:
        plt.plot([ep1[k - 1], ] * 2, [-1.2, k + 1], 'r--', lw=1)
        plt.plot([ep1[k - 2], ] * 2, [-1.2, k + 1], 'r--', lw=1)
    # axs.invert_yaxis()
    return fig


def Friedman_chart(index_bar, picked_keys,
                   figname, figsize='S-NT',
                   alpha=.05, logger=None,
                   anotCD=False, offset=-1):
    avg_order = np.mean(index_bar, axis=0)
    # # N: number of the used datasets
    # # k: number of the compared algorithms/methods
    N, k = np.shape(index_bar)  # or `avg_accuracy`
    threshold = stats.f.pdf(
        1 - alpha, k - 1, (k - 1) * (N - 1))

    # mark, tau_F, tau_chi2 = Friedman_test(index_bar, alpha)
    _, tau_F, tau_chi2 = Friedman_test(index_bar, alpha)
    CD, q_alpha = Nememyi_posthoc_test(index_bar, alpha)
    elegant_print(["\n%s\n" % figname,
                   "$tau_{chi^2}$ " + str(tau_chi2),
                   "$tau_F$       " + str(tau_F),
                   "Threshold     " + str(threshold)
                   ], logger)
    if tau_F > threshold:
        elegant_print([
            "拒绝 '所有算法性能相同' 这个假设",
            "Refused hypothesis about `all methods perform the same`"
        ], logger)
    elegant_print(["$q_alpha$    " + str(q_alpha),
                   "CD           " + str(CD)], logger)
    ep1 = avg_order - CD / 2  # left   # eq_1
    ep2 = avg_order + CD / 2  # right  # eq_2

    fig = plt.figure(figsize=_setup_config['L-NT'])
    plt.plot(avg_order, np.arange(k), '.', markersize=11)
    plt.yticks(np.arange(k), picked_keys)
    fig = _Friedman_plt(fig, k, ep1, ep2, anotCD, CD, offset)
    fig = _setup_figsize(fig, figsize, invt=True)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# ----------------------------------
# 统计检验图 aggregated rank

cnames = {
    #  1-10
    'black': '#000000',
    'dimgray': '#696969',
    'gray': '#808080',
    'darkgray': '#A9A9A9',
    'silver': '#C0C0C0',
    'lightgray': '#D3D3D3',
    'gainsboro': '#DCDCDC',
    'whitesmoke': '#F5F5F5',
    'white': '#FFFFFF',
    'snow': '#FFFAFA',
    # 11-20
    'rosybrown': '#BC8F8F',
    'lightcoral': '#F08080',
    'indianred': '#CD5C5C',
    'brown': '#A52A2A',
    'firebrick': '#B22222',
    'maroon': '#800000',
    'darkred': '#8B0000',
    'red': '#FF0000',
    'mistyrose': '#FFE4E1',
    'salmon': '#FA8072',
    # 21-30
    'tomato': '#FF6347',
    'darksalmon': '#E9967A',
    'coral': '#FF7F50',
    'orangered': '#FF4500',
    'lightsalmon': '#FFA07A',
    'sienna': '#A0522D',
    'seashell': '#FFF5EE',
    'chocolate': '#D2691E',
    'saddlebrown': '#8B4513',
    'sandybrown': '#FAA460',
    # 31-40
    'peachpuff': '#FFDAB9',
    'peru': '#CD853F',
    'linen': '#FAF0E6',

    'bisque': '#FFE4C4',
    'darkorange': '#FF8C00',
    'burlywood': '#DEB887',
    'antiquewhite': '#FAEBD7',
    'tan': '#D2B48C',
    'navajowhite': '#FFDEAD',
    'blanchedalmond': '#FFEBCD',
    # 41-50
    'papayawhip': '#FFEFD5',
    'moccasin': '#FFE4B5',
    'orange': '#FFA500',
    'wheat': '#F5DEB3',
    'oldlace': '#FDF5E6',
    'floralwhite': '#FFFAF0',
    'darkgoldenrod': '#B8860B',
    'goldenrod': '#DAA520',
    'cornsilk': '#FFF8DC',
    'gold': '#FFD700',
    # 51-60
    'lemonchiffon': '#FFFACD',
    'khaki': '#F0E68C',
    'palegoldenrod': '#EEE8AA',
    'darkkhaki': '#BDB76B',
    'ivory': '#FFFFF0',
    'beige': '#F5F5DC',
    'lightyellow': '#FFFFE0',
    'lightgoldenrodyellow': '#FAFAD2',
    'olive': '#808000',
    'yellow': '#FFFF00',
    # 61-70
    'olivedrab': '#6B8E23',
    'yellowgreen': '#9ACD32',
    'darkolivegreen': '#556B2F',
    'greenyellow': '#ADFF2F',
    'chartreuse': '#7FFF00',
    'lawngreen': '#7CFC00',
    'honeydew': '#F0FFF0',
    'darkseagreen': '#8FBC8F',
    'palegreen': '#98FB98',

    'lightgreen': '#90EE90',
    # 71-80
    'forestgreen': '#228B22',
    'limegreen': '#32CD32',
    'darkgreen': '#006400',
    'green': '#008000',
    'lime': '#00FF00',
    'seagreen': '#2E8B57',
    'mediumseagreen': '#3CB371',
    'springgreen': '#00FF7F',
    'mintcream': '#F5FFFA',
    'mediumspringgreen': '#00FA9A',
    # 81-90
    'mediumaquamarine': '#66CDAA',
    'aquamarine': '#7FFFD4',
    'turquoise': '#40E0D0',
    'lightseagreen': '#20B2AA',
    'mediumturquoise': '#48D1CC',
    'azure': '#F0FFFF',
    'lightcyan': '#E0FFFF',
    'paleturquoise': '#AFEEEE',
    'darkslategray': '#2F4F4F',
    'teal': '#008080',
    # 90-100
    'darkcyan': '#008B8B',
    'cyan': '#00FFFF',
    'aqua': '#00FFFF',
    'darkturquoise': '#00CED1',
    'cadetblue': '#5F9EA0',
    'powderblue': '#B0E0E6',
    'lightblue': '#ADD8E6',
    'deepskyblue': '#00BFFF',
    'skyblue': '#87CEEB',
    'lightskyblue': '#87CEFA',
    # 101-110
    'steelblue': '#4682B4',
    'aliceblue': '#F0F8FF',
    'dodgerblue': '#1E90FF',
    'lightslategray': '#778899',
    'slategray': '#708090',

    'lightsteelblue': '#B0C4DE',
    'cornflowerblue': '#6495ED',
    'royalblue': '#4169E1',
    'ghostwhite': '#F8F8FF',
    'lavender': '#E6E6FA',
    # 111-120
    'midnightblue': '#191970',
    'navy': '#000080',
    'darkblue': '#00008B',
    'mediumblue': '#0000CD',
    'blue': '#0000FF',
    'slateblue': '#6A5ACD',
    'darkslateblue': '#483D8B',
    'mediumslateblue': '#7B68EE',
    'mediumpurple': '#9370DB',
    'blueviolet': '#8A2BE2',
    # 121-130
    'indigo': '#4B0082',
    'darkorchid': '#9932CC',
    'darkviolet': '#9400D3',
    'mediumorchid': '#BA55D3',
    'thistle': '#D8BFD8',
    'plum': '#DDA0DD',
    'violet': '#EE82EE',
    'purple': '#800080',
    'darkmagenta': '#8B008B',
    'fuchsia': '#FF00FF',
    # 131-140
    'magenta': '#FF00FF',
    'orchid': '#DA70D6',
    'mediumvioletred': '#C71585',
    'deeppink': '#FF1493',
    'hotpink': '#FF69B4',
    'lavenderblush': '#FFF0F5',
    'palevioletred': '#DB7093',
    'crimson': '#DC143C',
    'pink': '#FFC0CB',
    'lightpink': '#FFB6C1',
}

cname_keys = list(cnames.keys())


cmap_names = [
    'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG',
    'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
    'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu',
    'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r',
    'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn',
    'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r',
    'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu',
    'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r',
    'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu',
    'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r',
    'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds',
    'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r',
    'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia',
    'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r',
    'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot',
    'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r',
    'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 
    'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r',
    'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest',
    'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r',
    'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray',
    'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r',
    'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg',
    'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r',
    'gray', 'gray_r', 'hot', 'hot_r', 'hsv',
    'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r',
    'jet', 'jet_r', 'magma', 'magma_r', 'mako',
    'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r',
    'pink', 'pink_r', 'plasma', 'plasma_r', 'prism',
    'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r',
    'seismic', 'seismic_r', 'spring', 'spring_r', 'summer',
    'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r',
    'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain',
    'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r',
    'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag',
    'vlag_r', 'winter', 'winter_r'
]


def stat_chart_stack(index_bar, picked_keys,
                     figname, figsize='S-NT',
                     annots="aggr.rank.accuracy",
                     cmap_name='GnBu_r',
                     rotation=25):
    pick_idxbar = np.cumsum(index_bar, axis=0)
    pick_leng_pru = len(picked_keys)  # aka. pick_name_pru
    pick_leng_set = np.shape(index_bar)[0]  # avg_accuracy

    cs, cl = _setup_rgb_color(
        pick_leng_set, cmap_name)  # np.shape(index_bar)[0],
    fig, _ = plt.subplots(figsize=_setup_config['L-NT'])
    ind = np.arange(1, pick_leng_pru + 1)  # _:axs
    plt.bar(ind, index_bar[0], color=cs[0])

    for i in range(1, pick_leng_set):
        plt.bar(ind, index_bar[i], bottom=pick_idxbar[i - 1],
                color=cs[i % cl])  # cmap='viridis')

    plt.ylabel("{}".format(annots))
    # plt.xticks(ind, picked_keys, rotation=25)
    plt.xticks(ind, picked_keys, rotation=rotation)
    plt.xlim(0.25, pick_leng_pru + 1 - 0.25)
    fig = _setup_figsize(fig, figsize, invt=False)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def stat_chart_group(avg_accuracy, pickup_set,
                     picked_keys,
                     figname, figsize='S-NT',
                     cmap_name='GnBu_r'):
    if not isinstance(avg_accuracy, np.ndarray):
        avg_accuracy = np.array(avg_accuracy)  # robustness
    pickup_set = np.array(pickup_set) - 1
    # Notice that `pickup_set` starts with 1, instead of 0.

    pick_leng_pru = len(picked_keys)
    pick_leng_set = np.shape(avg_accuracy)[0]
    cs, cl = _setup_rgb_color(pick_leng_set, cmap_name)

    ind = np.arange(1, pick_leng_pru + 1)
    fig, axs = plt.subplots(figsize=_setup_config['L-NT'])
    for id_set in pickup_set:
        plt.clf()   # reset
        val_mean = avg_accuracy[id_set, :]
        plt.bar(ind, val_mean, color=cs[id_set % cl])

        plt.xticks(ind, picked_keys, rotation=45)
        yt = axs.get_yticks()
        plt.xlim(0.25, pick_leng_pru + 1 - 0.25)
        plt.ylim(yt[0], 100)

        fig = _setup_figsize(fig, figsize, invt=False)
        _setup_figshow(fig, figname + "_ds" + str(id_set + 1))
    plt.close(fig)


# ----------------------------------
# 多分类问题中的混淆矩阵可视化实现


def visual_confusion_mat(y, hx, label_vals,
                         figname, figsize='M-WS',
                         title='Confusion matrix',
                         # cmap=None,
                         cmap_name='Blues',
                         normalize=True):
    # aka. def visualise_confusion_matrix()
    """
    cm:           confusion matrix, np.ndarray
    target_names: given classification classes such as [0, 1, 2]
                  the class names, e.g., ['high','medium','low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see:
                  http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions  # show proportions
    """

    cm = confusion_matrix(y, hx)  # y_true, y_pred
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy  # scalars
    # if cmap is None:
    #     cmap = plt.get_cmap('Blues')
    # cmap = plt.get_cmap(cmap_name)  # default:None
    cmap = plt.colormaps.get_cmap(cmap_name)

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    if title != '':
        plt.title(title)
    plt.colorbar()

    # aka. target_names
    if label_vals is not None:
        tick_marks = np.arange(len(label_vals))
        plt.xticks(tick_marks, label_vals, rotation=15)
        plt.yticks(tick_marks, label_vals)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]),
                                  range(cm.shape[1])):
        tmp = "white" if cm[i, j] > thresh else "black"

        # '''
        # if normalize:
        #     plt.text(j, i, "{:0.4f}".format(cm[i, j]),
        #              horizontalalignment="center", color=tmp)
        # else:
        #     plt.text(j, i, "{:,}".format(cm[i, j]),
        #              horizontalalignment="center", color=tmp)
        # '''

        tmp_annot = "{:0.4f}".format(cm[
            i, j]) if normalize else "{:,}".format(cm[i, j])
        plt.text(j, i, tmp_annot, horizontalalignment="center",
                 color=tmp)
        del tmp_annot

    # 'accuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass)
    key = 'accuracy={:.2f}%; misclass={:.2f}%'.format(
        accuracy * 100, misclass * 100)
    plt.ylabel('True label')
    plt.xlabel('Predicted label\n%s' % key)
    # fig = _setup_figsize(fig, figsize)  # ,invt=False)
    _style_set_fig(fig, siz=_setup_config[figsize])
    _setup_figshow(fig, figname)
    return


# ----------------------------------
# subchart

# subchart_backslash_distributed.m

_backslash_distributed = {
    "ua": "Accuracy (%)",
    "ut": "Time Cost (s)",
    "us": "Space Cost (#)"
}

_baseline_facecolor = {
    "ua": {'h1': tuple([i / 255 for i in [153, 50, 204]]),
           'h2': tuple([i / 255 for i in [204, 204, 254]])},
    "ut": {'h1': tuple([0, 0.4470, 0.7410]),
           'h2': tuple([i / 255 for i in [135, 206, 250]])},
    "us": {'h1': tuple([0, 0.5019, 0.5020]),
           'h2': tuple([i / 255 for i in [64, 224, 208]])}
}


# _barh_kwargs =
# _barh_fcolor =

_barh_patterns = (
    '/', '//', '-', '+', 'x',
    '\\', '\\\\', '*', 'o', 'O',
    '.'
)


def backslash_subchart(val_avg, val_std, pickup_uat,
                       picked_keys, figname, figsize='S-NT'):
    locate = PLT_LOCATION
    if pickup_uat == "ua":
        locate = "lower left"  # "SouthWest"
    annots = _backslash_distributed[pickup_uat]

    h1_facecolor = _baseline_facecolor[pickup_uat]['h1']
    h2_facecolor = _baseline_facecolor[pickup_uat]['h2']

    pick_leng_pru = len(picked_keys)
    ind = np.arange(pick_leng_pru)  # x locations for groups
    wid = .4  # width of the bars: can also be len(x) sequence
    gap = 0.  # .03
    fig, axs = plt.subplots(figsize=_setup_config['L-NT'])

    # p1 = axs.bar(ind - wid / 2 - gap,
    axs.bar(ind - wid / 2 - gap, val_avg[0], wid,
            yerr=val_std[0], color=h1_facecolor,
            label="Centralised", **_barh_kwargs)
    # p2 = axs.bar(ind + wid / 2 + gap,
    axs.bar(ind + wid / 2 + gap, val_avg[1], wid,
            yerr=val_std[1], color=h2_facecolor,
            label="Distributed", **_barh_kwargs)
    # box off

    t = axs.get_ylim()
    ymin = np.min(val_avg - val_std)
    ymax = np.max(val_avg + val_std)
    if pickup_uat == "ua":
        ymin = np.floor(ymin / 5) * 5 - 5
        ymax = np.min([t[1], ymax + 0.5])
        # axs.set_ylim(ymin, ymax)
        axs.set_ylim(0, ymax)
    elif pickup_uat in ["ut", "us"]:
        ymin = np.max([-2, np.floor(ymin) - 1])
        ymax = np.min([t[1], np.ceil(ymax / 10 + 1) * 10])
        # axs.set_ylim(np.max([ymin, 0]), ymax)
        axs.set_ylim(0, ymax)
    del ymin, ymax

    axs.legend(loc=locate, labelspacing=.05)
    axs.set_ylabel(annots)
    axs.set_xticks(ind)
    axs.set_xticklabels(picked_keys, rotation=25)
    axs.autoscale_view()
    fig = _setup_figsize(fig, figsize, invt=False)
    _setup_figshow(fig, figname + "_" + pickup_uat)
    plt.close(fig)
    return


# bar_chart_with_error.m

def bar_chart_with_error(greedy, ddismi,
                         tc_grd, tc_dsm,
                         sc_grd, sc_dsm,
                         pickup_pru, name_pru_set,
                         figname, figsize='S-NT'):
    #                picked_keys, cmap_name='Purples_r'
    # row: data_set; col: name_pru
    picked_keys = [name_pru_set[i] for i in pickup_pru]
    pick_leng_pru = len(picked_keys)

    A_avg = np.zeros(shape=(2, pick_leng_pru))
    A_std = np.zeros(shape=(2, pick_leng_pru))
    T_avg = np.zeros(shape=(2, pick_leng_pru))
    T_std = np.zeros(shape=(2, pick_leng_pru))
    S_avg = np.zeros(shape=(2, pick_leng_pru))
    S_std = np.zeros(shape=(2, pick_leng_pru))

    greedy *= 100.  # percent
    ddismi *= 100.  # percent

    A_avg[0, :] = np.mean(greedy[:, pickup_pru], axis=0)
    A_avg[1, :] = np.mean(ddismi[:, pickup_pru], axis=0)
    A_std[0, :] = np.std(greedy[:, pickup_pru], axis=0, ddof=1)
    A_std[1, :] = np.std(ddismi[:, pickup_pru], axis=0, ddof=1)

    T_avg[0, :] = np.mean(tc_grd[:, pickup_pru], axis=0)
    T_avg[1, :] = np.mean(tc_dsm[:, pickup_pru], axis=0)
    T_std[0, :] = np.std(tc_grd[:, pickup_pru], axis=0, ddof=1)
    T_std[1, :] = np.std(tc_dsm[:, pickup_pru], axis=0, ddof=1)

    S_avg[0, :] = np.mean(sc_grd[:, pickup_pru], axis=0)
    S_avg[1, :] = np.mean(sc_dsm[:, pickup_pru], axis=0)
    S_std[0, :] = np.std(sc_grd[:, pickup_pru], axis=0, ddof=1)
    S_std[1, :] = np.std(sc_dsm[:, pickup_pru], axis=0, ddof=1)

    backslash_subchart(A_avg, A_std, "ua", picked_keys, figname)
    backslash_subchart(T_avg, T_std, "ut", picked_keys, figname)
    backslash_subchart(S_avg, S_std, "us", picked_keys, figname)
    return


def multiple_hist_chart(Ys_avg, Ys_std, picked_keys,
                        annotX='', annotYs=tuple('Y'),
                        figname='', figsize='S-NT',
                        rotate=80):
    # aka. def multiple_hist_charts()
    pick_leng_pru, pick_len_fair = Ys_avg.shape
    ind = np.arange(pick_leng_pru) * 2
    wid = (2 - 0.4) / pick_len_fair
    # wid = .25  # width,index
    fig, ax = plt.subplots(figsize=_setup_config['L-NT'])

    kwargs = _barh_kwargs.copy()
    kwargs["error_kw"]["elinewidth"] = .7  # wid
    kwargs["error_kw"]["capsize"] = 3.6
    for k in range(pick_len_fair):
        gap = k * wid
        ax.bar(ind + gap, Ys_avg[:, k], width=wid,
               yerr=Ys_std[:, k], label=annotYs[k],
               # color=_barh_fcolor[k], **_barh_kwargs)
               color=_barh_fcolor[k], **kwargs)
    ymin, ymax = ax.get_ylim()
    # if ymax > 0:
    if ymax > 0 and ymax > abs(ymin):
        ax.set_ylim(0, ymax)
        if annotX.startswith('Fairness'):
            ax.set_ylim(0, ymax + 0.15)
    # elif ymin < 0 and ymax <= 0:
    elif ymin < 0 and abs(ymin) > ymax:
        ax.set_ylim(ymin, 0)
        if annotX.startswith('Fairness'):
            ax.set_ylim(ymin - 0.05, 0)

    ax.set_ylabel(annotX)
    ax.legend(loc=PLT_LOCATION, labelspacing=.05, fontsize=8)
    ax.set_xticks(ind + wid * (pick_len_fair // 2 - .5))
    ax.set_xticklabels(picked_keys, rotation=rotate)
    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def twinx_hist_chart(Y1_avg, Y1_std, Y2_avg, Y2_std,
                     # picked_keys, annotX='', annotY=tuple('Y'),
                     picked_keys, annotY=tuple('Y'),
                     annotX=('Fairness Measure', 'Objective Function'),
                     # annotX=('Fairness Measure', r'$\mathcal{L}$'),
                     figname='', figsize='S-NT', rotate=20):
    pick_leng_pru = len(picked_keys)
    pick_fair_1 = np.shape(Y1_avg)[1]
    pick_fair_2 = np.shape(Y2_avg)[1]
    ind = np.arange(pick_leng_pru) * 2
    wid = (2 - 0.5) / (pick_fair_1 + pick_fair_2)
    fig, ax1 = plt.subplots(figsize=_setup_config['L-NT'])
    ax2 = ax1.twinx()
    ps = []

    for k in range(pick_fair_1):
        gap = k * wid
        pt = ax1.bar(ind + gap, Y1_avg[:, k], width=wid,
                     yerr=Y1_std[:, k], label=annotY[k],
                     color=_barh_fcolor[k], **_barh_kwargs)
        ps.append(pt)
    for k in range(pick_fair_2):
        gap = (k + pick_fair_1) * wid
        pt = ax2.bar(ind + gap, Y2_avg[:, k], width=wid,
                     yerr= Y2_std[:, k], hatch='//',
                     label=annotY[k + pick_fair_1],
                     color=_barh_fcolor[k + pick_fair_1],
                     **_barh_kwargs)
        ps.append(pt)

    ymax = ax1.get_ylim()[1]  # ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(0, ymax)
    # ymin, ymax = ax2.get_ylim()
    # ax2.set_ylim(0, ymax)
    ax1.set_ylabel(annotX[0])
    ax2.set_ylabel(annotX[1])
    # ax1.set_ylabel(annotX)
    ax1.set_xticks(ind + .5)
    ax1.set_xticklabels(picked_keys, rotation=rotate)
    plt.legend(handles=ps, labelspacing=.05)
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def twinx_bars_chart(Y1_avg, Y1_std, Y2_avg, Y2_std,
                     picked_keys, annotX=tuple('X'),
                     figname='', figsize='S-NT', rotate=45):
    pick_leng_pru = len(picked_keys)
    pick_fair_1 = np.shape(Y1_avg)[0]
    pick_fair_2 = np.shape(Y2_avg)[0]
    ind_1 = np.arange(pick_fair_1) * 2
    ind_2 = np.arange(pick_fair_2) * 2 + pick_fair_1 * 2
    wid = (2 - 0.5) / pick_leng_pru
    fig, ax1 = plt.subplots(figsize=_setup_config['L-NT'])
    ax2 = ax1.twinx()
    ps = []

    for k in range(pick_leng_pru):
        gap = k * wid
        pt = ax1.bar(ind_1 + gap, Y1_avg[:, k], width=wid,
                     yerr=Y1_std[:, k], label=picked_keys[k],
                     color=_barh_fcolor[k], **_barh_kwargs)
        ps.append(pt)
    for k in range(pick_leng_pru):
        gap = k * wid
        ax2.bar(ind_2 + gap, Y2_avg[:, k], width=wid,
                yerr=Y2_std[:, k], color=_barh_fcolor[k],
                hatch='//', **_barh_kwargs)

    ymax = ax1.get_ylim()[1]  # ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(0, ymax)
    # ymin, ymax = ax2.get_ylim()
    # ax2.set_ylim(0, ymax)
    ax1.set_ylabel('Fairness Measure')
    ax2.set_ylabel(r'$diff($accuracy$)$')
    ax1.set_xticks(np.concatenate([ind_1, ind_2], axis=0))
    ax1.set_xticklabels(annotX, rotation=rotate)
    plt.legend(handles=ps, loc=PLT_LOCATION, labelspacing=.05)
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# ----------------------------------
# subchart

# line_chart.m
#   Effect of $\lambda$ value

_line_marker = [
    '--vr', '--om', '--sb', '--dk',
    '-xr', '-om', '-sb', '-dk',
]


def line_chart(data,
               pickup_thin, set_k2,
               pickup_lam2, set_lam2,
               figname, figsize='S-NT'):
    if not isinstance(data, np.ndarray):
        data = np.array(data)  # for list
    # end if for robustness

    # Global
    len_thin, len_lam2 = len(set_k2), len(set_lam2)
    key_thin = [str(np.around(
        set_k2[i], decimals=2)) for i in range(len_thin)]
    key_lam2 = [np.around(
        set_lam2[j], decimals=2) for j in range(len_lam2)]
    key_lam2 = [r"$\lambda$ = " + str(j) for j in key_lam2]

    cnt = 0  # counter = 1
    fig_thin, axs_thin = plt.subplots(
        figsize=_setup_config['L-NT'])
    for j in pickup_lam2:
        axs_thin.plot(set_k2, data[:, j],
                      _line_marker[cnt], label=key_lam2[j],
                      linewidth=1, markersize=3)
        cnt += 1
    # axs_thin.set_xlim(min(set_k2) - 0.3, max(set_k2) + 0.3)
    axs_thin.set_xlabel("Size of the Pruned Sub-Ensemble")
    axs_thin.set_ylabel("Accuracy (%)")
    plt.legend(loc=PLT_LOCATION, frameon=PLT_FRAMEBOX,
               labelspacing=.05)
    fig_thin = _setup_figsize(fig_thin, figsize)
    _setup_figshow(fig_thin, figname + "_lam2")
    plt.close(fig_thin)

    cnt = 0
    fig_lam2 = plt.figure(figsize=_setup_config['L-NT'])
    for i in pickup_thin:
        plt.plot(set_lam2, data[i, :],
                 _line_marker[cnt], label=key_thin[i],
                 linewidth=1, markersize=3)
        cnt += 1
    # plt.xlim(min(set_lam2) - 0.015, max(set_lam2) + 0.015)
    plt.xlabel(r"$\lambda$")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc=PLT_LOCATION, frameon=True,
               labelspacing=.05)
    fig_lam2 = _setup_figsize(fig_lam2, figsize)
    _setup_figshow(fig_lam2, figname + "_thin")
    plt.close(fig_lam2)
    return


def multiple_line_chart(X, Ys, annots=(
        r"$\lambda$", r"Test Accuracy (%)"),
        annotY=('Ensem',), mkrs=None,
        figname='lam', figsize='S-NT'):
    # aka. def line_chart_with_algs()
    if mkrs is None:
        mkrs = ['-'] + _line_marker
    pick_leng_pru = Ys.shape[1]
    fig, ax = plt.subplots(figsize=_setup_config['L-NT'])
    for j in range(pick_leng_pru):
        i = j % len(mkrs)
        ax.plot(X, Ys[:, j], mkrs[i], label=annotY[j],
                linewidth=1, markersize=2.5)  # 2
    ax.set_xlabel(annots[0])
    ax.set_ylabel(annots[1])
    ax.set_xlim(X[0], X[-1])
    plt.legend(loc=PLT_LOCATION, frameon=PLT_FRAMEBOX,
               labelspacing=.05)
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# comparison_subchart_accuracy.m
# def bsl_subchart_accuracy():
#   pass

# comparison_subchart_timecost.m
# def bsl_subchart_timecost():
#   pass


def baseline_subchart(val_avg, val_std, pickup_uat,
                      picked_keys, figname, figsize='S-NT'):
    locate = "lower left"  # "SouthWest"
    annots = _backslash_distributed[pickup_uat]

    h1_facecolor = _baseline_facecolor[pickup_uat]['h1']
    h2_facecolor = _baseline_facecolor[pickup_uat]['h2']

    pick_leng_pru = len(picked_keys)
    ind = np.arange(pick_leng_pru)  # locations for groups
    wid = .41  # width of the bars: can also be len(x) sequence
    kwargs = {
        "error_kw": {"ecolor": "0.2", "capsize": 6, },
        # "width": .45,
    }

    fig, axs = plt.subplots(figsize=_setup_config['L-NT'])
    # p1 = axs.bar(ind - wid / 2, val_avg[0], wid, yerr=val_std[0],
    #              color=h1_facecolor, label="Centralised", **kwargs)
    # p2 = axs.bar(ind + wid / 2, val_avg[1], wid, yerr=val_std[1],
    #              color=h2_facecolor, label="Distributed", **kwargs)
    axs.bar(ind - wid / 2, val_avg[0], wid, yerr=val_std[0],
            color=h1_facecolor, label="Centralised", **kwargs)
    axs.bar(ind + wid / 2, val_avg[1], wid, yerr=val_std[1],
            color=h2_facecolor, label="Distributed", **kwargs)

    if pickup_uat == "ua":
        t = axs.get_ylim()
        axs.set_ylim(50, t[1])
    axs.set_ylabel(annots)
    axs.set_xticks(ind)
    axs.set_xticklabels(picked_keys, rotation=35)
    axs.legend(loc=locate, labelspacing=.046)
    axs.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname + "_" + pickup_uat)
    plt.close(fig)
    return


# ----------------------------------
# subchart
# masegosa2020second.pdf
#
# aka. bar chart
# For one dataset, corresponding to one specific
# name_ens & abbr_cls

_hist_color = [
    'forestgreen',
    'slateblue', 'darkslateblue', 'mediumslateblue',
    'mediumpurple', 'indigo'
]

_hist_color = [
    tuple([i / 255 for i in [65, 189, 122]]),   # 41BD7A, 74
    tuple([i / 255 for i in [118, 138, 127]]),  # 768A7F, 54
    tuple([i / 255 for i in [126, 240, 110]]),  # 7EF06E, 94
    tuple([i / 255 for i in [235, 172, 242]]),  # EBACF2, 95
    tuple([i / 255 for i in [144, 68, 189]]),   # 9044BD, 74
]  # RGB, bright. 复合

_hist_linsty = [
    # 'dashed', 'solid', 'dotted', 'dashdot',
    '--', '-', '.', '-.',
]  # plt.plot(linestype=)


def _hist_calc_XY(X, Ys):
    k = np.shape(Ys)[1]  # N, k = np.shape(Ys)

    X_avg = None if X is None else np.mean(X, axis=0).tolist()
    Y_avg = np.mean(Ys, axis=0)
    Y_std = np.std(Ys, axis=0, ddof=1)

    ind = np.arange(k)
    # nb_st: split
    return X_avg, Y_avg, Y_std, ind


def histogram_hor(ind, X_avg, Y_avg, Y_std,
                  figname, figsize='M-WS',
                  annotX='X', annotY=('Y',), st=100,
                  remarks=None):
    # horizontal (acr)
    fig, ax = plt.subplots(figsize=_setup_config['L-NT'])
    ax.bar(ind, Y_avg, yerr=Y_std,
           color=_hist_color, **_barh_kwargs)

    x_min, x_max = ax.get_xlim()
    grid = np.linspace(x_min - .5, x_max + .5, num=st)
    if (X_avg is not None) and (remarks is not None):
        if isinstance(X_avg, float):
            ax.plot(grid, [X_avg] * st, '--', color='orange',
                    label=remarks[0])
        else:
            for k, v in enumerate(X_avg):
                ax.plot(grid, [v] * st, _hist_linsty[k],
                        color='orange', label=remarks[k])
    elif X_avg is not None:
        if isinstance(X_avg, float):
            ax.plot(grid, [X_avg] * st, '--', color='orange')
        else:
            for k, v in enumerate(X_avg):
                ax.plot(grid, [v] * st, _hist_linsty[k],
                        color='orange')
    ax.set_xlim(x_min, x_max)
    if remarks is not None:
        plt.legend(loc='upper right', frameon=False, labelspacing=.05)

    y_max = ax.get_ylim()[1]  # y_min, y_max = ax.get_ylim()
    ax.set_ylim(0, y_max)
    ax.set_ylabel(annotX)
    ax.set_xticks(ind)
    ax.set_xticklabels(annotY, rotation=80)

    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize, invt=False)
    _setup_figshow(fig, figname + '_hor')
    plt.close(fig)
    return


def histogram_vrt(ind, X_avg, Y_avg, Y_std,
                  figname, figsize='M-WS',
                  annotX='X', annotY=('Y',), st=100,
                  remarks=None):
    # vertical (vert)
    fig, ax = plt.subplots(figsize=_setup_config['L-NT'])
    ax.barh(ind, Y_avg, xerr=Y_std,
            color=_hist_color, **_barh_kwargs)

    y_min, y_max = ax.get_ylim()
    grid = np.linspace(y_min - .5, y_max + .5, num=st)
    if (X_avg is not None) and (remarks is not None):
        if isinstance(X_avg, float):
            ax.plot([X_avg] * st, grid, '--', color='orange',
                    label=remarks[0])
            # break
        else:
            for k, v in enumerate(X_avg):
                ax.plot([v] * st, grid, _hist_linsty[k],
                        color='orange', label=remarks[k])
    elif X_avg is not None:
        if isinstance(X_avg, float):
            ax.plot([X_avg] * st, grid, '--', color='orange')
        else:
            for k, v in enumerate(X_avg):
                ax.plot([v] * st, grid, _hist_linsty[k],
                        color='orange')
    ax.set_ylim(y_min, y_max)
    if remarks is not None:
        plt.legend(loc='upper right', frameon=False, labelspacing=.05)

    x_max = ax.get_xlim()[1]  # x_min, x_max = ax.get_xlim()
    ax.set_xlim(0, x_max)
    ax.set_xlabel(annotX)
    ax.set_yticks(ind)
    ax.set_yticklabels(annotY)

    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize, invt=True)
    _setup_figshow(fig, figname + '_vrt')
    plt.close(fig)
    return


def histogram_chart(X, Ys, figname, figsize='M-WS',
                    annotX='X', annotY=('Y',), st=100,
                    ind_hv='h'):
    X_avg, Y_avg, Y_std, ind = _hist_calc_XY(X, Ys)
    # y_labels = ['Size of the Original Ensemble',
    #             'Size of the Pruned Sub-Ensemble (expected)']
    y_labels = ['# Ensem', '# Sub-Ensem (Exp.)']

    if ind_hv == 'h':
        histogram_hor(ind, X_avg, Y_avg, Y_std,
                      figname, figsize, annotX, annotY, st,
                      y_labels)
    elif ind_hv == 'v':
        histogram_vrt(ind, X_avg, Y_avg, Y_std,
                      figname, figsize, annotX, annotY, st,
                      y_labels)
    return


# ----------------------------------
