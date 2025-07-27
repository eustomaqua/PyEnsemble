# coding: utf-8
#
# TARGET:
#   Measuring fairness via data manifolds
#


import matplotlib.pyplot as plt
import seaborn as sns
# import itertools
import numpy as np
import pandas as pd

# from fairml.facilc.draw_graph import (
# from pyfair.senior.draw_graph import (
from pyfair.facil.draw_prelim import (
    PLT_LOCATION, PLT_FRAMEBOX, _setup_config,
    _style_set_axis, _setup_figsize, _setup_figshow,
    _setup_rgb_color)
from pyfair.granite.draw_graph import _sns_line_err_bars
# from pyfair.senior.draw_graph import (
#     _sns_line_err_bars, _setup_rgb_color)
# _setup_locater,_set_quantile, cnames, cname_keys, cmap_names,
# _backslash_distributed, _barh_patterns, _sns_line_fit_regs,
from pyfair.facil.utils_const import DTY_FLT


# ===============================
# Preliminaries
# Matlab plot
# -------------------------------


# ===============================
# Python Matlablib plot


# -------------------------------
# multiple line_chart.m
#

_line_styler = [
    '-', '--', '-.', ':']  # solid, dashed, dash-dot, dotted
_line_marker = [
    '.', ',', 'o', 'v', '^', '<', '>',  # point, pixel, circle,
    '1', '2', '3', '4',  # triangle_.., tri_down/up/left/right,
    's', 'p', '*', 'h', 'H',  # square, pentagon, star, hexagon,
    '+', 'x', 'D', 'd', '|', '_']  # /thin_diamond, vline/hline

_colr_nms = [
    '#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30',
    '#4DBEEE', '#A2142F']  # Matlab 2014, parula


# '''
# def multiple_line_chart(X, Ys, annots=(
#     r"$\lambda$", r"Test Accuracy (%)"),
#         annotY=('',), mkrs=None,
#         figname='lam', figsize='S-NT'):
#     # X .shape (#num,)
#     # Ys.shape (#num, #baseline_for_comparison, #iter)
#
#     if mkrs is None:
#         mkrs = []
#         # mkrs += [for i in _line_marker]
#
#     num_bs = Ys.shape[1]
# '''


# def multiple_lines_with_errorbar(Ys, picked_keys, annotY='Acc',
# TODO!
def multiple_lines_with_errorbar(X, Ys, picked_keys=('Baseline #1',),
                                 annotX=r'$\lambda$', annotY='Acc',
                                 cmap_name='GnBu_r',
                                 figname='lam_sns', figsize='M-WS'):
    # similar usage: box_plot(Ys[:, i, :])
    #                only works for one algorithm
    # X or picked_keys: (#num,)
    # Ys.shape (#num, #baseline_for_comparison, nb_iter)

    # num, pick_baseline, _ = Ys.shape  # pick_baseline,nb_iter
    _, pick_baseline, _ = Ys.shape
    # fig, ax = plt.subplots(figsize=_setup_config['M-NT'])
    fig = plt.figure(figsize=_setup_config['M-NT'])
    ax = fig.gca()

    cs, _ = _setup_rgb_color(pick_baseline, cmap_name)  # ,cl
    kws = {'color': 'navy', 'lw': 1}  # plt.plot(.5, 0.5)
    for j in range(pick_baseline):
        kws['color'] = cs[j]
        kws['label'] = picked_keys[j]
        # TODO 好像有点不对
        _sns_line_err_bars(ax, kws, X, Ys[:, j, :].mean(axis=1))

    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# -------------------------------
# 箱线图 带误差线
#


def box_plot(Ys, picked_keys, annotY='Acc',
             annotX='', patch_artist=False,
             figname='box_lam', figsize='M-WS', rotate=60):
    # Ys.shape (#baseline_for_comparison, #iter)

    pick_baseline = Ys.shape[0]  # ,nb_iter #picked_ways/method,
    fig, ax = plt.subplots(figsize=_setup_config['M-NT'])
    ax.boxplot(Ys.T, patch_artist=patch_artist)  # bp=

    ind = np.arange(pick_baseline) + 1
    ax.set_xticks(ind)
    ax.set_xticklabels(picked_keys, rotation=rotate)
    ax.set_ylabel(annotY)
    ax.set_xlabel(annotX)

    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def styled_box_plot():
    pass


# -------------------------------

# -------------------------------


# ===============================
# Python Matlablib plot


# -------------------------------
# fairmanf
#   for 单独一个数据集，类似上图，results of 5 iterations
#   (real approx values) + analysis (like mean+-std)
#


def scatter_k_cv_with_real(X, Ys, z,  # y/z: real values
                           # picked_keys=('Baseline #1',),
                           annotX=r'hyper-pm', annotY='value',
                           tidy_cv=False, ddof=0,  # 1,tidy_cv=True,
                           figsize='M-WS',  # cmap_name='GnBu_r',
                           figname='hyperpm_effect'):
    # This is for results from k-cross validation
    # X : possible values of some certain hyper-parameter
    # Ys: results of 算法的估计值
    # z : real value of 真实值

    # X .shape= (#num,)
    # Ys.shape= (nb_iter, #num)  # Ys.shape= (#num, nb_iter)
    # z .shape= (nb_iter,)

    nb_iter = Ys.shape[0]  # nb_iter, num = Ys.shape
    fig, ax = plt.subplots(figsize=_setup_config['M-NT'])

    kws = {'color': '#F65F47', 'lw': 1}
    for i in range(nb_iter):
        plt.scatter(X, Ys[i], s=2.5, **kws)

    kws['color'] = '#465386'
    tz_avg, tz_std = np.mean(z), np.std(z, ddof=ddof)
    tx_min, tx_max = ax.get_xlim()
    line, = ax.plot([tx_min, tx_max], [tz_avg, tz_avg],
                    label='Real value', **kws)
    line.sticky_edges.x[:] = (tx_min, tx_max)
    ax.fill_between([tx_min, tx_max],  # ax.get_xlim(),
                    [tz_avg - tz_std] * 2,
                    [tz_avg + tz_std] * 2, alpha=.15,
                    facecolor='#465386')  # **kws)
    # kws.pop('edgecolor')
    # kws.pop('facecolor')

    kws['color'] = '#F65F47'
    if not tidy_cv:
        tX = np.array([X] * nb_iter)
        tYs = Ys.reshape(-1)
        # tz = np.array([z] * num).T
    else:
        tX = np.array([X] * nb_iter).T
        tYs = Ys.T.reshape(-1)
        # tz = np.array([z] * num)
    kws['linestyle'] = '--'
    _sns_line_err_bars(ax, kws, tX.reshape(-1), tYs)
    kws.pop('linestyle')

    ax.ticklabel_format(
        style='sci', scilimits=(-1, 2), axis='y')
    plt.legend(loc='best', frameon=False)  # PLT_LOCATION,PLT_FRAMEBOX)
    ax.set_xlabel(annotX)
    ax.set_ylabel(annotY)

    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# -------------------------------
# fairmanf
#   时间代价


def boxplot_k_cv_with_real(X, Ys, z,
                           annotX=r'hyperpm', annotY='value',
                           patch_artist=False, ddof=0,
                           figsize='M-WS',
                           figname='hyperpm_boxmu'):
    # X .shape= (#num,)
    # Ys.shape= (nb_iter, #num)
    # z .shape= (nb_iter,)

    nb_iter = Ys.shape[0]  # nb_iter, num = Ys.shape
    fig, ax = plt.subplots(figsize=_setup_config['M-NT'])

    ax.boxplot(Ys, positions=X, patch_artist=patch_artist)  # bp=
    # ax.set_xticks(X)
    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')

    kws = {'color': '#F65F47', 'lw': 1}
    for i in range(nb_iter):
        plt.scatter(X, Ys[i], s=2.5, **kws)
    kws['color'] = '#465386'
    tz_avg, tz_std = np.mean(z), np.std(z, ddof=ddof)
    tx_min, tx_max = ax.get_xlim()
    ax.plot([tx_min, tx_max], [tz_avg, tz_avg],
            label='Real value', **kws)
    ax.fill_between(
        [tx_min, tx_max],
        [tz_avg - tz_std] * 2, [tz_avg + tz_std] * 2,
        alpha=.15, facecolor='#465386')

    kws['color'] = '#F65F47'
    tX = np.array([X] * nb_iter).reshape(-1)
    # tz = np.array([z] * num).T.reshape(-1)
    kws['linestyle'] = '--'
    _sns_line_err_bars(ax, kws, tX, Ys.reshape(-1))
    kws.pop('linestyle')
    plt.legend(loc=PLT_LOCATION, frameon=PLT_FRAMEBOX)

    ax.set_xlabel(annotX)
    ax.set_ylabel(annotY)
    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# -------------------------------
# fairmanf
#   for 若干个数据集放在一起


# '''
# def _diff_between_approx_and_direct(Yss, zs):
#     # difference: abs(approx - direct) / direct
#     # Yss.shape= (#att_sens, nb_iter, #num)
#     # zs .shape= (#att_sens, nb_iter)
#
#     nb_att, nb_iter, num = Yss.shape
#     diff = np.zeros_like(Yss) - 1.
#     for j in range(nb_att):
#         for i in range(nb_iter):
#             diff[j][i] = np.abs(Yss[j][i] - zs[j][i])
#             diff[j][i] /= check_zero(zs[j][i])
#     return diff
#
#
# def approximated_dist_comparison(X, Yss, zs, picked_keys,
#                                  figsize='M-WS',
#                                  figname='hyperpm_multi'):
#     # nb_att, nb_iter, num = Yss.shape
#     diff = _diff_between_approx_and_direct(Yss, zs)
# '''


def approximated_dist_comparison(
        X, Ys, picked_keys, annotX='pm',
        annotY=r'$\frac{abs(\hat{\mathbf{D}}-\mathbf{D})}{\mathbf{D}}$',
        figsize='M-WS', cmap_name='Dark2_r',  # 'Accent_r',
        figname='hyperpm_multi'):
    # X  : possible values of some certain hyper-parameter
    # Yss: results of 算法的估计值
    # zs : real value of 真实值

    # X  .shape= (#num,)  # 21 for m2, 24 for m1
    # Yss.shape= (#att_sen, #iter, #num)
    # zs .shape= (#att_sen, #iter)

    # Ys = abs(Yss - zs) / zs
    # Ys .shape= (#att_sen, #iter, #num)

    nb_att, nb_iter = Ys.shape[:2]  # nb_att,nb_iter,num=Ys.shape
    fig, ax = plt.subplots(figsize=_setup_config['M-NT'])

    # cs, cl = _setup_rgb_color(nb_iter, cmap_name)
    cs = sns.color_palette(cmap_name)  # cl = len(cs)
    kws = {'color': 'navy', 'lw': 1}
    if isinstance(X, list):
        tX = X * nb_iter
    else:
        # tX = np.repeat(X, nb_iter).reshape(num, -1).T.reshape(-1)
        tX = np.array([X] * nb_iter).reshape(-1)
    for i in range(nb_att):
        kws['color'] = cs[i]
        tYs = Ys[i].reshape(-1)  # .shape= (#num*nb_iter,)
        plt.scatter(tX, tYs, s=2.5, label=picked_keys[i], **kws)
        _sns_line_err_bars(ax, kws, tX, tYs)

    # bp = ax.boxplot()
    plt.legend(loc='best', frameon=True)
    ax.set_xlabel(annotX)
    ax.set_ylabel(annotY)
    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def multiple_scatter_comparison(X, Yss, zs, picked_keys,
                                annotX=r'hyper-pm',
                                annotY='Approximated value',
                                patch_artist=False, ddof=0,
                                cmap_name='Accent',
                                figsize='M-WS',  # scat
                                figname='hyperpm_bboxs'):
    # X  .shape= (#num,)
    # Yss.shape= (#att_sen, #iter, #num)
    # zs .shape= (#att_sen, #iter)
    # picked_keys.shape= (#att_sen,) list of names

    # nb_att, nb_iter, num = Yss.shape
    nb_att, nb_iter, _ = Yss.shape
    fig, ax = plt.subplots(figsize=_setup_config['M-NT'])
    cs = sns.color_palette(cmap_name)
    cl = len(cs)

    tX = np.array([X] * nb_iter).reshape(-1)
    for i in range(nb_att):
        kws = {'color': cs[i % cl], 'lw': 1}
        tYs = Yss[i].reshape(-1)
        plt.scatter(tX, tYs, s=2.5, **kws)
        kws['label'] = picked_keys[i]
        kws['linestyle'] = '--'
        _sns_line_err_bars(ax, kws, tX, tYs)

    tx_min, tx_max = ax.get_xlim()
    tz_avg = np.mean(zs, axis=1)            # (#att_sen,)
    # tz_std = np.std(zs, axis=1, ddof=ddof)  # (#att_sen,)
    for i in range(nb_att):
        kws = {'color': cs[i % cl], 'lw': 1}
        ax.plot([tx_min, tx_max], [tz_avg[i], tz_avg[i]], **kws)
        # '''
        # ax.fill_between([tx_min, tx_max],
        #             [tz_avg[i] - tz_std[i]] * 2,
        #             [tz_avg[i] + tz_std[i]] * 2,
        #             alpha=.15, facecolor=cs[i % cl])
        # '''

    ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    plt.legend(loc=PLT_LOCATION, frameon=PLT_FRAMEBOX)
    ax.set_xlabel(annotX)
    ax.set_ylabel(annotY)
    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# ===============================
# Python Matlablib plot


# -------------------------------
# fairmanf


# -------------------------------
# Linear regression with marginal distributions


def _marginal_distrib_step1(grid, df_all, col_X, current_palette):

    # 4.1 绘制长度的边缘分布图
    ax1 = plt.subplot(grid[0, 0: 5])
    ax1.spines[:].set_linewidth(.4)  # 设置坐标轴线宽
    ax1.tick_params(width=.6, length=2.5, labelsize=8
                    )  # 设置坐标轴刻度的宽度与长度、数值刻度的字体
    sns.kdeplot(data=df_all, x=col_X, hue='learning',
                fill=True, common_norm=False, legend=False,
                palette=current_palette, alpha=.5,
                linewidth=.5, ax=ax1)  # 边缘分布图
    # ax1.set_xlim(-75, 1575)
    ax1.set_xticks([])
    ax1.set_xlabel("")
    ax1.set_yticks([])
    ax1.set_ylabel("")

    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    sns.despine(ax=ax1, bottom=True,
                top=True, right=True, left=True)
    return ax1


def _marginal_distrib_step2(grid, df_all, col_Y, current_palette):

    # 4.2 绘制宽度的边缘分布图
    ax2 = plt.subplot(grid[1: 6, 5])
    ax2.spines[:].set_linewidth(.4)
    ax2.tick_params(width=.6, length=2.5, labelsize=8)
    sns.kdeplot(data=df_all, y=col_Y, hue='learning',
                fill=True, common_norm=False, legend=False,
                palette=current_palette, alpha=.5,
                linewidth=.5, ax=ax2)
    # ax2.set_ylim(-10, 210)
    ax2.set_xticks([])
    ax2.set_xlabel("")
    ax2.set_yticks([])
    ax2.set_ylabel("")

    sns.despine(ax=ax2, left=True,
                right=True, top=True, bottom=True)
    return ax2


def _marginal_distrib_step3(grid, dfs_pl, columns, col_X, col_Y,
                            # mycolor=None, annotX=r'X', annotY=r'Y'):
                            annotX=r'X', annotY=r'Y',  # mycolor=None,
                            # cmap_name='muted'):  # deep
                            mycolor=None, loc=None):
    _curr_sz = [30, 15, 15, 10, 10, 15]  # 20
    _curr_mark = ['*', '^', 'v', 'x', 'D', 'd']
    if len(columns) > 6:
        _curr_sz = [30, 15, 15, 10, 10, 10, 15, 15, 15]
        _curr_mark = ['*', '^', 'v', 'x', 'o', 'D', 'd', '<', '>']

    # 4.3 绘制二元分布图（散点图）
    ax3 = plt.subplot(grid[1: 6, 0: 5])
    ax3.spines[:].set_linewidth(.4)
    ax3.tick_params(width=.6, length=2.5, labelsize=8)
    ax3.grid(linewidth=.6, ls='-.', alpha=.4)

    for i, df in enumerate(dfs_pl):
        ax3.scatter(x=df[col_X], y=df[col_Y], s=_curr_sz[i], alpha=1,
                    marker=_curr_mark[i], color=mycolor[i],
                    edgecolors='w', linewidths=.5, label=columns[i])
    # _curr_font = 'Times New Roman'  # 'SimSun'
    _curr_font = plt.rcParams['font.family']
    legend_font = {'family': _curr_font, 'size': 8}
    _curr_nc = 1 if len(columns) <= 4 else 2

    loc_kws = {'loc': (.98, 1.01), 'frameon': False, 'ncol': _curr_nc
               # } if loc == None else {'loc': loc, 'frameon': True}
               } if loc is None else {'loc': loc, 'frameon': True}
    ax3.legend(
        prop=legend_font, labelspacing=.35,
        handleheight=1.2, handletextpad=0,
        columnspacing=.3, **loc_kws)
    del _curr_nc

    ax3.set_xlabel(annotX, fontsize=9, family=_curr_font, x=.55)
    ax3.set_ylabel(annotY, fontsize=9, family=_curr_font, y=.55)
    return ax3


def _marginal_distr_read_in(raw_df, col_X, col_Y, tag_Ys,
                            picked_keys):
    # 1. 读取数据
    dfs_pl = []

    # 2. 重组表格数据
    for i, tag_Y in enumerate(tag_Ys):
        tmp = raw_df[[col_X, tag_Y]].rename(columns={tag_Y: col_Y})
        tmp = tmp.apply(pd.to_numeric, errors='coerce')
        tmp['learning'] = picked_keys[i]
        dfs_pl.append(tmp)

    df_all = pd.concat(dfs_pl, axis=0).reset_index(drop=True)  # 合并表格
    return dfs_pl, df_all


def scatter_with_marginal_distrib(df, col_X, col_Y, tag_Ys,
                                  picked_keys,
                                  annotX='acc', annotY='fair',
                                  cmap_name='muted',
                                  figsize='M-WS', figname='smd'):
    # X .shape= (#num,)
    # Ys.shape= (#num, #baseline_for_comparison)
    # columns, i.e., picked_keys.shape= (#baseline,)
    # num, nb_way = Ys.shape

    dfs_pl, df_all = _marginal_distr_read_in(
        df, col_X, col_Y, tag_Ys, picked_keys)

    # 3. 设置seaborn颜色格式
    current_palette = sns.color_palette(cmap_name, len(picked_keys))
    sns.palplot(current_palette)

    # 4. 开始绘图
    fig = plt.figure(figsize=_setup_config[figsize], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)
    # columns = picked_keys

    # ax1, ax2, ax3 =
    _marginal_distrib_step1(grid, df_all, col_X, current_palette)
    _marginal_distrib_step2(grid, df_all, col_Y, current_palette)
    _marginal_distrib_step3(grid, dfs_pl, picked_keys, col_X, col_Y,
                            annotX, annotY, current_palette)

    # 5. 保存图片
    _setup_figshow(fig, figname=figname)
    plt.close()
    return


# -------------------------------
# Linear regression with marginal distributions


def _marginal_distr_step4(grid, dfs_pl, columns, col_X, col_Y,
                          annotX, annotY, mycolor, snspec='sty1',
                          identity=None, distrib=True,
                          curr_legend_nb_split=4):  # or 6
    _curr_sz = [30, 15, 15, 10, 10, 15]  # 20
    _curr_mk = ['*', '^', 'v', 'x', 'D', 'd']  # mark
    if len(columns) > 6:
        # _curr_sz = [30, 15, 15, 10, 10, 10, 15, 15, 15]
        # _curr_mk = ['*', '^', 'v', 'x', 'o', 'D', 'd', '<', '>']
        _curr_mk = ['D', '^', 'v', 'o', '*', 's', 'd', '<', '>']  # 'p'
        _curr_sz = [10, 15, 15, 12, 29, 11, 15, 15, 15]
    # _curr_mc = ['w'] * 3 + [None] + [''] * 5

    ax4 = plt.subplot(grid[1: 6, 0: 5])
    ax4.spines[:].set_linewidth(.4)
    ax4.tick_params(width=.6, length=2.5, labelsize=8)
    if distrib:
        ax4.grid(linewidth=.6, ls='-.', alpha=.4)

    # if snspec:
    if snspec == 'sty1':  # 's1', 'sns1'
        for i, df in enumerate(dfs_pl):
            tX = df[col_X].values.astype(DTY_FLT)
            tY = df[col_Y].values.astype(DTY_FLT)
            R = np.corrcoef(tX, tY)[1, 0]
            key = 'Correlation = %.4f' % R
            # regr = np.polyfit(tX, tY, deg=1)
            # estimated = np.polyval(regr, tX)

            ax4.scatter(x=df[col_X], y=df[col_Y], s=_curr_sz[i],
                        alpha=1,  # edgecolors=_curr_mc[i],
                        marker=_curr_mk[i], color=mycolor[i],
                        edgecolors='w', linewidths=.5,
                        label='{:4s} {}'.format(columns[i], key))
            # kws = {'color': mycolor[i], 'lw': .87, 'alpha': 1}
            # _sns_line_err_bars(ax4, kws, tX, tY)
            if identity is None:
                _sns_line_err_bars(ax4, {'color': mycolor[
                    i], 'lw': .87, 'alpha': 1}, tX, tY)

    elif snspec.startswith('sty5'):
        tmp_Xs, tmp_Ys = [], []
        for i, df in enumerate(dfs_pl):
            tX = df[col_X].values.astype(DTY_FLT)
            tY = df[col_Y].values.astype(DTY_FLT)
            R = np.corrcoef(tX, tY)[1, 0]
            key = 'Correlation = %.4f' % R
            ax4.scatter(x=df[col_X], y=df[col_Y], s=_curr_sz[i],
                        alpha=1,
                        marker=_curr_mk[i], color=mycolor[i],
                        edgecolors='w', linewidths=.5,
                        label='{:4s} {}'.format(columns[i], key))
            tmp_Xs.append(tX)
            tmp_Ys.append(tY)

        for i, (tX, tY) in enumerate(zip(tmp_Xs, tmp_Ys)):
            kws = {'color': mycolor[i], 'lw': .87, 'alpha': 1}
            _sns_line_err_bars(ax4, kws, tX, tY)
        del tX, tY, tmp_Xs, tmp_Ys
        if identity is not None:
            tx_min, tx_max = ax4.get_xlim()
            ax4.plot([tx_min, tx_max], [tx_min, tx_max], 'k--', lw=.5,
                     label=identity)  # label=r'identity')
            del tx_min, tx_max

    elif snspec == 'sty4a':
        tmp_Xs, tmp_Ys = [], []
        for i, df in enumerate(dfs_pl):
            tX = df[col_X].values.astype(DTY_FLT)
            tY = df[col_Y].values.astype(DTY_FLT)
            ax4.scatter(
                x=df[col_X], y=df[col_Y], s=_curr_sz[i], alpha=1,
                marker=_curr_mk[i], color=mycolor[i],
                edgecolors='w', linewidths=.5, label=columns[i])
            tmp_Xs.append(tX)
            tmp_Ys.append(tY)
        for i, (tX, tY) in enumerate(zip(tmp_Xs, tmp_Ys)):
            kws = {'color': mycolor[i], 'lw': .87, 'alpha': .78}
            _sns_line_err_bars(ax4, kws, tX, tY)
        del tX, tY, tmp_Xs, tmp_Ys
        if identity is not None:
            tx_min, tx_max = ax4.get_xlim()
            ax4.plot([tx_min, tx_max], [tx_min, tx_max], 'k--', lw=.5,
                     label=identity)  # label=r'identity')
            del tx_min, tx_max
    elif snspec == 'sty4b':
        tmp_Xs, tmp_Ys = [], []
        for i, df in enumerate(dfs_pl):
            tX = df[col_X].values.astype(DTY_FLT)
            tY = df[col_Y].values.astype(DTY_FLT)
            ax4.scatter(
                x=df[col_X], y=df[col_Y], s=_curr_sz[i], alpha=1,
                marker=_curr_mk[i], color=mycolor[i],
                edgecolors='w', linewidths=.5, label=columns[i])
            tmp_Xs.append(tX)
            tmp_Ys.append(tY)
        del tX, tY, tmp_Xs, tmp_Ys
        if identity is not None:
            tx_min, tx_max = ax4.get_xlim()
            ax4.plot([tx_min, tx_max], [tx_min, tx_max], 'k--', lw=.5,
                     label=identity)  # label=r'identity')
            del tx_min, tx_max

    elif snspec == 'sty3':
        for i, df in enumerate(dfs_pl):
            sns.regplot(x=col_X, y=col_Y, data=df, label=columns[i],
                        marker=_curr_mk[i], color=mycolor[i],
                        line_kws={'lw': .87},
                        scatter_kws={'s': _curr_sz[i] / 4})

    elif snspec == 'sty2':
        for i, df in enumerate(dfs_pl):
            tX = df[col_X].values.astype(DTY_FLT)
            tY = df[col_Y].values.astype(DTY_FLT)
            R = np.corrcoef(tX, tY)[1, 0]
            key = 'Correlation = %.4f' % R
            # regr = np.polyfit(tX, tY, deg=1)
            # estimated = np.polyval(regr, tX)
            ax4.scatter(tX, tY,
                        label='{:4s} {}'.format(columns[i], key),
                        s=_curr_sz[i] / 4, marker=_curr_mk[i],
                        color=mycolor[i])
            kws = {'color': mycolor[i], 'lw': .87, 'alpha': 1}
            _sns_line_err_bars(ax4, kws, tX, tY)

    _curr_ft = plt.rcParams['font.family']  # 'Times New Roman'
    legend_font = {'family': _curr_ft, 'size': 7}
    _curr_lc = (.98, 1.01) if distrib else PLT_LOCATION
    _curr_nc = 1 if len(columns) <= curr_legend_nb_split else 2
    # _curr_nc = 1 if len(columns) <= 4 else 2  # <=4,6
    if snspec == 'sty5b':
        _curr_nc = 1
    _curr_kw = {'frameon': False}
    if not distrib:
        _curr_kw['frameon'] = True
        _curr_kw['framealpha'] = .7
    if _curr_nc >= 2:
        _curr_kw['columnspacing'] = .4
    ax4.legend(
        prop=legend_font, labelspacing=.35, handleheight=1.2,
        handletextpad=0, loc=_curr_lc, ncol=_curr_nc, **_curr_kw)
    del _curr_lc, _curr_nc, _curr_kw
    if not distrib:
        _style_set_axis(ax4)

    ax4.set_xlabel(annotX, fontsize=9, family=_curr_ft, x=.55)
    ax4.set_ylabel(annotY, fontsize=9, family=_curr_ft, y=.55)
    return ax4


def _marginal_distr_step5(grid, dfs_pl, col_X, col_Y, mycolor):
    if col_X is not None:

        # 4.1 绘制长度的边缘分布图
        ax1 = plt.subplot(grid[0, 0: 5])
        ax1.spines[:].set_linewidth(.4)  # 设置坐标轴线宽
        ax1.tick_params(width=.6, length=2.5, labelsize=8
                        )  # 设置坐标轴刻度的宽度与长度、数值刻度的字体
        for i, df in enumerate(dfs_pl):
            sns.kdeplot(data=df, x=col_X, fill=True,
                        common_norm=False, legend=False, color=mycolor[i],
                        alpha=.5, linewidth=.5, ax=ax1)  # 边缘分布图
        ax1.set_xticks([])
        ax1.set_xlabel("")
        ax1.set_yticks([])
        ax1.set_ylabel("")
        sns.despine(ax=ax1, left=True, bottom=True)

    else:
        ax1 = None
    if col_Y is not None:

        # 4.2 绘制宽度的边缘分布图
        ax2 = plt.subplot(grid[1: 6, 5])
        ax2.spines[:].set_linewidth(.4)
        ax2.tick_params(width=.6, length=2.5, labelsize=8)
        for i, df in enumerate(dfs_pl):
            sns.kdeplot(
                data=df, y=col_Y, fill=True,
                common_norm=False, legend=False, color=mycolor[i],
                alpha=.5, linewidth=.5, ax=ax2)
        ax2.set_xticks([])
        ax2.set_xlabel("")
        ax2.set_yticks([])
        ax2.set_ylabel("")
        sns.despine(ax=ax2, left=True, bottom=True)

    else:
        ax2 = None
    return ax1, ax2


def line_reg_with_marginal_distr(df, col_X, col_Y, tag_Ys,
                                 picked_keys,
                                 annotX='acc', annotY='fair',
                                 invt_a=False, snspec='sty0',
                                 distrib=True,
                                 cmap_name='muted',
                                 figsize='M-WS', figname='smd',
                                 identity=None,
                                 curr_legend_nb_split=4):
    dfs_pl, df_all = _marginal_distr_read_in(
        df, col_X, col_Y, tag_Ys, picked_keys)
    mycolor = sns.color_palette(cmap_name, len(picked_keys))

    fig = plt.figure(figsize=_setup_config[figsize], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)

    if invt_a:
        col_X, col_Y = col_Y, col_X
        annotX, annotY = annotY, annotX
    if distrib:
        # ax1 = _marginal_distrib_step1(grid, df_all, col_X, mycolor)
        # ax2 = _marginal_distrib_step2(grid, df_all, col_Y, mycolor)
        _marginal_distrib_step1(grid, df_all, col_X, mycolor)
        _marginal_distrib_step2(grid, df_all, col_Y, mycolor)
    if snspec == 'sty0':
        # ax3 =
        _marginal_distrib_step3(grid, dfs_pl, picked_keys,
                                col_X, col_Y, annotX, annotY,
                                mycolor)  # , distrib=distrib)
    elif snspec in ['sty1', 'sty2', 'sty3',  # 'sty4', 'sty5',
                    'sty6', 'sty4a', 'sty4b', 'sty5a', 'sty5b']:
        # ax4 =
        _marginal_distr_step4(
            grid, dfs_pl, picked_keys, col_X, col_Y,
            annotX, annotY, mycolor, snspec,
            identity=identity, distrib=distrib,
            curr_legend_nb_split=curr_legend_nb_split)

    _setup_figshow(fig, figname=figname)
    plt.close(fig)
    return


def single_line_reg_with_distr(X, Y, annots=('X', 'Y', 'Z'),
                               figname='sing_linreg', figsize='M-WS',
                               linreg=False, distrib=False,
                               snspec='sty2', cmap_name='coolwarm',
                               sci_format_y=False):
    # mycolor = sns.color_palette(cmap_name)

    fig = plt.figure(figsize=_setup_config[figsize], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)

    if distrib:
        df = pd.DataFrame({'x': X, 'y': Y})
        kwd = {
            'fill': True,
            'common_norm': False, 'legend': False,
            # 'palette': mycolor,
            'alpha': .5, 'linewidth': .5}

        # def _marginal_distrib_step1:
        ax1 = plt.subplot(grid[0, 0: 5])
        ax1.spines[:].set_linewidth(.4)
        ax1.tick_params(width=.6, length=2.5, labelsize=8)
        sns.kdeplot(data=df, x='x', ax=ax1, **kwd)
        ax1.set_xticks([])
        ax1.set_xlabel("")
        ax1.set_yticks([])
        ax1.set_ylabel("")
        sns.despine(ax=ax1, left=True, bottom=True)

        # def _marginal_distrib_step2:
        ax2 = plt.subplot(grid[1: 6, 5])
        ax2.spines[:].set_linewidth(.4)
        ax2.tick_params(width=.6, length=2.5, labelsize=8)
        sns.kdeplot(data=df, y='y', ax=ax2, **kwd)
        ax2.set_xticks([])
        ax2.set_xlabel("")
        ax2.set_yticks([])
        ax2.set_ylabel("")
        sns.despine(ax=ax2, left=True, bottom=True)

    # def _marginal_distr_step4:
    ax4 = plt.subplot(grid[1: 6, 0: 5])
    ax4.spines[:].set_linewidth(.4)
    ax4.tick_params(width=.6, length=2.5, labelsize=8)
    # ax4.grid(linewidth=.6, ls='-.', alpha=.4)
    _curr_ft = plt.rcParams['font.family']  # 'Times New Roman'
    legend_font = {'family': _curr_ft, 'size': 8}

    R = np.corrcoef(X, Y)[1, 0]
    key = 'Correlation = %.4f' % R
    regr = np.polyfit(X, Y, deg=1)
    estimated = np.polyval(regr, X)
    Z = sorted(X)
    annotZ = annots[2] if len(annots) > 2 else r'$f(x)=x$'

    if distrib:
        sns.regplot(
            x='x', y='y', data=df, label=key,
            # line_kws={'lw': .87}, scatter_kws={'s': 10})
            scatter_kws={'s': 27, 'edgecolors': 'w',
                         'lw': .1},  # , 'color': 'blue'},
            line_kws={'lw': 1})
        if snspec == 'sty5':
            plt.plot(Z, Z, 'k--', lw=1,
                     label='{:4s}{}'.format('', annotZ))
            plt.plot(X, estimated, '-', lw=1, color='navy')
        ax4.legend(
            prop=legend_font, labelspacing=.35, handleheight=1.2,
            # handletextpad=0, loc=(.98, 1.01), frameon=False)
            handletextpad=0, loc='best', frameon=False)
    elif linreg:  # if snspec == 'sty1':

        if snspec == 'sty2':
            # ax4.scatter(X, Y, label=key)
            ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                        linewidths=.5, label=key)  # s='.',
            plt.plot(X, estimated, 'k-', lw=1)
        elif snspec == 'sty3a':
            ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                        linewidths=.4)
            plt.plot(X, estimated, '-', lw=1, label=key, color='navy')
            plt.plot(Z, Z, 'k--', lw=1, label=annotZ)
            # plt.plot(X, estimated, '-', lw=1, label=key, color='navy')
        elif snspec == 'sty3b':
            ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                        linewidths=.4, label=key)
            plt.plot(X, estimated, '-', lw=1, color='navy')
            plt.plot(Z, Z, 'k--', lw=1, label=annotZ)
        elif snspec == 'sty4':
            ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                        linewidths=.2, s=27)
            plt.plot(Z, Z, 'k--', lw=1, label=annotZ)
        elif snspec == 'sty6':
            ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                        linewidths=.4)
            tx_min, tx_max = ax4.get_xlim()
            annotZ = r'$\hat{\mathbf{D}}_{\cdot}=\mathbf{D}_{\cdot}$'
            if len(annots) > 2:
                annotZ = annots[2]
            ax4.plot([tx_min, tx_max], [0, 0], 'k--', lw=1, label=annotZ)
            del tx_min, tx_max

        # if snspec != 'sty4':
        if snspec not in ['sty4', 'sty6']:
            kws = {'color': 'navy', 'lw': 1}
            _sns_line_err_bars(ax4, kws, X, Y)
        ax4.legend(prop=legend_font, loc='best', frameon=False)
        _style_set_axis(ax4)
    del annotZ, Z

    ax4.set_xlabel(annots[0], fontsize=9, family=_curr_ft, x=.55)
    ax4.set_ylabel(annots[1], fontsize=9, family=_curr_ft, y=.55)
    # def end.
    if sci_format_y:  # if sci_y_format:
        # ax4.ticklabel_format(style='sci', scilimits=(-2, 3), axis='y')
        ax4.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
        ax4.yaxis.get_offset_text().set_fontsize(8)  # 7)

    _setup_figshow(fig, figname=figname)
    plt.close(fig)
    return


# -------------------------------
# shaded uncertainty region in line plot


def _uncertainty_read_in(dfs_pl, col_X, col_Y, num_gap=1000,
                         alpha_loc='b4|af', alpha_rev=False):
    # dfs_pl: list of pd.DataFrame,
    #         len()= #baseline_for_comparison
    # each element is pd.DataFrame,
    #         columns= ['col_X', 'col_Y', 'learning']
    #         shape  = (#num, 3)

    X = np.linspace(0, 1, num_gap)
    baseline_Ys = []
    for df in dfs_pl:  # for i, df in enumerate(dfs_pl):
        Ys = []
        for alpha in X:
            # '''
            # if (alpha_loc == 'b4') and (not alpha_rev):
            #     tmp = df[col_X] * alpha + (1. - alpha) * df[col_Y]
            # elif (alpha_loc == 'af') and (not alpha_rev):
            #     tmp = df[col_X] * (1. - alpha) + alpha * df[col_Y]
            # elif (alpha_loc == 'b4') and alpha_rev:  # reverse
            #     tmp = (1 - df[col_X]) * alpha + (1. - alpha) * df[col_Y]
            # elif (alpha_loc == 'af') and alpha_rev:  # reverse
            #     tmp = (1 - df[col_X]) * (1. - alpha) + alpha * df[col_Y]
            # '''

            # assert alpha_loc in ('b4', 'af')
            if (not alpha_rev) and (alpha_loc == 'b4'):
                tmp = df[col_X] * alpha + (1. - alpha) * df[col_Y]
            elif not alpha_rev:           # (alpha_loc == 'af') and
                tmp = df[col_X] * (1. - alpha) + alpha * df[col_Y]
            elif alpha_loc == 'b4':     # and alpha_rev:  # reverse
                tmp = (1 - df[col_X]) * alpha + (1. - alpha) * df[col_Y]
            else:  # if (alpha_loc=='af') and alpha_rev:  # reverse
                tmp = (1 - df[col_X]) * (1. - alpha) + alpha * df[col_Y]
            Ys.append(tmp.values)  # (#num,)
        baseline_Ys.append(Ys)   # (num_gap, #num)
    # baseline_Ys.shape= (#baseline, #gap, #num)

    # baseline_Ys = np.array(baseline_Ys, dtype='float')
    # return X, baseline_Ys, np.mean(
    #     baseline_Ys, axis=2), np.std(baseline_Ys, axis=2, ddof=ddof)
    return X, np.array(baseline_Ys, dtype=DTY_FLT)


def _sub_unc_text(annotY, alpha_loc):
    # if annotY is None:
    #     if alpha_loc == 'b4':
    #         annotY = r'$\alpha·$ performance $+(1-\alpha)·$ fairness'
    #     elif alpha_loc == 'af':
    #         annotY = r'$(1-\alpha)·$ performance $+\alpha·$ fairness'
    # else:
    #     if alpha_loc == 'b4':
    #         annotY = r'$\alpha·${} $+($1$-\alpha)·$ fairness'.format(annotY)
    #     elif alpha_loc == 'af':
    #         annotY = r'$($1$-\alpha)·${} $+\alpha·$ fairness'.format(annotY)

    assert alpha_loc in ['b4', 'af']
    if annotY is None:
        if alpha_loc == 'b4':
            annotY = r'$\alpha·$ performance $+(1-\alpha)·$ fairness'
        elif alpha_loc == 'af':
            annotY = r'$(1-\alpha)·$ performance $+\alpha·$ fairness'
        return annotY
    if alpha_loc == 'b4':
        annotY = r'$\alpha·${} $+($1$-\alpha)·$ fairness'.format(annotY)
    elif alpha_loc == 'af':
        annotY = r'$($1$-\alpha)·${} $+\alpha·$ fairness'.format(annotY)
    return annotY    # CC 7->6


def _uncertainty_plotting(X, Ys, picked_keys, annotY=None, ddof=0,
                          alpha_loc='b4|af', cmap_name='husl',
                          figsize='M-WS', figname='lwu',
                          alpha_clarity=.3):
    '''
    _curr_sty = ['-.', '-.', '-.', '--', '-', '-']
    if len(picked_keys) > 6:
        # _curr_sty = ['-.', '-.', '-.', '--', '-', '-', ':', '-', ':']
        _curr_sty.extend([':', '-', ':'])
    '''
    _curr_sty = ['-.', '-.', '-.', '--', '-', '-'] + [':', '-', ':']

    # X                                     # (#gap,)
    Ys_avg = np.mean(Ys, axis=2)            # (#baseline, #gap)
    Ys_std = np.std(Ys, axis=2, ddof=ddof)  # (#baseline, #gap)
    # picked_keys                           # (#baseline,)

    fig, ax = plt.subplots(figsize=_setup_config['M-NT'])
    # clrs = sns.color_palette(cmap_name, len(picked_keys))
    # with sns.axes_style('darkgrid'):
    #   # epochs = list(range(101))
    clrs = sns.color_palette(cmap_name, len(picked_keys))  # palette=)

    for i, _ in enumerate(picked_keys):  # _:key
        ax.plot(X, Ys_avg[i],
                _curr_sty[i], label=picked_keys[i], c=clrs[i], lw=1)
        # ax.plot(X, Ys_avg[i], label=picked_keys[i], c=clrs[i])
        ax.fill_between(
            X, Ys_avg[i] - Ys_std[i], Ys_avg[i] + Ys_std[i],
            # alpha=.3, facecolor=clrs[i])
            alpha=alpha_clarity, facecolor=clrs[i])  # .1
    # ax.legend()
    # ax.set_yscale('log')

    ax.legend(labelspacing=.1, prop={
        'size': 9 if len(picked_keys) <= 6 else 8})
    ax.set_xlim(X[0], X[-1])
    ax.set_xlabel(r'$\alpha$')
    annotY = _sub_unc_text(annotY, alpha_loc)

    # '''
    # assert alpha_loc in ('b4', 'af')
    # if (annotY is None) and (alpha_loc == 'b4'):
    #     annotY = r'$\alpha·$ performance $+(1-\alpha)·$ fairness'
    # elif annotY is None:  # and (alpha_loc == 'af'):
    #     annotY = r'$(1-\alpha)·$ performance $+\alpha·$ fairness'
    # elif alpha_loc == 'b4':
    #     annotY = r'$\alpha·${} $+($1$-\alpha)·$ fairness'.format(annotY)
    # else:  # alpha_loc == 'af':
    #     annotY = r'$($1$-\alpha)·${} $+\alpha·$ fairness'.format(annotY)
    # '''
    ax.set_ylabel(annotY, fontsize=9)
    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def lineplot_with_uncertainty(df, col_X, col_Y, tag_Ys, picked_keys,
                              # annotX='acc', annotY='fair',
                              annotY=None, ddof=0, num_gap=100,
                              alpha_loc='b4', cmap_name='husl',
                              alpha_rev=True,  # middle of annotY
                              figsize='M-WS', figname='lwu',
                              alpha_clarity=.3):
    dfs_pl, _ = _marginal_distr_read_in(  # ,df_all
        df, col_X, col_Y, tag_Ys, picked_keys)  # alp_loc/rev
    X, Ys = _uncertainty_read_in(dfs_pl, col_X, col_Y, num_gap=num_gap,
                                 alpha_loc=alpha_loc, alpha_rev=alpha_rev)
    kwargs = {'figsize': figsize, 'figname': figname}
    _uncertainty_plotting(X, Ys, picked_keys, annotY, ddof,
                          alpha_loc=alpha_loc,
                          alpha_clarity=alpha_clarity,
                          cmap_name=cmap_name, **kwargs)


# -------------------------------


# ===============================
# Python Matlablib plot


# -------------------------------
# fairmanf ext.(extension)


# -------------------------------
# refers to:
#   def single_line_reg_with_distr():
#


def _subproc_pl_lin_reg(ax4, X, Y, Z, annotZ, snspec, clr='navy',
                        reverse=False, corr=True, sz=None, mk=None):
    if Y is None:
        return

    R = np.corrcoef(X, Y)[1, 0]
    key = 'Correlation = %.4f' % R
    regr = np.polyfit(X, Y, deg=1)
    estimated = np.polyval(regr, X)
    # # Z = sorted(X)
    # '''
    # if not reverse:
    #     key = '{} {}'.format(key, Z)  # Z, key)
    # elif reverse:
    #     key = '{:9s} {}'.format(Z, key)
    # '''
    key = '{} {}'.format(key, Z) if (
        not reverse) else '{:9s} {}'.format(Z, key)

    if snspec == 'sty3a':
        ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                    linewidths=.4, color=clr)
        plt.plot(X, estimated, '-', lw=1, label=key, color=clr)
        # plt.plot(Z, Z, 'k--', lw=1, label=annotZ)
    elif snspec == 'sty3b':
        ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                    linewidths=.4, label=key, color=clr)
        plt.plot(X, estimated, '-', lw=1, color=clr)
    elif snspec == 'sty4':
        ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                    linewidths=.2, s=27, label=Z, color=clr)
    elif snspec == 'sty6':
        ax4.scatter(x=X, y=Y, alpha=1, edgecolors='w',
                    linewidths=.4, label=Z, color=clr)

    elif snspec == 'sty5a':
        ax4.scatter(x=X, y=Y, s=sz, marker=mk, alpha=1,
                    edgecolors='w', linewidths=.5, color=clr)
        ax4.plot(X, estimated, '-', lw=1,
                 label=key if corr else Z, color=clr)
    elif snspec == 'sty5b':
        ax4.scatter(x=X, y=Y, s=sz, marker=mk, alpha=1,
                    edgecolors='w', linewidths=.5, color=clr,
                    label=key if corr else Z)
        ax4.plot(X, estimated, '-', lw=1, color=clr)

    return


def _subproc_pl_lin_reg_alt(ax4, X, Y, snspec, clr='navy'):
    if Y is None:
        return
    if snspec not in ['sty4', 'sty6']:
        kws = {'color': clr, 'lw': 1}
        _sns_line_err_bars(ax4, kws, X, Y)
    return


def _subproc_pl_identity(ax4, Xs, annotZ, snspec):
    # Zs = [sorted(X) for X in Xs]  # , clr='navy'):
    tX = np.concatenate(Xs, axis=0)
    Z = sorted(tX)  # tZ = sorted(tX)
    if snspec in ['sty3a', 'sty3b', 'sty4',
                  'sty5a', 'sty5b']:
        plt.plot(Z, Z, 'k--', lw=1, label=annotZ)
    elif snspec == 'sty6':
        tx_min, tx_max = ax4.get_xlim()
        ax4.plot([tx_min, tx_max], [0, 0], 'k--', lw=1, label=annotZ)
        del tx_min, tx_max
    del Z, tX
    return


_pl_myclr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


def multi_lin_reg_with_distr(Xs, Ys, Zs, annots=('X', 'Y', 'Z'),
                             figname='pl_linreg', figsize='M-WS',
                             # linreg=False,  # distrib=False,
                             snspec='sty2', cmap_name='coolwarm',
                             sci_format_y=True):  # default:False
    # mycolor = sns.color_palette(cmap_name)
    fig = plt.figure(figsize=_setup_config[figsize], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)

    ax4 = plt.subplot(grid[1: 6, 0: 5])
    ax4.spines[:].set_linewidth(.4)
    ax4.tick_params(width=.6, length=2.5, labelsize=8)
    _curr_ft = plt.rcParams['font.family']  # 'Times New Roman'
    legend_font = {'family': _curr_ft, 'size': 8}

    annotZ = r'$\hat{\mathbf{D}}_\cdot=\mathbf{D}_\cdot$'
    annotZ = annots[2] if len(annots) > 2 else r'$f(x)=x$'
    # myclr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    #          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    myclr = _pl_myclr

    _subproc_pl_lin_reg(
        ax4, Xs[0], Ys[0][0], Zs[0][0], annotZ, snspec, myclr[0])
    _subproc_pl_lin_reg(
        ax4, Xs[0], Ys[0][1], Zs[0][1], annotZ, snspec, myclr[1])
    _subproc_pl_lin_reg(
        ax4, Xs[1], Ys[1], Zs[1], annotZ, snspec, myclr[2])
    _subproc_pl_lin_reg_alt(ax4, Xs[0], Ys[0][0], snspec, clr=myclr[0])
    _subproc_pl_lin_reg_alt(ax4, Xs[0], Ys[0][1], snspec, clr=myclr[1])
    _subproc_pl_lin_reg_alt(ax4, Xs[1], Ys[1], snspec, clr=myclr[2])
    _subproc_pl_identity(ax4, Xs, annotZ, snspec)

    if snspec in ['sty3a', 'sty3b', ]:
        _curr_fram = {'frameon': False, 'loc': 'upper left'}
    elif snspec in ['sty6', ]:
        _curr_fram = {'frameon': True, 'framealpha': .5,
                      'loc': 'upper right'}  # 'loc': 'best'}
    elif snspec in ['sty4']:
        _curr_fram = {'loc': 'best', 'frameon': True, 'framealpha': .5}
    ax4.legend(prop=legend_font, labelspacing=.35, **_curr_fram)
    if sci_format_y:
        ax4.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
        ax4.yaxis.get_offset_text().set_fontsize(8)  # 7)
    _style_set_axis(ax4)
    ax4.set_xlabel(annots[0], fontsize=9, family=_curr_ft, x=.55)
    ax4.set_ylabel(annots[1], fontsize=9, family=_curr_ft, y=.55)
    _setup_figshow(fig, figname=figname)
    plt.close(fig)
    return


# -------------------------------
# refers to
#   def multi_lin_reg_with_distr
#


def multi_lin_reg_without_distr(X, Ys, Zs, annots=('X', 'Y', 'Z'),
                                figname='pl_linreg', figsize='M-WS',
                                snspec='sty4',  # cmap_names='coolwarm',
                                sci_format_y=False):
    fig = plt.figure(figsize=_setup_config[figsize], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)
    ax4 = plt.subplot(grid[1: 6, 0: 5])
    ax4.spines[:].set_linewidth(.4)
    ax4.tick_params(width=.6, length=2.5, labelsize=8)
    _curr_ft = plt.rcParams['font.family']  # 'Times New Roman'
    legend_font = {'family': _curr_ft, 'size': 8}

    myclr = _pl_myclr
    annotZ = annots[2] if len(annots) > 2 else r'$f(x)=x$'
    n_k = len(Ys)  # aka. len(Zs)
    start_i = 2 if n_k == 2 else 1
    for i in range(n_k):
        _subproc_pl_lin_reg(ax4, X, Ys[i], Zs[i], annotZ,
                            snspec, myclr[i + start_i])
    for i in range(n_k):
        _subproc_pl_lin_reg_alt(
            ax4, X, Ys[i], snspec, myclr[i + start_i])
    _subproc_pl_identity(ax4, [X, X], annotZ, snspec)

    if snspec in ['sty3a', 'sty3b', ]:
        _curr_fram = {'frameon': False, 'loc': 'upper left',
                      'framealpha': .5}  # 'loc': 'best',
    elif snspec in ['sty6', ]:
        _curr_fram = {'frameon': True, 'framealpha': .5,
                      'loc': 'upper right'}  # 'loc': 'best'}
    elif snspec in ['sty4']:
        _curr_fram = {'loc': 'best', 'frameon': True, 'framealpha': .5}
    ax4.legend(prop=legend_font, labelspacing=.35, **_curr_fram)
    if sci_format_y:
        ax4.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
        ax4.yaxis.get_offset_text().set_fontsize(8)
    _style_set_axis(ax4)
    ax4.set_xlabel(annots[0], fontsize=9, family=_curr_ft, x=.55)
    ax4.set_ylabel(annots[1], fontsize=9, family=_curr_ft, y=.55)
    _setup_figshow(fig, figname=figname)
    plt.close(fig)
    return


# -------------------------------


def scatter_parl_chart_renew(centralised, distributed, mp_cores=3,
                             figname='parl', figsize='M-WS',
                             identity=True):
    speed_up = np.divide(centralised, distributed)
    efficiency = speed_up / mp_cores
    annotY = [r'Speedup = $\frac{ T }{ T_{par} }$',
              r'Efficiency = $\frac{T}{m T_{par}}$']
    annotX = r'Sequential running time $T$ (sec)'
    picked_key = r'$m$ = {}'.format(mp_cores)

    fig = plt.figure(figsize=_setup_config['L-NT'])
    plt.scatter(centralised, speed_up, s=19, c='royalblue',
                label=picked_key, edgecolors='w', linewidths=.2)
    if identity:
        annotZ = 'Speedup =1'  # 'speedup =1'
        tx_min, tx_max = fig.gca().get_xlim()
        plt.plot([tx_min, tx_max], [1, 1], 'k--', lw=.8, label=annotZ)
    # plt.xlabel('No parallel computing')
    # plt.ylabel('Computing in parallel')
    plt.xlabel(annotX)
    plt.ylabel(annotY[0])  # plt.ylabel('Speedup')
    plt.legend(loc=PLT_LOCATION, labelspacing=.1, frameon=True)
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname + '_parl_sp')
    plt.clf()  # fig)

    plt.scatter(centralised, efficiency, s=19, c='slateblue',
                label=picked_key, edgecolors='w', linewidths=.2)
    if identity:
        annotZ = 'Efficiency =1'  # 'efficiency =1'
        plt.plot([tx_min, tx_max], [1, 1], 'k--', lw=.8, label=annotZ)
        del tx_min, tx_max
    plt.xlabel(annotX)
    plt.ylabel(annotY[1])  # plt.ylabel('Efficiency')
    plt.legend(loc=PLT_LOCATION, labelspacing=.1, frameon=True)
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname + '_parl_ep')
    plt.close(fig)
    return


# def _hyper_pm_step4(ax4, snspec):
#   pass
def hyper_params_lin_reg(X, Ys, tag_Ys, picked_keys,
                         annots=('acc', 'fair', 'Z'),
                         figname='smd', figsize='M-WS',
                         distrib=False, identity=True,
                         sci_format_y=False, corr=False,
                         curr_legend_nb_split=4,
                         cmap_name='muted', snspec='sty5b'):
    mycolor = sns.color_palette(cmap_name, len(picked_keys))
    fig = plt.figure(figsize=_setup_config[figsize], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)

    _curr_mk = ['D', '^', 'v', 'o', '*', 's', 'd', '<', '>']
    _curr_sz = [10, 15, 15, 12, 29, 11, 15, 15, 15]
    ax4 = plt.subplot(grid[1: 6, 0: 5])
    ax4.spines[:].set_linewidth(.4)
    ax4.tick_params(width=.6, length=2.5, labelsize=8)
    if distrib:
        ax4.grid(linewidth=.6, ls='-.', alpha=.4)

    annotZ = annots[2] if len(annots) > 2 else r'$f(x)=x$'
    # identity
    # nb_choice = len(picked_keys)
    # for i in range(nb_choice):
    for i, key in enumerate(picked_keys):
        _subproc_pl_lin_reg(
            ax4, X, Ys[key], tag_Ys[key], annotZ, snspec,
            mycolor[i], True, corr, _curr_sz[i], _curr_mk[i])
    for i, key in enumerate(picked_keys):
        _subproc_pl_lin_reg_alt(ax4, X, Ys[key], snspec, mycolor[i])
    if identity:
        _subproc_pl_identity(ax4, [X, X], annotZ, snspec)

    if snspec in ['sty3a', 'sty3b']:
        _curr_fram = {'frameon': False, 'loc': 'upper left'}
    elif snspec in ['sty5a', 'sty5b']:
        _curr_nc = 1 if len(picked_keys) <= curr_legend_nb_split else 2
        _curr_fram = {
            'frameon': True, 'loc': 'upper left', 'framealpha': .5,
            'handleheight': 1.2, 'handletextpad': 0, 'ncol': _curr_nc}
        del _curr_nc
    elif snspec in ['sty6', ]:
        _curr_fram = {'frameon': True, 'framealpha': .5,
                      'loc': 'upper right'}  # 'loc': 'best'}
    elif snspec in ['sty4']:
        _curr_fram = {'loc': 'best', 'frameon': True, 'framealpha': .5}
    _curr_ft = plt.rcParams['font.family']  # 'Times New Roman'
    legend_font = {'family': _curr_ft, 'size': 8}
    ax4.legend(prop=legend_font, labelspacing=.35, **_curr_fram)
    if sci_format_y:
        ax4.ticklabel_format(style='sci', scilimits=(-3, 4), axis='y')
        ax4.yaxis.get_offset_text().set_fontsize(8)  # 7)
    _style_set_axis(ax4)
    ax4.set_xlabel(annots[0], fontsize=9, family=_curr_ft, x=.55)
    ax4.set_ylabel(annots[1], fontsize=9, family=_curr_ft, y=.55)
    del _curr_ft, legend_font, _curr_fram, _curr_sz, _curr_mk

    _setup_figshow(fig, figname=figname)
    plt.close(fig)
    return


# -------------------------------

# -------------------------------


# ===============================
# RR plot


# -------------------------------

# -------------------------------

# -------------------------------


# ===============================
# FairGBM
# https://arxiv.org/pdf/2209.07850


# -------------------------------
# Figure 2(a)


def FairGBM_scatter(Xs, Ys, annot, label=('X', 'Y'),
                    cmap_name='deep',  # 'colorblind',
                    figname='FairGBM', figsize='M-WS'):
    # Xs.shape  (#model, #experiment /iteration)
    # Ys.shape  (#model, #experiment /iteration)
    curr_palette = sns.color_palette(cmap_name, len(annot))
    sns.palplot(curr_palette)  # current color theme

    dfs_pl = []
    for i, ant in enumerate(annot):
        tmp = pd.DataFrame({'x': Xs[i], 'y': Ys[i]})
        # tmp = tmp.apply(pd.to_numeric, errors='coerce')
        tmp['learning'] = ant
        dfs_pl.append(tmp)
    df_all = pd.concat(dfs_pl, axis=0).reset_index(drop=True)

    fig = plt.figure(figsize=_setup_config[figsize], dpi=300)
    plt.subplots_adjust(left=.11, bottom=.11, right=.98, top=.995)
    grid = plt.GridSpec(6, 6, wspace=.05, hspace=.05)
    # ax1, ax2, ax3 =
    _marginal_distrib_step1(grid, df_all, 'x', curr_palette)
    _marginal_distrib_step2(grid, df_all, 'y', curr_palette)
    _marginal_distrib_step3(grid, dfs_pl, annot, 'x', 'y',
                            label[0], label[1], curr_palette,
                            loc='best')

    _setup_figshow(fig, figname=figname)
    plt.close()
    return


# -------------------------------
# Figure 2(b)


def FairGBM_tradeoff_v1(X, Y, annot, label=('X', 'Y'),
                        num_gap=1000,
                        alpha_loc='b4|af', alpha_rev=False,
                        alpha_clarity=.15, cmap_name='colorblind',
                        figname='FairGBM', figsize='M-WS'):
    # X.shape  (#model,)
    # Y.shape  (#model,)

    Z = np.linspace(0, 1, num_gap)
    baseline_Ys = []
    # for i, ant in enumerate(annot):
    #   tmp = []
    for alpha in Z:
        if (alpha_loc == 'b4') and (not alpha_rev):
            tmp = X * alpha + (1. - alpha) * Y
            label_y = r'$\alpha·$ performance $+(1-\alpha)·$ fairness'
        elif (alpha_loc == 'af') and (not alpha_rev):
            tmp = X * (1. - alpha) + alpha * Y
            label_y = r'$(1-\alpha)·$ performance $+\alpha·$ fairness'
        elif (alpha_loc == 'b4') and alpha_rev:
            tmp = (1. - X) * alpha + (1. - alpha) * Y
            label_y = r'$\alpha·${} $+($1$-\alpha)·$ {}'.format(*label)
        elif (alpha_loc == 'af') and alpha_rev:
            tmp = (1. - X) * (1. - alpha) + alpha * Y
            label_y = r'$($1$-\alpha)·${} $+\alpha·$ {}'.format(*label)
        # tmp.shape  (#model,)
        baseline_Ys.append(tmp)
    baseline_Ys = np.array(baseline_Ys).transpose()
    # baseline_Ys.shape  (#model, #gap)

    curr_palette = sns.color_palette(cmap_name, len(annot))
    sns.palplot(curr_palette)
    _curr_sty = ['-.', '-.', '-.', '--', '-', '-', ':', '-', ':']
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    kws = {'color': 'navy', 'lw': 1}
    for i, ant in enumerate(annot):
        kws['color'] = curr_palette[i]
        kws['label'] = ant  # annot[i]
        kws['linestyle'] = _curr_sty[i]
        _sns_line_err_bars(ax, kws, Z, baseline_Ys[i])

    ax.set_xlabel(r'$\alpha$')  # label[0])
    ax.set_ylabel(label_y)      # label[1])
    plt.legend(loc='best', frameon=True,
               labelspacing=.07, prop={'size': 9})
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname=figname)
    plt.close()
    return


def FairGBM_tradeoff_v3(Xs, Ys, annot, label=('X', 'Y'),
                        num_gap=1000, cmap_name='colorblind',
                        alpha_loc='b4|af', alpha_rev=False,
                        alpha_clarity=.15,
                        figname='FairGBM', figsize='M-WS'):
    # Xs.shape  (#model, #experiment /iteration)
    # Ys.shape  (#model, #experiment /iteration)
    Z = np.linspace(0, 1, num_gap)  # (#gap,)
    baseline_Ys = []
    for alpha in Z:
        if (alpha_loc == 'b4') and (not alpha_rev):
            tmp = Xs * alpha + (1. - alpha) * Ys
        elif (alpha_loc == 'af') and (not alpha_rev):
            # tmp = Xs * (1. - alpha) + alpha * Ys[i]
            tmp = Xs * (1. - alpha) + alpha * Ys
        elif (alpha_loc == 'b4') and alpha_rev:
            tmp = (1. - Xs) * alpha + (1. - alpha) * Ys
        elif (alpha_loc == 'af') and alpha_rev:
            tmp = (1. - Xs) * (1. - alpha) + alpha * Ys
        # tmp.shape  (#model, #experiment /iteration)
        baseline_Ys.append(tmp)
    baseline_Ys = np.array(baseline_Ys).transpose(1, 0, 2)
    # baseline_Ys.shape  (#model, #gap, #iteration)
    # baseline_Ys = np.array(baseline_Ys)  # (#model, #gap,#..)
    n = baseline_Ys.shape[2]
    ZZ = np.repeat(Z, n).reshape(-1, n).T.reshape(-1)
    base_Ys = baseline_Ys.reshape(-1, num_gap * n)
    # ZZ = np.array([Z for _ in range(n)]).reshape(-1)
    del n

    kwargs = {'figsize': figsize, 'figname': figname}
    kwargs['alpha_loc'] = alpha_loc
    kwargs['alpha_clarity'] = alpha_clarity
    curr_palette = sns.color_palette(cmap_name, len(annot))
    sns.palplot(curr_palette)
    _curr_sty = ['-.', '-.', '-.', '--', '-', '-', ':', '-', ':']
    if alpha_loc == 'b4':
        label_y = r'$\alpha·${} $+($1$-\alpha)·$ {}'.format(*label)
    elif alpha_loc == 'af':
        label_y = r'$($1$-\alpha)·${} $+\alpha·$ {}'.format(*label)
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot(111)
    kws = {'color': 'navy', 'lw': 1}
    for i, ant in enumerate(annot):
        kws['color'] = curr_palette[i]
        kws['label'] = ant  # annot[i]
        kws['linestyle'] = _curr_sty[i]
        _sns_line_err_bars(ax, kws, ZZ, base_Ys[i])

    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(label_y)
    plt.legend(loc='best', frameon=True,
               labelspacing=.07, prop={'size': 9})
    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname=figname)
    plt.close()
    return


def FairGBM_tradeoff_v2(Xs, Ys, annot, label=('X', 'Y'),
                        num_gap=1000, cmap_name='colorblind',
                        alpha_loc='b4|af', alpha_rev=False,
                        alpha_clarity=.15,
                        figname='FairGBM', figsize='M-WS'):
    # Xs.shape  (#model, #experiment /iteration)
    # Ys.shape  (#model, #experiment /iteration)
    Z = np.linspace(0, 1, num_gap)  # (#gap,)
    baseline_Ys = []
    for alpha in Z:
        if (alpha_loc == 'b4') and (not alpha_rev):
            tmp = Xs * alpha + (1. - alpha) * Ys
        elif (alpha_loc == 'af') and (not alpha_rev):
            # tmp = Xs * (1. - alpha) + alpha * Ys[i]
            tmp = Xs * (1. - alpha) + alpha * Ys
        elif (alpha_loc == 'b4') and alpha_rev:
            tmp = (1. - Xs) * alpha + (1. - alpha) * Ys
        elif (alpha_loc == 'af') and alpha_rev:
            tmp = (1. - Xs) * (1. - alpha) + alpha * Ys
        # tmp.shape  (#model, #experiment /iteration)
        baseline_Ys.append(tmp)
    baseline_Ys = np.array(baseline_Ys).transpose(1, 0, 2)
    # baseline_Ys.shape  (#model, #gap, #iteration)
    # baseline_Ys = np.array(baseline_Ys)  # (#model, #gap,#..)

    kwargs = {'figsize': figsize, 'figname': figname}
    kwargs['alpha_loc'] = alpha_loc
    kwargs['alpha_clarity'] = alpha_clarity
    _uncertainty_plotting(Z, baseline_Ys, annot, label[1], **kwargs)

    # _setup_figshow(fig, figname=figname)
    # plt.close()
    return


# -------------------------------

# -------------------------------

# -------------------------------
