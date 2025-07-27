# coding: utf-8
# Author: Yijun


import itertools
import numpy as np
# from scipy import stats
# from sklearn.metrics import confusion_matrix
import matplotlib as mpl
import matplotlib.pyplot as plt


# from fairml.facils.data_entropy import prob, jointProb
# from fairml.facils.data_distance import (
from pyfair.marble.data_distance import (
    Wasserstein_dis,
    KL_divergence, Bhattacharyya_dist,  # JS_divergence,
    Hellinger_dist_v2, JS_div,      # Hellinger_dist_v1,
    _discrete_joint_cnts)           # f_divergence,

# from fairml.facilc.draw_hypos import (  # .draw.utils_hypos
#     Friedman_test, Nememyi_posthoc_test)
# from pyfair.senior.draw_graph import (  # .draw.utils_graph
from pyfair.facil.draw_prelim import (
    PLT_FRAMEBOX, PLT_LOCATION, DTY_PLT,  # PLT_AX_STYLE,
    _setup_config, _barh_kwargs, _barh_fcolor,
    _setup_figsize, _setup_figshow,  # _sns_line_err_bars,
    _setup_locater, _style_set_fig, _style_set_axis)
from pyfair.granite.draw_graph import _sns_line_err_bars


mpl.use('Agg')  # set the 'backend'
# plt.switch_backend('agg')
# mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# mpl.rc('text', usetex=True)


# =====================================
# Preliminaries


_multi_figsize = {1: (3.05, 2.52),
                  2: (5.61, 2.52),
                  3: (8.35, 2.57),
                  4: (11.35, 2.57)}  # _setup


# ----------------------------------
# 散点图和拟合曲线，相关系数


# colour matching / combination
_nat_sci_cs = [
    [223, 158, 155],  # DF9E9B
    [153, 186, 223],  # 99BADF
    [216, 231, 202],  # D8E7CA
    [153, 205, 206],  # 99CDCE
    [153, 154, 205],  # 999ACD
    [255, 208, 233],  # FFD0E9
]  # RGB, HEX  # (i/255 for i in colour)
_nat_sci_cs = [
    '#DF9E9B', '#99BADF', '#D8E7CA',
    '#99CDCE', '#999ACD', '#FFD0E9',
]


_nat_sci_cs = [
    # '#1d3557', '#457b9d', '#a8dadb', '#e73847', '#fadaef',
    # '#457b9d', '#a8dadb', '#fadaef', '#e73847', '#1d3557',
    '#e73847', '#a8dadb', '#457b9d', '#1d3557',  # '#fadaef',
    # '#1A0841', '#4F9DA6', '#FFAD5A', '#FF5959',
    # [26, 8, 65], [79, 157, 166], [255, 173, 90], [255, 89, 89],
    # '#e73847', '#fadaef', '#a8dadb', '#457b9d', '#1d3557',
    # '#264653', '#2a9d8e', '#e9c46b', '#f3a261', '#e66f51',
    # '#F0E6E4', '#CEB5B9', '#AE8FCE', '#98A1B1', '#b36asf',
    # '#F8CDCF', '#FAC5A4', '#FBEAA6', '#C2E99E', '#ABEAF9',
]


def _sub_spread_COR(Xs, Ys, annotZs):
    # Correlation
    Rs = [np.corrcoef(X, Y)[1, 0] for X, Y in zip(Xs, Ys)]
    keys = ["Correlation = {:.4f} ({:s})".format(
        r, Z) for r, Z in zip(Rs, annotZs)]
    # Regression
    regr = [np.polyfit(X, Y, deg=1) for X, Y in zip(Xs, Ys)]
    estimated = [np.polyval(r, X) for r, X in zip(regr, Xs)]
    # num_z = len(annotZs)  # equal to len(Xs)==len(Ys)
    return keys, estimated  # , num_z


# def multi_scatter_hor(X, Ys, annots, annotZs, figname,
#                       figsize, identity, base, locate):
def multi_scatter_hor(X, Ys, annots, annotZs, identity):
    '''
    # Correlation
    Rs = [np.corrcoef(X, Y)[1, 0] for Y in Ys]
    # key = ["{:3s} = {:.4f}".format(Z, r) for r, Z in zip()]
    key = [
        "Correlation = {:.4f} ({:s})".format(
            r, Z) for r, Z in zip(Rs, annotZs)]
    # Regression
    regr = [np.polyfit(X, Y, deg=1) for Y in Ys]
    estimated = [np.polyval(r, X) for r in regr]
    '''
    num_z = len(annotZs)  # equal to len(Xs)==len(Ys)
    key, estimated = _sub_spread_COR([X] * num_z, Ys, annotZs)

    fig = plt.figure(figsize=_setup_config['L-NT'])
    if identity:
        for i in range(num_z):
            plt.scatter(X, Ys[i], c=_nat_sci_cs[i], s=15)
            plt.plot(X, estimated[i], '-', lw=1,
                     label=key[i], color=_nat_sci_cs[i])
            # ZX = sorted(Xs[i])
        ZX = sorted(X)
        plt.plot(ZX, ZX, "k--", lw=1, label=r"$f(x)=x$")
    else:
        for i in range(num_z):
            plt.scatter(X, Ys[i], c=_nat_sci_cs[i], label=key[i],
                        s=10)
            plt.plot(X, estimated[i], c=_nat_sci_cs[i], lw=1.5)
    axs = plt.gca()
    for i in range(num_z):
        _sns_line_err_bars(axs, {  # "alpha": .2
            "color": _nat_sci_cs[i], "alpha": .3,
            "lw": 1}, X, Ys[i])
    # '''
    # if base is not None:
    #     _setup_locater(fig, base)
    # '''
    plt.xlabel(annots[0])
    plt.ylabel(annots[1])
    # '''
    # plt.legend(loc=locate, frameon=PLT_FRAMEBOX,
    #            labelspacing=.07, prop={'size': 9})
    # if figsize is not None:
    #     fig = _setup_figsize(fig, figsize)
    # _setup_figshow(fig, figname)
    # plt.close(fig)
    # return
    # '''
    return fig


# def multi_scatter_vrt(X, Ys, annots, annotZs, figname,
#                       figsize, identity, base, locate):
def multi_scatter_vrt(X, Ys, annots, annotZs, identity):
    '''
    # Correlation
    Rs = [np.corrcoef(Y, X)[1, 0] for Y in Ys]
    key = ["Correlation = {:.4f} ({:s})".format(
        r, Z) for r, Z in zip(Rs, annotZs)]
    # Regression
    regr = [np.polyfit(Y, X, deg=1) for Y in Ys]
    estimated = [np.polyval(r, Y) for r, Y in zip(regr, Ys)]
    '''
    num_z = len(annotZs)
    key, estimated = _sub_spread_COR(Ys, [X] * num_z, annotZs)

    fig = plt.figure(figsize=_setup_config['L-NT'])
    if identity:
        for i in range(num_z):
            plt.scatter(Ys[i], X, c=_nat_sci_cs[i], s=10)  # s=15)
            plt.plot(Ys[i], estimated[i], '-', lw=1,
                     label=key[i], color=_nat_sci_cs[i])
        ZX = sorted(X)
        plt.plot(ZX, ZX, "k--", lw=1, label=r"$f(x)=x$")
    else:
        for i in range(num_z):
            plt.scatter(Ys[i], X, c=_nat_sci_cs[i], label=key[i],
                        s=10)
            plt.plot(Ys[i], estimated[i], c=_nat_sci_cs[i], lw=1)
    axs = plt.gca()
    for i in range(num_z):
        _sns_line_err_bars(axs, {
            "color": _nat_sci_cs[i], "alpha": .3,
            "lw": 1}, Ys[i], X)
    # '''
    # if base is not None:
    #     _setup_locater(fig, base)
    # '''
    # axs.set_aspect(1)
    plt.xlabel(annots[1])
    plt.ylabel(annots[0])
    # '''
    # plt.legend(loc=locate, frameon=PLT_FRAMEBOX,
    #            labelspacing=.07, prop={'size': 9})
    # if figsize is not None:
    #     fig = _setup_figsize(fig, figsize)
    # _setup_figshow(fig, figname)
    # plt.close(fig)
    # return
    # '''
    return fig


def multiple_scatter_chart(X, Ys, annots=('X', 'Ys'),
                           annotZs=('Z1', 'Z2', 'Zs'),
                           figname="", figsize='M-WS',
                           ind_hv='h', identity=True,
                           base=None,  # diff=0.05,
                           locate=PLT_LOCATION):
    '''
    if ind_hv == 'h':
        multi_scatter_hor(X, Ys, annots, annotZs,
                          figname + "_hor", figsize,
                          identity, base, locate)
    elif ind_hv == 'v':
        multi_scatter_vrt(X, Ys, annots, annotZs,
                          figname + "_vrt", figsize,
                          identity, base, locate)
    '''

    if ind_hv == 'h':
        fig = multi_scatter_hor(X, Ys, annots, annotZs,
                                identity)
    elif ind_hv == 'v':
        fig = multi_scatter_vrt(X, Ys, annots, annotZs,
                                identity)
    figname += ("_hor" if ind_hv == 'h' else "_vrt")
    if base is not None:
        _setup_locater(fig, base)
    plt.legend(loc=locate, frameon=PLT_FRAMEBOX,
               labelspacing=.07, prop={'size': 9})
    if figsize is not None:
        fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def _alt_confus_cm(num_z, Mat_k):  # ,normalize):
    cm = np.zeros((num_z, num_z))
    for i in range(num_z):
        for j in range(num_z):
            cm[i, j] = np.corrcoef(Mat_k[i], Mat_k[j])[1, 0]

    # if normalize:
    #     cm = cm / cm.sum(axis=1)[:, np.newaxis]
    # # thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    return cm


def analogous_confusion(Mat, label_vals, figname,
                        figsize='L-NT', cmap=None,
                        normalize=True, rotate=25):
    # similar to/as? `visual_confusion_mat`
    if cmap is None:
        # cmap = plt.get_cmap("Blues")  # "Blues_r")
        cmap = plt.colormaps.get_cmap("Blues")
    num_z = len(Mat)  # i.e. len(label_vals)
    # cm = np.zeros((num_z, num_z))
    # for i in range(num_z):
    #     for j in range(num_z):
    #         cm[i, j] = np.corrcoef(Mat[i], Mat[j])[1, 0]
    cm = _alt_confus_cm(num_z, Mat)

    fig = plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(num_z)
    plt.xticks(tick_marks, label_vals, rotation=rotate)
    plt.yticks(tick_marks, label_vals)
    if normalize:
        cm = cm / cm.sum(axis=1)[:, np.newaxis]
    # thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(num_z),
                                  range(num_z)):
        plt.text(j, i, "{:.3f}".format(cm[i, j]),
                 horizontalalignment="center", color="k",
                 size="small")
    _style_set_fig(fig, siz=_setup_config[figsize])
    _setup_figshow(fig, figname)
    return


def _ext_confus_cm(cm, cmap, figsize):
    fig, ax = plt.subplots(figsize=(8, 6))
    if figsize == 'extra':
        del fig, ax
        fig, ax = plt.subplots(figsize=_setup_config[figsize])
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    num_za, num_zb = cm.shape
    num_shrink = float(num_za) / float(num_zb) + .01
    plt.colorbar(ax=ax, shrink=num_shrink)  # .8)

    for i, j in itertools.product(
            range(num_za), range(num_zb)):
        plt.text(j, i, "{:.3f}".format(
            cm[i, j]), horizontalalignment="center",
            color="k", size="small")
    return fig, ax


def _alt_confus_cm_asym(Mat_A, Mat_B):
    num_za = int(Mat_A.shape[0])
    num_zb = int(Mat_B.shape[0])
    cm = np.zeros((num_za, num_zb))
    for i in range(num_za):
        for j in range(num_zb):
            cm[i, j] = np.corrcoef(Mat_B[j], Mat_A[i])[1, 0]
    return cm


def analogous_confusion_extended(Mat_A, Mat_B, key_A, key_B,
                                 figname, cm=None, figsize='L-NT',
                                 cmap_name="Blues_r", rotate=5):
    num_za, num_zb = len(key_A), len(key_B)
    if cm is None:
        cm = _alt_confus_cm_asym(Mat_A, Mat_B)

    # cmap = plt.get_cmap(cmap_name)
    cmap = plt.colormaps.get_cmap(cmap_name)
    fig, _ = _ext_confus_cm(cm, cmap, figsize)

    # _style_set_axis(ax, invt=True)
    tick_mk_a = np.arange(num_za)
    tick_mk_b = np.arange(num_zb)
    plt.xticks(tick_mk_b, key_B, rotation=rotate)
    plt.yticks(tick_mk_a, key_A)
    _style_set_fig(fig, siz=_setup_config[figsize])
    _setup_figshow(fig, figname)
    return


# '''
# def analogous_confusion_extended(Mat_A, Mat_B, key_A, key_B,
#                                  figname, cm=None, figsize='L-NT',
#                                  cmap_name="Blues_r", rotate=5):
#     cmap = plt.get_cmap(cmap_name)
#     num_za, num_zb = len(key_A), len(key_B)
#     if cm is None:
#         cm = np.zeros((num_za, num_zb))
#         for i in range(num_za):
#             for j in range(num_zb):
#                 cm[i, j] = np.corrcoef(Mat_B[j], Mat_A[i])[1, 0]
#     fig, ax = plt.subplots(figsize=(8, 6))
#     if figsize == 'extra':
#         del fig, ax
#         fig, ax = plt.subplots(figsize=_setup_config[figsize])
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     num_shrink = float(num_za) / float(num_zb) + .01
#     plt.colorbar(ax=ax, shrink=num_shrink)  # .8)
#
#     # _style_set_axis(ax, invt=True)
#     tick_mk_a = np.arange(num_za)
#     tick_mk_b = np.arange(num_zb)
#     plt.xticks(tick_mk_b, key_B, rotation=rotate)
#     plt.yticks(tick_mk_a, key_A)
#     for i, j in itertools.product(range(num_za), range(num_zb)):
#         plt.text(j, i, "{:.3f}".format(
#             cm[i, j]), horizontalalignment="center",
#             color="k", size="small")
#     _style_set_fig(fig, siz=_setup_config[figsize])
#     _setup_figshow(fig, figname)
#     return
# '''


# def _alter_sub_Pearson_cor(x, ys, annotZs):
#     # Correlation & Regression
#     Rs = [np.corrcoef(x, y)[1, 0] for y in ys]
#     key = ["Correlation = {:.4f} ({:s})".format(
#         r, Z) for r, Z in zip(Rs, annotZs)]
#     regr = [np.polyfit(x, y, deg=1) for y in ys]
#     estimated = [np.polyval(r, x) for r in regr]
#     return key, estimated


def _alternative_multi_scatter_hor(X, Ys, sens,
                                   annots, annotZs,
                                   figname, identity,
                                   locate, box, invt,
                                   # share=False,
                                   # kwargs=dict()):
                                   kwargs):  # dict()
    sa_len, num_z = len(sens), len(annotZs)
    fig, ax = plt.subplots(1, sa_len,  # sharey=share,
                           figsize=_multi_figsize[sa_len])
    for k, (x, ys, sa) in enumerate(zip(X, Ys, sens)):

        # '''
        # # Correlation & Regression
        # Rs = [np.corrcoef(x, y)[1, 0] for y in ys]
        # key = ["Correlation = {:.4f} ({:s})".format(
        #     r, Z) for r, Z in zip(Rs, annotZs)]
        # regr = [np.polyfit(x, y, deg=1) for y in ys]
        # estimated = [np.polyval(r, x) for r in regr]
        # '''
        # key, estimated = _alter_sub_Pearson_cor(x, ys, annotZs)
        key, estimated = _sub_spread_COR([x] * num_z, ys, annotZs)

        for i in range(num_z):
            ax[k].scatter(
                x, ys[i], c=_nat_sci_cs[i], label=key[i], s=10)
            ax[k].plot(x, estimated[i], c=_nat_sci_cs[i], lw=1)
            _sns_line_err_bars(ax[k], {
                "color": _nat_sci_cs[i], "alpha": .43, "lw": 1
            }, x, ys[i])

        if identity:
            zx = sorted(x)
            ax[k].plot(zx, zx, "k--", lw=1, label=r"$f(x)=x$")
        # ax[k].set_xlabel(annots[0])
        ax[k].set_xlabel(annots[0] + sa.upper())
        ax[k].set_ylabel(annots[1])
        ax[k].legend(loc=locate, frameon=box, framealpha=.67,
                     labelspacing=.07, prop={'size': 9}, **kwargs)
        # if PLT_AX_STYLE:
        #     _style_set_axis(ax[k], invt)
        _style_set_axis(ax[k], invt)
    fig.tight_layout()
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def _alternative_multi_scatter_vrt(X, Ys, sens,
                                   annots, annotZs,
                                   figname, identity,
                                   locate, box, invt,
                                   # share=False,
                                   # kwargs=dict()):
                                   kwargs):  # dict()
    sa_len, num_z = len(sens), len(annotZs)
    fig, ax = plt.subplots(1, sa_len,  # sharex=share,
                           figsize=_multi_figsize[sa_len])
    for k, (x, ys, sa) in enumerate(zip(X, Ys, sens)):

        # '''
        # # Correlation & Regression
        # Rs = [np.corrcoef(y, x)[1, 0] for y in ys]
        # key = ["Correlation = {:.4f} ({:s})".format(
        #     r, Z) for r, Z in zip(Rs, annotZs)]
        # regr = [np.polyfit(y, x, deg=1) for y in ys]
        # estimated = [np.polyval(r, y) for r, y in zip(regr, ys)]
        # '''
        key, estimated = _sub_spread_COR(ys, [x] * num_z, annotZs)

        for i in range(num_z):
            ax[k].scatter(
                ys[i], x, c=_nat_sci_cs[i], label=key[i], s=10)
            ax[k].plot(ys[i], estimated[i], c=_nat_sci_cs[i], lw=1)
            _sns_line_err_bars(ax[k], {
                "color": _nat_sci_cs[i], "alpha": .43, "lw": 1
            }, ys[i], x)

        if identity:
            zx = sorted(x)
            ax[k].plot(zx, zx, "k--", lw=1, label=r"$f(x)=x$")
        ax[k].set_xlabel(annots[1])
        ax[k].set_ylabel(annots[0] + sa.upper())
        ax[k].legend(loc=locate, frameon=box, framealpha=.67,
                     labelspacing=.07, prop={'size': 9},
                     # handletextpad=.04,
                     # borderpad=.27)  # columnspacing=.07)
                     **kwargs)
        # if PLT_AX_STYLE:
        #     _style_set_axis(ax[k], invt)
        _style_set_axis(ax[k], invt)
    fig.tight_layout()
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def multiple_scatter_alternative(X, Ys, sens,
                                 annots, annotZs,
                                 figname, ind_hv='v',
                                 identity=False,
                                 invt=False,
                                 # share=False,
                                 box=PLT_FRAMEBOX,
                                 locate=PLT_LOCATION):
    # annotX = " ".join([annots[0], sa.upper()])
    annotXs = (annots[0] + " ", annots[1])
    kwargs = {"handletextpad": .04, "borderpad": .27}
    if ind_hv == 'h':
        _alternative_multi_scatter_hor(
            X, Ys, sens, annotXs, annotZs, figname + "_hor",
            identity, locate, box, invt, kwargs)
    elif ind_hv == 'v':
        _alternative_multi_scatter_vrt(
            X, Ys, sens, annotXs, annotZs, figname + "_vrt",
            identity, locate, box, invt, kwargs)
    return


def analogous_confusion_alternative(Mat, sens,
                                    label_vals, figname,
                                    cmap=None, rotate=35,
                                    normalize=False):
    _figsize = {1: (3.05, 2.52),
                2: (9.25, 3.95),
                3: (15.25, 4.18),
                4: (18.47, 3.76)}
    sa_len, num_z = len(sens), len(Mat[0])
    tick_marks = np.arange(num_z)
    if cmap is None:
        # cmap = plt.get_cmap("Blues")
        cmap = plt.colormaps.get_cmap("Blues")
    fig, ax = plt.subplots(1, sa_len,
                           figsize=_figsize[sa_len])
    # ax = ax.flatten()
    # for k, (mt, sa) in enumerate(zip(Mat, sens)):
    for k, (_, sa) in enumerate(zip(Mat, sens)):  # mt,
        # cm = np.zeros((num_z, num_z))
        # for i in range(num_z):
        #     for j in range(num_z):
        #         cm[i, j] = np.corrcoef(Mat[k][i], Mat[k][j])[1, 0]
        cm = _alt_confus_cm(num_z, Mat[k])  # ,normalize)

        img = ax[k].imshow(cm, interpolation='nearest', cmap=cmap)
        ax[k].set_xticks(tick_marks)
        ax[k].set_yticks(tick_marks)
        ax[k].set_xticklabels(label_vals)
        ax[k].set_yticklabels(label_vals)
        plt.setp(ax[k].get_xticklabels(), rotation=rotate,
                 ha="right", rotation_mode="anchor")
        # Rotate the tick labels and set their alignment.
        # ax[k].set_xlabel(sa.upper())  # Sensitive attributes

        ax[k].set_xlabel("Attribute " + sa.upper() if (
            sa.lower() != "joint") else sa.upper() + " Attribute")
        # '''
        # if sa.lower() != "joint":
        #     ax[k].set_xlabel("Attribute " + sa.upper())
        # else:
        #     ax[k].set_xlabel(sa.upper() + " Attribute")  # sa+
        # '''
        if normalize:
            cm = cm / cm.sum(axis=1)[:, np.newaxis]
        # thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(num_z),
                                      range(num_z)):
            ax[k].text(j, i, "{:.3f}".format(cm[i, j]),
                       horizontalalignment='center',
                       color='k', size='small')
    fig.colorbar(img, ax=[ax[i] for i in range(sa_len)],
                 fraction=0.03, pad=0.05)
    # fig.tight_layout()
    fig.savefig("{}{}".format(figname, DTY_PLT),
                dpi=300, bbox_inches='tight')
    # plt.colorbar()
    # _setup_figshow(fig, figname)
    return


# ----------------------------------
# subchart


def single_hist_chart(Y_avg, Y_std, picked_keys, annotX='',
                      annotY='Time Cost', figname='',
                      figsize='S-NT', rotate=80):  # L-NT,L-WS
    picked_len = len(picked_keys)  # pick_len_pru/pick_leng_fair
    ind = np.arange(picked_len) * 2  # 3
    wid = (2 - .4) / 1  # picked_len
    fig, ax = plt.subplots(figsize=_setup_config['L-NT'])
    kwargs = _barh_kwargs.copy()
    kwargs['error_kw']['elinewidth'] = .6
    ax.bar(ind, Y_avg, width=wid, yerr=Y_std,
           color=_barh_fcolor[-4], **kwargs)  # 4,5
    ax.set_ylabel(annotY)
    ax.set_xlabel(annotX)
    # ax.set_xticks(ind)
    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ind))
    ax.set_xticklabels(picked_keys, rotation=rotate, fontsize=8)
    ax.autoscale_view()
    fig = _setup_figsize(fig, figsize)
    fig.canvas.draw()
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# =====================================
# SG


# -------------------------------------
# 带误差的折线图


def _line_std_drawer(data, ddof=1):
    avg = np.mean(data, axis=1)
    std = np.std(data, axis=1, ddof=ddof)  # default:0
    r1 = list(map(lambda x: x[0] - x[1], zip(avg, std)))  # 上方差
    r2 = list(map(lambda x: x[0] + x[1], zip(avg, std)))  # 下方差
    return avg, r1, r2


def _line_std_colors(annot, index, ls="-", alpha=.3,
                     visible=True, palette=None):
    setting_ls = {"label": annot, "linestyle": ls}
    setting_sd = {"alpha": alpha, "visible": visible}
    if palette is not None:
        color = palette(index)
        setting_ls["color"] = color
        setting_sd["color"] = color
    return setting_ls, setting_sd


def lines_with_std_3d(X, Ys, figname, figsize='L-WS',
                      annotX='X', annotY=('Y',),
                      annotZ="Metric", visible=True,
                      baseline=None, annotBL="Ensem", ddof=1):
    # aka. def line_chart_with_std()
    # NB. X.shape = (?,)
    #     Ys.shape = (nb_alg, ?, nb_iter=5)
    #     baseline.shape = (nb_iter=5,)
    # palette = plt.get_cmap("Set1")
    palette = plt.colormaps.get_cmap("Set1")
    font = {"family": plt.rcParams['font.family']}  # "Times New Roman"}
    # iters = X.copy()
    fig, axs = plt.subplots(figsize=_setup_config['L-NT'])

    if baseline is not None:
        leng = len(X)
        data_bl = np.repeat(baseline, leng).reshape(-1, leng).T
        avg, r1, r2 = _line_std_drawer(data_bl, ddof=ddof)
        config_ls, config_sd = _line_std_colors(
            "Ensem", 0, "--", .1, visible, palette=palette)
        axs.plot(X, avg, **config_ls)
        axs.fill_between(X, r1, r2, **config_sd)

        ax_y = np.mean(baseline)
        ax_x = np.percentile(X, 67)
        plt.text(ax_x, ax_y, annotBL)

    print(annotX, annotZ, "|", annotY)
    for i, annot in enumerate(annotY):
        print("\t", i, annot)
        avg, r1, r2 = _line_std_drawer(Ys[i], ddof=ddof)
        config_ls, config_sd = _line_std_colors(
            annot, i + 1, visible=visible, palette=palette)
        axs.plot(X, avg, **config_ls)
        axs.fill_between(X, r1, r2, **config_sd)

    axs.set_xlim([X[0], X[-1]])  # plt.xlim
    axs.set_xlabel(annotX)
    axs.set_ylabel(annotZ)
    axs.legend(loc=PLT_LOCATION, frameon=PLT_FRAMEBOX,
               prop=font, labelspacing=.4)

    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


def lines_with_std_2d(X, Ys, figname, figsize='L-WS',
                      annotX='X', annotY=('Y',),
                      annotZ="Size of (Sub-)Ensemble",
                      visible=True):
    # NB. X.shape = (?,)
    #     Ys.shape = (nb_alg, ?)
    # palette = plt.get_cmap("Set1")
    palette = plt.colormaps.get_cmap("Set1")
    font = {"family": plt.rcParams['font.family']}  # "Times New Roman"}
    fig, axs = plt.subplots(figsize=_setup_config['L-NT'])

    for i, annot in enumerate(annotY):
        # config_ls, config_sd = _line_std_colors(
        config_ls, _ = _line_std_colors(
            annot, i + 1, visible=visible, palette=palette)
        axs.plot(X, Ys[i], **config_ls)

    axs.set_xlim([X[0], X[-1]])  # plt.xlim
    axs.set_xlabel(annotX)
    axs.set_ylabel(annotZ)
    axs.legend(loc="upper right", frameon=PLT_FRAMEBOX,
               prop=font, labelspacing=.4)

    fig = _setup_figsize(fig, figsize)
    _setup_figshow(fig, figname)
    plt.close(fig)
    return


# -------------------------------------
# 频数直方图


def _discrete_illustrate(IndexSlices, SubIndices):
    dis_1 = Wasserstein_dis(SubIndices, IndexSlices)
    dis_2 = JS_div(SubIndices, IndexSlices)
    dist = "EMD= {:.4f}\nJS = {:.4f}".format(dis_1, dis_2)

    px, py, _ = _discrete_joint_cnts(  # ,vXY
        IndexSlices, SubIndices, density=True)

    ans_1 = KL_divergence(py, px)
    # ans_2 = JS_divergence(py, px)
    # ans_3 = f_divergence(py, px)
    # ans_4 = Hellinger_dist_v1(py, px)
    ans_5 = Hellinger_dist_v2(py, px)
    ans_6 = Bhattacharyya_dist(py, px)
    dist = (
        "{}\nKL = {:.4f}\nHellinger dist= {:.4f}"
        "\nBhattacharyya = {:.4f}".format(dist, ans_1, ans_5, ans_6))
    return dist, [dis_1, dis_2, ans_1, ans_5, ans_6]


def _discrete_not_split(freq_x, freq_y, comp_y, dist, annots,
                        figname, figsize):
    comp_x = freq_x
    fc_p, fc_q = 'r', 'b'
    annotX, annotY = annots
    font = {"family": plt.rcParams['font.family']}
    # before having _discrete_not_split

    fig, axes = plt.subplots(figsize=_setup_config[figsize])
    # axes.bar(freq_x, freq_y, fc=fc_p)
    # axes.bar(comp_x, comp_y, fc=fc_q, label=dist)
    wid = .4  # .5, \pm.25
    axes.bar(
        [i - .2 for i in freq_x], freq_y, wid, fc=fc_p)
    axes.bar([i + .2 for i in comp_x], comp_y,
             wid, fc=fc_q, label=dist)

    # axes.bar(comp_x, base_y, fc=fc_q, label=dist)
    axes.set_xlabel(annotX)
    axes.set_ylabel(annotY)
    axes.legend(loc="upper right", frameon=PLT_FRAMEBOX,
                    prop=font, labelspacing=.4)
    _setup_figshow(fig, figname + "_merge")
    plt.close(fig)

    fig, axes = plt.subplots(figsize=_setup_config[figsize])
    axes.barh(freq_x, freq_y, fc=fc_p)
    tmp_y = [-i for i in comp_y]
    axes.barh(comp_x, tmp_y, fc=fc_q, label=dist)
    axes.set_xlabel(annotY)
    axes.set_ylabel(annotX)
    axes.legend(loc="upper left", frameon=PLT_FRAMEBOX,
                prop=font, labelspacing=.4)
    _setup_figshow(fig, figname + "_transverse")  # 横向
    plt.close(fig)
    return


def discrete_bar_comparison(IndexSlices, SubIndices,
                            figname="", figsize='L-WS',
                            density=False, split=True,
                            annotX="Index of Features",
                            # annotY="Size of (Sub-)Ensemble",
                            annotY="Frequency", nb_iter=5, k=-1):
    # e.g., item_pru = [True, False, ..]
    # SubIndices = IndexSlices[pru_item]
    # IndexSlices.shape (100,368/*)
    # SubIndices .shape (nb_pru, 368/*)

    TmpIndices = np.concatenate(SubIndices)
    nb_iter = len(TmpIndices)
    IdxOrigin = IndexSlices.reshape(-1)
    IdxPruned = TmpIndices.reshape(-1)

    freq_y, comp_y, freq_x = _discrete_joint_cnts(
        IdxOrigin, IdxPruned, density)
    comp_x = freq_x
    if (not density) and (nb_iter > 0):
        temporary = [_discrete_joint_cnts(
            IdxOrigin, j.reshape(-1), density,
            v=freq_x) for j in SubIndices]
        _, temporary, _ = zip(*temporary)
        comp_y = np.mean(temporary, axis=0)
    if density:
        annotY = "Probability"
    fc_p, fc_q = 'r', 'b'

    font = {"family": plt.rcParams['font.family']}  # "Times New Roman"}
    dist, dist_arr = _discrete_illustrate(IdxOrigin, IdxPruned)
    fig, axes = plt.subplots(1, 2, sharey=True,
                             # sharex=True,
                             figsize=(7.5, 4))
    axes[0].bar(freq_x, freq_y, fc=fc_p)
    axes[1].bar(comp_x, comp_y, fc=fc_q, label=dist)
    axes[0].set_ylabel(annotY)
    axes[0].set_xlabel(annotX)
    axes[1].set_xlabel(annotX + " ({})".format(k))
    # axes[1].set_xlabel(annotX + " (k={})".format(k))
    axes[1].legend(loc="upper right", frameon=PLT_FRAMEBOX,
                   prop=font, labelspacing=.4)
    _setup_figshow(fig, figname + "_split")
    plt.close(fig)

    if not split:
        _discrete_not_split(freq_x, freq_y, comp_y, dist,
                            (annotX, annotY), figname, figsize)

    return dist_arr


# -------------------------------------
#

# -------------------------------------
#
