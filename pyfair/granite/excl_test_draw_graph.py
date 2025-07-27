# coding: utf-8
# test_fairness.py


import numpy as np
import pandas as pd

# from prgm.nucleus.utils_graph import *
# from fairml.facilc.draw_graph import (
from pyfair.granite.draw_graph import (
    scatter_and_corr, sns_scatter_corr, Friedman_chart,
    stat_chart_stack, stat_chart_group, visual_confusion_mat,
    bar_chart_with_error, multiple_hist_chart,
    twinx_hist_chart, twinx_bars_chart, line_chart,
    baseline_subchart, histogram_chart, sns_corr_chart)
# from pyfair.senior.draw_graph import _setup_config, _setup_figshow


sz = 21
X = np.random.randint(100, size=sz).astype('float')
Y = X + np.random.rand(sz) * 100
df = pd.DataFrame({'X': X, 'Y': Y})

index_bar = np.array([[1, 2, 3],
                      [1, 2.5, 2.5],
                      [1, 2, 3],
                      [1, 2, 3]])

N, k = 4, 3
avg_accuracy = np.random.rand(N, k) * .3 + .7
pick_name_pru = ('Alg #A', 'Alg #B', 'Alg #C')


def test_scatter_corr():
    sns_scatter_corr(X, Y, 'chart_a1', 'M-WS')
    sns_scatter_corr(X, Y, 'chart_a2', 'S-NT')

    scatter_and_corr(X, Y, 'chart_a3', 'M-WS')
    scatter_and_corr(X, Y, 'chart_a4', 'S-NT')

    sens = np.random.randint(2, size=sz).tolist()
    sens = ['Female' if i else 'Male' for i in sens]
    sens = np.array(sens)
    sns_corr_chart(X, Y, sens, 'chart_a5', 'S-NT')


def test_friedman_chart():
    avg_order = np.mean(index_bar, axis=0)
    assert np.all(np.equal(avg_order,
                           [1, 2.125, 2.875]))

    # friedman_chart(
    Friedman_chart(
        index_bar, pick_name_pru, 'chart_b1', alpha=.05,
        anotCD=True)
    Friedman_chart(
        index_bar, pick_name_pru, 'chart_b2', alpha=.1,
        anotCD=True)

    stat_chart_stack(
        index_bar, pick_name_pru, 'chart_b3')  # .jpg
    stat_chart_group(avg_accuracy * 100,
                     [1, 2, 3, 4],
                     pick_name_pru, 'chart_b4')


def test_visualise():
    y_true = [2, 0, 2, 2, 0, 1]
    y_pred = [0, 0, 2, 2, 0, 2]

    visual_confusion_mat(y_true, y_pred,
                         ['high', 'medium', 'low'],
                         'chart_b5')  # .jpg
    visual_confusion_mat(y_true, y_pred,
                         ['high', 'medium', 'low'],
                         'chart_b6', normalize=False)

    visual_confusion_mat(y_true, y_pred,
                         ['high', 'medium', 'low'],
                         'chart_b7', title='',
                         normalize=False)


def test_backslash():
    nb_set, nb_cmp = 11, 7
    pick_name_pru = [
        "Alg #" + str(i + 1) for i in range(nb_cmp)]
    pickup_pru = list(range(nb_cmp))
    acc, tim = .3, 4.1
    nb_cls = 11  # nb_cls, nb_pru = 11, 5

    greedy = np.random.rand(nb_set, nb_cmp) * acc + (1 - acc)
    ddismi = np.random.rand(nb_set, nb_cmp) * acc + (1 - acc)
    tc_grd = np.random.rand(nb_set, nb_cmp) + tim
    tc_dsm = np.random.rand(nb_set, nb_cmp) + tim - 1
    sp_grd = np.random.randint(nb_cls - 1, size=(nb_set, nb_cmp)) + 1
    sp_dsm = np.random.randint(nb_cls - 2, size=(nb_set, nb_cmp)) + 2

    bar_chart_with_error(greedy, ddismi,
                         tc_grd, tc_dsm,
                         sp_grd, sp_dsm,
                         pickup_pru, name_pru_set=pick_name_pru,
                         figname='chart_c1')

    def _helper(pickup_uat):
        avg = np.zeros((2, nb_cmp))
        std = np.zeros((2, nb_cmp))

        if pickup_uat == "ua":
            grd, dsm = greedy, ddismi
        elif pickup_uat == "ut":
            grd, dsm = tc_grd, tc_dsm
        elif pickup_uat == "us":
            grd, dsm = sp_grd, sp_dsm

        avg[0, :] = np.mean(grd, axis=0)
        avg[1, :] = np.mean(dsm, axis=0)
        std[0, :] = np.std(grd, axis=0, ddof=1)
        std[1, :] = np.std(dsm, axis=0, ddof=1)

        baseline_subchart(avg, std, pickup_uat,
                          pick_name_pru, 'chart_c3')
    _helper("ua")
    _helper("ut")
    _helper("us")


def test_linechart():
    nb_thin, nb_lam2 = 5, 7  # 11
    data = np.random.rand(nb_thin, nb_lam2) * .3 + .7

    pickup_thin = [0, 1, 2, 3, 4]
    pickup_lam2 = list(range(nb_lam2))
    set_k2 = np.linspace(.05, .45, nb_thin) * 100
    set_lam2 = np.linspace(0, 1, nb_lam2)

    line_chart(data, pickup_thin, set_k2, pickup_lam2, set_lam2,
               'chart_c2')


def test_histchart():
    # from prgm.nucleus.utils_graph import _hist_calc_XY
    from pyfair.granite.draw_graph import _hist_calc_XY
    nb = 50  # nb, st = 50, 100
    figname = 'chart_c4'

    # X = np.random.rand(nb, 1) * 70
    # Y = np.random.rand(nb, 4) * 5 + X  # or X.T
    X = np.random.rand(nb) * 170
    Y = np.random.rand(nb, 4) * 35 + np.reshape(-1, 1)
    X_avg, Y_avg, Y_std, ind = _hist_calc_XY(X, Y)
    # pdb.set_trace()
    # assert X.shape[0] == Y_avg.shape[0] == Y_std.shape[0]
    # assert Y_avg.shape[1] == Y_std.shape[1] == len(ind)
    assert Y_avg.shape[0] == Y_std.shape[0] == len(ind)
    assert isinstance(X_avg, float) and X.shape[0] == nb

    # histogram_hor(X, Y, figname)
    # histogram_vrt(X, Y, figname)
    # histogram_chart(X, Y, figname, ind_hv='h')
    # histogram_chart(X, Y, figname, ind_hv='v')

    annots = ['idx = {}'.format(i) for i in range(4)]
    histogram_chart(X, Y, figname, annotX='X',
                    annotY=annots, ind_hv='h')
    histogram_chart(X, Y, figname, annotX='X',
                    annotY=annots, ind_hv='v')

    # X_std = [X_avg, X_avg / 2]
    Z = np.random.rand(nb, 2) * 170
    Z = np.c_[X, X / 2]
    _, _, Y_std, ind = _hist_calc_XY(X, Y)  # X_avg,Y_avg,
    histogram_chart(Z, Y, figname, annotX='X',
                    annotY=annots, ind_hv='h')
    histogram_chart(Z, Y, figname, annotX='X',
                    annotY=annots, ind_hv='v')


def test_multi_hists():
    # from prgm.nucleus.utils_graph import multiple_hist_chart
    nb_pru, nb_fair = 10, 4

    Ys_avg = np.random.rand(nb_pru, nb_fair) * 50
    Ys_std = np.random.rand(nb_pru, nb_fair)
    picked_keys = ['Alg #' + str(i) for i in range(nb_pru)]
    annots = ['GF ' + str(j) for j in range(1, nb_fair + 1)]

    # multiple_hist_chart(Ys_avg, Ys_std, picked_keys, annots,
    #                     figname='chart_c5')
    multiple_hist_chart(Ys_avg, Ys_std, picked_keys,
                        '', annots, figname='chart_c5')

    nb_pru = 3  # nb_pru, nb_comp = 3, 3
    Ys_avg = np.random.rand(nb_pru, nb_fair) * 50
    Ys_std = np.random.rand(nb_pru, nb_fair)
    Yt_avg = np.random.rand(nb_pru, nb_fair - 1) * 40
    Yt_std = np.random.rand(nb_pru, nb_fair - 1)
    picked_keys = ['Alg #' + str(i) for i in range(nb_pru)]
    annots = ['GF ' + str(j + 1) for j in range(nb_fair * 2 - 1)]
    twinx_hist_chart(Ys_avg, Ys_std, Yt_avg, Yt_std,
                     # picked_keys, '', annots, 'chart_c6')
                     picked_keys, annots, figname='chart_c6')
    twinx_bars_chart(Ys_avg.T, Ys_std.T, Yt_avg.T, Yt_std.T,
                     picked_keys, annots, figname='chart_c7')
