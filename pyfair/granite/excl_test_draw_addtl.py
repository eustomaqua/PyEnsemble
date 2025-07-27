# coding: utf-8


import numpy as np
import pandas as pd
# from fairml.facilc.draw_addtl import (
from pyfair.granite.draw_addtl import (
    multiple_lines_with_errorbar, box_plot, scatter_k_cv_with_real,
    boxplot_k_cv_with_real, approximated_dist_comparison,
    multiple_scatter_comparison, scatter_with_marginal_distrib,
    line_reg_with_marginal_distr, single_line_reg_with_distr,
    lineplot_with_uncertainty, multi_lin_reg_with_distr,
    FairGBM_tradeoff_v3, FairGBM_tradeoff_v2)
lc_error_bar = multiple_lines_with_errorbar


# -----------------------
# nucleus.utils_graph.py


def excl_excl_test_linechart():
    # from prgm.nucleus.utils_graph import line_chart
    from pyfair.granite.draw_graph import line_chart
    nb_thin, nb_lam2 = 5, 7  # 11
    data = np.random.rand(nb_thin, nb_lam2) * .3 + .7

    pickup_thin = [0, 1, 2, 3, 4]
    pickup_lam2 = list(range(nb_lam2))
    set_k2 = np.linspace(.05, .45, nb_thin) * 100
    set_lam2 = np.linspace(0, 1, nb_lam2)

    line_chart(data, pickup_thin, set_k2, pickup_lam2, set_lam2,
               'chart_c2')


def excl_excl_test_multi_line():
    # from prgm.nucleus.utils_graph import multiple_line_chart
    from pyfair.granite.draw_graph import multiple_line_chart
    num, nb_iter, baseline = 10, 5, 4
    X = np.linspace(0, 1, num)  # doc issue  # no issue
    Ys = np.random.rand(num, baseline, nb_iter)
    annotY = ['BS#' + str(i) for i in range(baseline)]
    multiple_line_chart(X, Ys, annotY=annotY,
                        figname='mulin_lam')
    multiple_line_chart(X, Ys[:, :, 0], annotY=annotY,
                        figname='mulin_lam_sing')
    # Ys = [np.random.rand() for i in range(5)]


# -----------------------
# nucleus.fair_graph.py


def excl_test_multi_lins():
    '''
    from prgm.nucleus.oracle_graph import multiple_line_chart as lc_v2
    from prgm.nucleus.utils_graph import multiple_line_chart as lc_v1
    from prgm.nucleus.oracle_graph import \
        multiple_lines_with_errorbar as lc_error_bar
    '''
    # lc_error_bar = multiple_lines_with_errorbar

    num, nb_iter, baseline = 10, 5, 4
    X = np.linspace(0, 1, num)  # doc issue  # no issue
    Ys = np.random.rand(num, baseline, nb_iter) * 10
    annotY = ['BS#' + str(i) for i in range(baseline)]

    # lc_v2(X, Ys)  # , figname='lam_box')
    # lc_v1(X, Ys, annotY=annotY, figname='lam_lin')
    # pdb.set_trace()
    lc_error_bar(X, Ys, picked_keys=annotY)


def excl_excl_test_box_plot():
    # from prgm.nucleus.oracle_graph import box_plot
    nb_iter, baseline = 5, 4
    Ys = np.random.rand(baseline, nb_iter) * 100.

    annotYs = ['BS#{}'.format(i + 1) for i in range(baseline)]
    box_plot(Ys, annotYs, annotY='Test Accuracy (%)')

    annotYs = [str(i + 1) for i in range(baseline)]
    box_plot(Ys, annotYs, annotY='Acc (%)', annotX=r'$\lambda$',
             figname='box_v2', rotate=0)


# -----------------------
# nucleus.utils_graph.py


def excl_test_fairmanf_2nd():
    # from prgm.nucleus.oracle_graph import scatter_k_cv_with_real
    num, nb_iter = 7, 3

    X = np.arange(num) + 2  # 2.
    # Ys = np.random.rand(nb_iter, num).T + 59.7
    Ys = np.random.rand(nb_iter, num) + 60.
    z = np.random.rand(nb_iter) + 60

    # scatter_k_cv_with_real(X, Ys.T, z)
    scatter_k_cv_with_real(X, Ys, z, tidy_cv=False)
    scatter_k_cv_with_real(X, Ys, z, figname='hyperpm_effect_p')

    # from prgm.nucleus.oracle_graph import boxplot_k_cv_with_real
    boxplot_k_cv_with_real(X, Ys, z)
    # from prgm.nucleus.oracle_graph import box_plot
    box_plot(Ys.T, X, 'Approximated', 'pm', figname='hyperpm_lam')


def excl_test_fairmanf_1st():
    # from prgm.nucleus.oracle_graph import approximated_dist_comparison
    num, nb_iter, nb_att = 7, 3, 2

    # X = np.arange(num).tolist()  # + 4.
    X = np.arange(num) + 4.
    Ys = np.random.rand(nb_att, nb_iter, num) * 10
    picked_keys = ['att#' + str(i + 1) for i in range(nb_att)]
    approximated_dist_comparison(X, Ys, picked_keys)

    # from prgm.nucleus.oracle_graph import multiple_scatter_comparison
    z = np.random.rand(nb_att, nb_iter) * 10
    multiple_scatter_comparison(X, Ys, z, picked_keys)


# -----------------------
# Plot 2A, 2B
# https://blog.csdn.net/happy_wealthy/article/details/110646127


def excl_test_fairmanf_exp2a():
    '''
    from prgm.nucleus.oracle_graph import scatter_with_marginal_distrib
    from prgm.nucleus.oracle_graph import (
        lineplot_with_uncertainty, _uncertainty_plotting)
    from prgm.nucleus.oracle_graph import line_reg_with_marginal_distr
    from prgm.nucleus.oracle_graph import single_line_reg_with_distr
    '''
    # from fairml.facilc.draw_addtl import _uncertainty_plotting
    num, bl = 100, 4

    dat_acc = np.random.rand(num) * .1 + .7
    dat_fir = np.random.rand(num, bl) * .1 + .6
    df = pd.DataFrame(
        np.concatenate([dat_acc.reshape(-1, 1), dat_fir], axis=1),
        columns=['acc', 'GM1', 'GM2', 'GM3', 'fair'])

    # df_little, df_middle, df_big
    # df_all ['BV','BW'] lightgbm,fairgbm,bagging
    df_1 = df[['acc', 'GM1']].rename(columns={'GM1': 'fair'})
    df_2 = df[['acc', 'GM2']].rename(columns={'GM2': 'fair'})
    df_3 = df[['acc', 'GM3']].rename(columns={'GM3': 'fair'})
    df_1['learning'] = 'DP'
    df_2['learning'] = 'EO'
    df_3['learning'] = 'PQP'
    '''
    df_alternative = pd.concat([
        df_1, df_2, df_3], axis=0).reset_index(drop=True)  # 合并表格
    '''

    tag_Ys, picked_keys = ['GM1', 'GM2', 'GM3'], ['DP', 'EO', 'PQP']
    scatter_with_marginal_distrib(
        df, 'acc', 'fair', tag_Ys, picked_keys, figname='chart_f3')
    lineplot_with_uncertainty(df, 'acc', 'fair', tag_Ys, picked_keys,
                              figname='chart_f1')
    lineplot_with_uncertainty(df, 'acc', 'fair', tag_Ys, picked_keys,
                              alpha_loc='af', figname='chart_f2')

    '''
    line_reg_with_marginal_distr(
        df, 'acc', 'fair', tag_Ys, picked_keys,
        invt_a=False, linreg=False, figname='chart_g1')
    line_reg_with_marginal_distr(
        df, 'acc', 'fair', tag_Ys, picked_keys,
        invt_a=False, linreg=True, snspec=True, figname='chart_g2')
    line_reg_with_marginal_distr(
        df, 'acc', 'fair', tag_Ys, picked_keys,
        invt_a=False, linreg=True, snspec=False, figname='chart_g3')
    line_reg_with_marginal_distr(
        df, 'acc', 'fair', tag_Ys, picked_keys,
        invt_a=True, linreg=False, figname='chart_g4')
    line_reg_with_marginal_distr(
        df, 'acc', 'fair', tag_Ys, picked_keys,
        invt_a=True, linreg=True, snspec=True, figname='chart_g5')
    line_reg_with_marginal_distr(
        df, 'acc', 'fair', tag_Ys, picked_keys,
        invt_a=True, linreg=True, snspec=False, figname='chart_g6')
    '''

    for i in range(4 + 2):
        snspec = 'sty' + str(i)
        line_reg_with_marginal_distr(
            df, 'acc', 'fair', tag_Ys, picked_keys, invt_a=False,
            snspec=snspec, figname='chart_gx_g{}'.format(i))
        line_reg_with_marginal_distr(
            df, 'acc', 'fair', tag_Ys, picked_keys, invt_a=True,
            snspec=snspec, figname='chart_gy_g{}'.format(i))

    for snspec in ['sty4a', 'sty4b']:
        line_reg_with_marginal_distr(
            df, 'acc', 'fair', tag_Ys, picked_keys, invt_a=False,
            snspec=snspec, figname='chart_gx_g{}'.format(snspec[-2:]))

    tX = df['acc'].values.astype('float')
    tY = df['fair'].values.astype('float')
    single_line_reg_with_distr(tX, tY, figname='chart_hd1', snspec='sty1')
    single_line_reg_with_distr(tX, tY, distrib=True, figname='chart_hd2')
    single_line_reg_with_distr(
        tX, tY, distrib=True, snspec='sty5', figname='chart_hd5')
    for snspec in ['sty2', 'sty4', 'sty6']:
        single_line_reg_with_distr(
            tX, tY, linreg=True, snspec=snspec,
            figname='chart_hl{}'.format(snspec[-1]))
    for snspec in ['sty3a', 'sty3b']:
        single_line_reg_with_distr(
            tX, tY, linreg=True, snspec=snspec,
            figname='chart_hl{}'.format(snspec[-2:]))


def generate_dfs(num, bl):
    dat_acc = np.random.rand(num) * .1 + .7
    dat_fir = np.random.rand(num, bl) * .1 + .6
    df = pd.DataFrame(np.concatenate([
        dat_acc.reshape(-1, 1), dat_fir], axis=1), columns=[
        'acc', 'GM1', 'GM2', 'GM3', 'fair'])

    df_1 = df[['acc', 'GM1']].rename(columns={'GM1': 'fair'})
    df_2 = df[['acc', 'GM2']].rename(columns={'GM2': 'fair'})
    df_3 = df[['acc', 'GM3']].rename(columns={'GM3': 'fair'})
    df_1['learning'] = 'DP'
    df_2['learning'] = 'EO'
    df_3['learning'] = 'PQP'
    df_alternative = pd.concat([
        df_1, df_2, df_3], axis=0).reset_index(drop=True)  # 合并表格

    tag_Ys = ['GM1', 'GM2', 'GM3']
    picked_keys = ['DP', 'EO', 'PQP']
    col_X, col_Y = 'acc', 'fair'
    return df, df_alternative, tag_Ys, picked_keys, col_X, col_Y


def test_fairmanf_exp2b():
    # from prgm.nucleus.oracle_graph import (
    #     line_reg_with_marginal_distr, single_line_reg_with_distr,
    #     scatter_with_marginal_distrib, lineplot_with_uncertainty)
    num, bl = 100, 4
    df, _, tag_Ys, picked_keys, col_X, col_Y = generate_dfs(num, bl)
    assert isinstance(tag_Ys, list)

    scatter_with_marginal_distrib(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_sd1')
    '''
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_sd2')
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_sd3',
        distrib=False)
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_sd4',
        invt_a=True)
    # default: invt_a=False, snspec='sty0', distrib=True,
    '''

    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_slr5',
        snspec='sty5')
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_slr4a',
        snspec='sty4a', identity='identity')
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_slr4b',
        snspec='sty4b', identity='identity')

    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_slr5p',
        snspec='sty5', distrib=False, identity='  identity')
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_slr4ap',
        snspec='sty4a', identity='identity', distrib=False)
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_slr4bp',
        snspec='sty4b', identity='identity', distrib=False)

    '''
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_slr1',
        snspec='sty1')
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_slr3',
        snspec='sty3')
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_slr2',
        snspec='sty2')

    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_slr1p',
        snspec='sty1', distrib=False)
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_slr3p',
        snspec='sty3', distrib=False)
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys, figname='chart_slr2p',
        snspec='sty2', distrib=False)
    '''


# -----------------------


# -----------------------
# nucleus.utils_graph.py


def test_fairmanf_ext_plt3():
    # from prgm.nucleus.oracle_graph import (
    #     single_line_reg_with_distr, multi_lin_reg_with_distr)
    num, bl = 100, 4
    df, _, tag_Ys, picked_keys, _, _ = generate_dfs(num, bl)
    assert isinstance(tag_Ys, list) and len(tag_Ys) == 3
    assert len(picked_keys) == 3  # _,_: col_X,col_Y

    tX = df['acc'].values.astype('float')
    tY = df['fair'].values.astype('float')
    # single_line_reg_with_distr(tX, tY, figname='chart_hd1', snspec='sty1')
    '''
    single_line_reg_with_distr(tX, tY, distrib=True, figname='chart_hd2a')
    single_line_reg_with_distr(tX, tY, distrib=True, snspec='sty5', figname='chart_hd5')
    single_line_reg_with_distr(tX, tY, linreg=True, snspec='sty2', figname='chart_hd2b')
    single_line_reg_with_distr(tX, tY, linreg=True, snspec='sty3a', figname='chart_hd3a')
    '''
    single_line_reg_with_distr(tX, tY, linreg=True, snspec='sty3b', figname='chart_hd3b')
    single_line_reg_with_distr(tX, tY, linreg=True, snspec='sty4', figname='chart_hd4')
    single_line_reg_with_distr(tX, tY, linreg=True, snspec='sty6', figname='chart_hl6')

    # df, df_alt, _, _, _, _ =
    tZ = np.random.rand(num) * .2
    tXs = [tX + tZ, tX - tZ]
    tYs = [[df['GM1'].values.astype('float'),
            df['GM2'].values.astype('float')],
           df['GM3'].values.astype('float')]
    # multi_lin_reg_with_distr(tXs, tYs, snspec='sty6', figname='chart_md6')
    # multi_lin_reg_with_distr(tXs, tYs, snspec='sty4', figname='chart_md4')
    # multi_lin_reg_with_distr(tXs, tYs, snspec='sty3a', figname='chart_md3a')
    # multi_lin_reg_with_distr(tXs, tYs, snspec='sty3b', figname='chart_md3b')
    tZs = [['GM1', 'GM2'], 'GM3']
    multi_lin_reg_with_distr(
        tXs, tYs, tZs, snspec='sty6', figname='chart_md6')
    multi_lin_reg_with_distr(
        tXs, tYs, tZs, snspec='sty4', figname='chart_md4')
    multi_lin_reg_with_distr(
        tXs, tYs, tZs, snspec='sty3a', figname='chart_md3a')
    multi_lin_reg_with_distr(
        tXs, tYs, tZs, snspec='sty3b', figname='chart_md3b')

    # pdb.set_trace()
    return


def test_fairmanf_ext_plt4s():
    # from prgm.nucleus.oracle_graph import (
    #     scatter_with_marginal_distrib, line_reg_with_marginal_distr)
    num, bl = 100, 4
    df, _, tag_Ys, picked_keys, col_X, col_Y = generate_dfs(num, bl)
    scatter_with_marginal_distrib(df, col_X, col_Y, tag_Ys, picked_keys, figname='cheers_ls0')
    line_reg_with_marginal_distr(df, col_X, col_Y, tag_Ys, picked_keys, figname='cheers_li0')
    kws = {'identity': 'identity'}  # None}

    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys,
        snspec='sty1', figname='cheers_li1')  # ,**kws)
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys,
        snspec='sty1', figname='cheers_li1p', **kws)
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys,
        snspec='sty5a', figname='cheers_li5a', identity=None)
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys,
        snspec='sty5b', figname='cheers_li5b', **kws)

    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys,
        snspec='sty4a', figname='cheers_li4a', **kws)
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys,
        snspec='sty4b', figname='cheers_li4b', **kws)

    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys,
        snspec='sty2', figname='cheers_li2')  # ,**kws)
    line_reg_with_marginal_distr(
        df, col_X, col_Y, tag_Ys, picked_keys,
        snspec='sty3', figname='cheers_li3')  # ,**kws)
    # line_reg_with_marginal_distr(df, col_X, col_Y, tag_Ys, picked_keys, snspec='sty6', figname='cheers_li6', **kws)

    return


# -----------------------


def test_fairGBM():
    # from prgm.nucleus.oracle_graph import (
    #     FairGBM_scatter, FairGBM_tradeoff_v1,
    #     FairGBM_tradeoff_v2, FairGBM_tradeoff_v3)
    n, it = 3, 11  # nb_model
    annot_model = ['Model {}'.format(i + 1) for i in range(n)]
    # annot_fair = ['DP', 'EO', 'PQP', 'DR']
    label = ('error rate', 'Fairness')
    Xs = np.random.rand(n, it)
    Ys = np.random.rand(n, it)
    kws = {'num_gap': 100, 'alpha_loc': 'b4'}
    FairGBM_tradeoff_v3(Xs, Ys, annot_model, label,
                        figname='fairgbm_v3', **kws)
    FairGBM_tradeoff_v2(Xs, Ys, annot_model, label,
                        figname='fairgbm_v2', **kws)
    # pdb.set_trace()
    return


# -----------------------
