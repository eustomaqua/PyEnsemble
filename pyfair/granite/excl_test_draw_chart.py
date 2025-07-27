# coding: utf-8

import numpy as np
# from prgm.nucleus.utils_chart import *
# from fairml.facilc.draw_chart import (
from pyfair.granite.draw_chart import (
    # multi_scatter_hor, multi_scatter_vrt, multiple_scatter_chart,
    multiple_scatter_chart,
    analogous_confusion, multiple_scatter_alternative,
    analogous_confusion_alternative, single_hist_chart,
    lines_with_std_3d, lines_with_std_2d, discrete_bar_comparison,
    PLT_LOCATION, PLT_FRAMEBOX, _line_std_drawer)
# from pyfair.senior.draw_chart import _setup_config, _setup_figshow

np.random.seed(1523)
sz = 21
# df = pd.DataFrame({'X': X, 'Y': Ys[0]})
annots = ('X', 'Ys')
annotZs = ('Y1', 'Y2', 'Y3', 'Y4')


def test_multi_scatter():
    # figsize, base, locate = 'M-WS', None, PLT_LOCATION
    # fsz = (7, 6)
    identity, figname = False, 'chart_d'

    X = np.random.randint(100, size=sz).astype('float')
    Ys = np.random.rand(4, sz).transpose() * 100
    Ys = Ys.T

    # '''
    # multi_scatter_hor(X, Ys, annots, annotZs, figname + '1',
    #                   figsize, identity, base, locate)
    # multi_scatter_vrt(X, Ys, annots, annotZs, figname + '2',
    #                   figsize, identity, base, locate)
    # '''

    # identity = False
    multiple_scatter_chart(
        X, Ys, annots, annotZs, figname, ind_hv='h',
        identity=identity)
    multiple_scatter_chart(
        X, Ys, annots, annotZs, figname, ind_hv='v',
        identity=identity)


def test_mu_scatter_alter():
    # from prgm.nucleus.utils_chart import (
    from pyfair.granite.draw_chart import (
        _alternative_multi_scatter_hor,
        _alternative_multi_scatter_vrt)
    identity, locate, box = True, PLT_LOCATION, PLT_FRAMEBOX
    invt, fn = True, 'chart_e'

    # https://www.apa.org/topics/racism-bias-discrimination/types-stress
    # race, gender, age, or sexual orientation.
    sens = ['race', 'sex', 'age', 'orientate']
    kwargs = {"handletextpad": .04, "borderpad": .27}
    Zs = [np.random.rand(4, sz) for _ in range(4)]
    N = [np.random.rand(sz) for _ in range(4)]

    _alternative_multi_scatter_hor(
        N, Zs, sens, annots, annotZs, fn + '3',
        identity, locate, box, invt, kwargs)
    _alternative_multi_scatter_vrt(
        N, Zs, sens, annots, annotZs, fn + '4',
        identity, locate, box, invt, kwargs)

    multiple_scatter_alternative(
        N, Zs, sens, annots, annotZs, fn, ind_hv='h')
    multiple_scatter_alternative(
        N, Zs, sens, annots, annotZs, fn, ind_hv='v')


def test_analogous_confu():
    nb_iter, criteria = 5, 7
    Mat = np.random.rand(nb_iter, criteria) * 100
    key = ['Acc', 'P', 'R', 'F_1', 'tpr', 'fpr', 'fnr']

    fn = 'chart_f'
    Mat = Mat.T
    key[3] = r'f$_1$'  # r'$f_1$'

    analogous_confusion(Mat, key, fn + '5')

    sens = ['race', 'sex', 'age', 'orientate']
    analogous_confusion_alternative(
        [Mat for _ in range(4)], sens, key, fn + '6')
    analogous_confusion_alternative(
        [Mat for _ in range(4)], sens, key, fn + '6p',
        normalize=True)


def test_hist_chart():
    # from prgm.nucleus.utils_chart import (
    # from fairml.facilc.draw_chart import (
    #     _line_std_drawer, _line_std_colors)

    Ys = np.random.rand(sz, 4).tolist()
    Y_avg = np.mean(Ys, axis=0)
    Y_std = np.std(Ys, axis=0)
    sens = ['race', 'sex', 'age', 'orient']

    single_hist_chart(
        Y_avg, Y_std, sens, 'X', 'Y', 'chart_f7', rotate=15)

    avg_1, r1_p, r2_p = _line_std_drawer(Ys)
    avg_0, r1_q, r2_q = _line_std_drawer(Ys, ddof=0)
    assert np.all(np.equal(avg_1, avg_0))
    assert len(r1_p) == len(r2_p) == len(r1_q) == len(r2_q)

    X = np.random.rand(sz).tolist()
    X = sorted(X)
    Ys = np.transpose(Ys).tolist()
    lines_with_std_2d(X, Ys, 'chart_f8', annotY=sens)

    nb_iter = 5
    Ys = np.random.rand(4, sz, nb_iter).tolist()
    lines_with_std_3d(X, Ys, 'chart_f9', annotY=sens)


def test_cfalab_bar():
    _mx = 5  # _max
    IndexSlices = np.random.randint(_mx, size=[4, _mx])
    SubIndices = np.random.randint(_mx, size=[2, _mx])

    fn = 'chart_g0'
    # d1, d2 =
    discrete_bar_comparison(IndexSlices, SubIndices, fn)
    discrete_bar_comparison(
        IndexSlices, SubIndices, fn, split=False)

    fn = fn.replace('g0', 'h0')
    # d3, d4 =
    discrete_bar_comparison(
        IndexSlices, SubIndices, fn, density=True)
    discrete_bar_comparison(
        IndexSlices, SubIndices, fn, density=True, split=False)
