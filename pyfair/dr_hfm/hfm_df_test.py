# coding: utf-8
#
# Usage: to measure the bias level within one classifier
#


import numpy as np
# import pdb
from pyfair.dr_hfm.hfm_df import bias_degree_bin, bias_degree_nonbin

from pyfair.dr_hfm.earlybreak import (
    EffHD_bin, EffHD_nonbin, EffHD_multivar,
    Naive_bin, Naive_nonbin, Naive_multivar,
    NaiveHDD, HDD_earlybreak, HDD_randomize)
from pyfair.dr_hfm.dist_drt import (
    DirectDist_bin, DirectDist_nonbin, DirectDist_multiver)
from pyfair.facil.utils_const import check_equal


def test_bias_level():
    res = bias_degree_bin(0, 0)
    ans = bias_degree_bin(0, 1e-7)
    # pdb.set_trace()
    assert res[0] == 0
    assert ans[0] >= 0

    res = bias_degree_nonbin(0, 0)
    ans_1 = bias_degree_nonbin(0, 1e-7)
    ans_2 = bias_degree_nonbin(0, 1e7)
    assert res[0] == 0
    assert ans_1[0] >= 0
    assert ans_2[0] >= 0
    return


# ------------------------------------------


def test_hausdorff():
    n, nd, nb = 110, 4, 56
    A = np.random.rand(nb, nd)
    B = np.random.rand(n - nb, nd)

    ind = HDD_randomize(A)
    assert len(set(ind)) == nb
    ans = NaiveHDD(A, B)
    res = HDD_earlybreak(A, B)

    assert check_equal(ans, res)
    assert ans == res
    return


def test_earlybreak():
    n, nd = 110, 4
    X_nA_y = np.random.rand(n, 1 + nd)
    idx_Si = np.random.randint(2, size=n, dtype='bool')
    Si = X_nA_y[idx_Si]
    Si_c = X_nA_y[~idx_Si]
    assert len(Si) + len(Si_c) == n

    A_j = np.zeros((n, 2))
    idx_Sj = np.random.randint(3, size=n)
    A_j[:, 0] = idx_Sj
    idx_Sjs = [idx_Sj == 1, idx_Sj == 0, idx_Sj == 2]
    idx_Sk = np.random.randint(4, size=n)
    A_j[:, 1] = idx_Sk
    idx_Sks = [idx_Sk == 1, idx_Sk == 0, idx_Sk == 2, idx_Sk == 3]
    idx_Ai_Sj = [idx_Sjs, idx_Sks]
    del idx_Sj, idx_Sk, idx_Sks

    ans = DirectDist_bin(X_nA_y, idx_Si)
    tmp = Naive_bin(X_nA_y, idx_Si)
    res = EffHD_bin(X_nA_y, idx_Si)
    # pdb.set_trace()
    assert check_equal(ans[0][0], [tmp[0], res[0]])

    ans = DirectDist_nonbin(X_nA_y, idx_Sjs)
    tmp = Naive_nonbin(X_nA_y, idx_Sjs)
    res = EffHD_nonbin(X_nA_y, idx_Sjs)
    # pdb.set_trace()
    assert check_equal(ans[0][0], [tmp[0], res[0]])
    # '''
    # (Pdb) tmp   (0.6579016325539099, 0.1144411563873291)
    # (Pdb) res   (0.6579016325539099, 0.0027289390563964844)
    #
    # (Pdb) tmp   (0.5805998775010773, 0.0011272430419921875)
    # (Pdb) res   (0.5805998775010773, 0.0023679733276367188)
    # (Pdb) ans   ((0.5805998775010773, 0.30795127445857207), 0.0011289119720458984)
    # '''

    ans = DirectDist_multiver(X_nA_y, idx_Ai_Sj)
    tmp = Naive_multivar(X_nA_y, idx_Ai_Sj)
    res = EffHD_multivar(X_nA_y, idx_Ai_Sj)
    # pdb.set_trace()
    ans = (ans[0][:-1], ans[1])
    tmp = (tmp[0][0], tmp[1])
    res = (res[0][0], res[1])
    assert check_equal(ans[0][0], [tmp[0], res[0]])
    return
