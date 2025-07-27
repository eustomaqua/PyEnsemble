# coding: utf-8
#
# Usage: to calculate the distance directly
#


from pyfair.dr_hfm.dist_drt import (
    DistDirect_Euclidean, DistDirect_halfway_min, DistDirect_mediator,
    DirectDist_bin, DirectDist_nonbin, DirectDist_multiver)
import numpy as np
# import pdb
from pyfair.facil.utils_const import check_equal, check_belong


def generate_dat(n, nd, na, nai, nc=2):
    """ parameters
    n  : number of instances in a dataset
    nd : number of non-sensitive features
    na : number of sensitive attributes
    nai: number of values within one sensitive attribute
    nc : number of classes/labels
    """

    X = np.random.rand(n, nd) * 10
    y = np.random.randint(nc, size=n)  # binary classification
    A = np.random.randint(nai, size=(n, na))

    X_nA_y = np.concatenate([y.reshape(-1, 1), X], axis=1)
    indices = [[A[:, i] == j for j in range(nai)] for i in range(na)]
    return X_nA_y, indices


def no_less_than_check(A_pl, b):
    if not check_belong(A_pl, list, tuple):
        assert A_pl >= b or check_equal(A_pl, b, 1e-5)
        return

    for j in A_pl[1:]:
        assert A_pl[0] == j
    assert A_pl[0] >= b or check_equal(A_pl[0], b)
    return


def test_mediator():
    n, nd, na, nai = 30, 4, 2, 2
    X_nA_y, indices = generate_dat(n, nd, na, nai)
    idx_Si = indices[0][1]
    Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]

    res = DistDirect_Euclidean(Sj[0], Sj_c[1])
    assert res >= 0
    tmp = DistDirect_halfway_min(Sj[0], Sj_c)
    assert tmp >= 0
    ans = DistDirect_mediator(X_nA_y, idx_Si)
    assert ans[0] >= 0 and ans[1] >= 0

    # pdb.set_trace()
    return


def compare_direct(nai):
    n, nd, na = 30, 4, 2
    X_nA_y, indices = generate_dat(n, nd, na, nai)

    res_alt = DirectDist_bin(X_nA_y, indices[0][1])
    res = DirectDist_bin(X_nA_y, indices[0][1])
    assert res_alt[0][0] == res[0][0] >= 0  # max
    assert res_alt[0][1] == res[0][1] >= 0  # avg
    # pdb.set_trace()
    # # assert res_alt[1] > res[1]  # time_elapsed
    # # assert res_alt[1] > res[1] or check_equal()
    # no_less_than_check(res_alt[1], res[1])

    tmp_2, _ = DirectDist_nonbin(X_nA_y, indices[1])
    tmp_1, _ = DirectDist_nonbin(X_nA_y, indices[0])
    ans, tim = DirectDist_multiver(X_nA_y, indices)
    ans_max, ans_avg, tmp = ans

    for i in [0, 1, ]:  # max, avg,
        assert tmp[i][0] == tmp_1[i] and tmp[i][1] == tmp_2[i]
    assert ans_max == max(tmp_1[0], tmp_2[0]) == max(tmp[0])
    assert ans_avg == (tmp_1[1] + tmp_2[1]) / na
    assert ans_avg == sum(tmp[1]) / na

    assert tim >= sum(tmp[2])  # time_elapsed
    assert ans_max >= 0 and ans_avg >= 0
    return


def test_direct_dist():
    # n, nd, na, nai = 30, 4, 2, 2
    compare_direct(2)
    compare_direct(4)
    compare_direct(3)
    return
