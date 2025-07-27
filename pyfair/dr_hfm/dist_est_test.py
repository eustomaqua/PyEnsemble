# coding: utf-8
#
# Usage: to approximate the distance quickly
#        when faced with one bi-valued sensitive attribute
#


import numpy as np
from pathos import multiprocessing as pp
import pdb


from pyfair.dr_hfm.dist_est_bin import (
    weight_generator, AcceleDist_bin, ApproxDist_bin,
    projector, ApproxDist_bin_revised)
from pyfair.dr_hfm.dist_est_nonbin import (
    orthogonal_weight, AcceleDist_nonbin, ApproxDist_nonbin,
    ApproxDist_nonbin_mpver)  # , ExtendDist_multiver_mp)

from pyfair.dr_hfm.dist_drt import (
    DirectDist_bin, DirectDist_nonbin, DirectDist_multiver)
from pyfair.dr_hfm.dist_drt_test import no_less_than_check


from pyfair.dr_hfm.dist_est_bin import (
    AcceleDist_bin_alter, ApproxDist_bin_alter,
    sub_accelerator_smaler, subalt_accel_smaler,
    sub_accelerator_larger, subalt_accel_larger)
# from pyfair.dr_hfm.dist_drt import DistDirect_Euclidean


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
    vec_w = np.random.rand(1 + nd)
    vec_w /= np.sum(vec_w)
    return X_nA_y, A, indices, vec_w


def compare_accele(nai, m1, m2):
    n, nd, na = 30, 4, 2
    X_nA_y, A, indices, vec_w = generate_dat(n, nd, na, nai)
    k = 0
    idx_S1, Ap = indices[k][1], A[:, k]
    idx_S0 = ~idx_S1

    # res_alt = DirectDist_bin(X_nA_y, idx_S1)
    res_1 = DirectDist_bin(X_nA_y, idx_S1)
    index_alt = [~indices[k][1], indices[k][1]]  # indices[0]
    res_2 = DirectDist_nonbin(X_nA_y, index_alt)
    assert res_1[0][0] == res_2[0][0]  # max
    assert res_1[0][1] == res_2[0][1]  # avg

    Aq = Ap.copy()
    Aq[Ap > 1] = 0
    tmp_1 = AcceleDist_bin(X_nA_y, Aq, idx_S0, idx_S1, m2, vec_w)
    tmp_2 = AcceleDist_nonbin(X_nA_y, Aq, m2, vec_w)
    tmp_3 = AcceleDist_bin(X_nA_y, Ap, idx_S0, idx_S1, m2, vec_w)
    tmp_4 = AcceleDist_nonbin(X_nA_y, Ap, m2, vec_w)
    # # assert tmp_1[0] == tmp_2[0][0] == tmp_3[0] >= tmp_4[0][0]
    # # assert tmp_2[0][1] >= tmp_4[0][1]  # avg  #↑↓ max `tat=tmp_alt`
    # assert tmp_1[0][0] == tmp_2[0][0] == tmp_3[0][0] >= tmp_4[0][0]
    # assert tmp_1[0][1] == tmp_2[0][1] == tmp_3[0][1] >= tmp_4[0][1]

    tmp_1, _ = tmp_1
    tmp_2, _ = tmp_2
    tmp_3, _ = tmp_3
    tmp_4, _ = tmp_4
    # pdb.set_trace()

    # '' '
    # assert tmp_1[0] == tmp_2[0] == tmp_3[0]  # >= tmp_4[0]
    # assert tmp_1[1] == tmp_2[1] == tmp_3[1]  # >= tmp_4[1]
    # assert tmp_3[0] >= tmp_4[0] or check_equal(tmp_3[0], tmp_4[0])
    # assert tmp_3[1] >= tmp_4[1] or check_equal(tmp_3[1], tmp_4[1])
    # '' '

    no_less_than_check([tmp_1[0], tmp_2[0], tmp_3[0]], res_2[0][0])
    no_less_than_check([tmp_1[1], tmp_2[1], tmp_3[1]], res_2[0][1])
    # error
    # no_less_than_check([tmp_1[0], tmp_2[0], tmp_3[0]], tmp_4[0])
    # no_less_than_check([tmp_1[1], tmp_2[1], tmp_3[1]], tmp_4[1])
    return


def compare_approx(nai, m1, m2, n_e=2):
    n, nd, na = 30, 4, 2
    X_nA_y, A, indices, vec_w = generate_dat(n, nd, na, nai)
    k = 0
    idx_S1, Ap = indices[k][1], A[:, k]
    # idx_S0 = ~idx_S1
    Aq = Ap.copy()
    Aq[Ap > 1] = 0

    res_1 = DirectDist_bin(X_nA_y, idx_S1)
    index_alt = [~indices[k][1], indices[k][1]]
    res_2 = DirectDist_nonbin(X_nA_y, index_alt)
    res_3 = DirectDist_nonbin(X_nA_y, indices[k])
    assert res_1[0][0] == res_2[0][0]  # max
    assert res_1[0][1] == res_2[0][1]  # avg

    ans_1 = ApproxDist_bin(X_nA_y, Aq, idx_S1, m1, m2)
    ans_2 = ApproxDist_bin_revised(X_nA_y, idx_S1, m1, m2)  # Aq,
    ans_3 = ApproxDist_nonbin(X_nA_y, Aq, m1, m2, n_e)
    ans_4 = ApproxDist_bin(X_nA_y, Ap, idx_S1, m1, m2)
    ans_5 = ApproxDist_bin_revised(X_nA_y, idx_S1, m1, m2)  # Ap,
    ans_6 = ApproxDist_nonbin(X_nA_y, Ap, m1, m2, n_e)

    ans_1, _ = ans_1
    ans_2, _ = ans_2
    ans_3, _ = ans_3
    ans_4, _ = ans_4
    ans_5, _ = ans_5
    ans_6, _ = ans_6

    tmp_2, _ = ApproxDist_bin_revised(X_nA_y, ~idx_S1, m1, m2)  # Aq,
    tmp_5, _ = ApproxDist_bin_revised(X_nA_y, ~idx_S1, m1, m2)  # Ap,
    # assert ans_2[0] == ans_5[0] == tmp_2[0] == tmp_5[0]

    # pdb.set_trace()
    for i in [0, 1]:  # max, avg
        if i == 0:
            no_less_than_check(ans_1, res_2[0][i])
            no_less_than_check(ans_4, res_2[0][i])

        no_less_than_check(ans_2[i], res_2[0][i])
        no_less_than_check(ans_3[i], res_2[0][i])
        no_less_than_check(ans_5[i], res_2[0][i])
        no_less_than_check(ans_6[i], res_3[0][i])

    # assert ans_1 == ans_2[0] == ans_3[0] == ans_4 == ans_5[0] >= ans_6[0]
    # assert ans_2[1] == ans_3[1] == ans_5[1] >= ans_6[1]  # avg  # ↑ max
    # '' '
    # assert ans_1 >= res_2[0][0]
    # assert ans_2[0] >= res_2[0][0] and ans_2[1] >= res_2[0][1]
    # assert ans_3[0] >= res_2[0][0] and ans_3[1] >= res_2[0][1]
    # assert ans_4 >= res_2[0][0]
    # assert ans_5[0] >= res_2[0][0] and ans_5[1] >= res_2[0][1]
    # assert ans_6[0] >= res_3[0][0] and ans_6[1] >= res_3[0][1]
    # '' '

    # assert ans_2[0] >= res_2[0] or check_equal(
    #     ans_2[0], res_2[0])     # max
    # assert ans_2[1] >= res_2[1] or check_equal(
    #     ans_2[1], res_2[1])     # avg
    # assert ans_1 >= res_2[0] or check_equal(
    #     ans_1, res_2[0])        # max

    del vec_w
    return


def compare_multiver(nai, m1, m2, n_e=2):
    n, nd, na = 30, 4, 2
    X_nA_y, A, indices, vec_w = generate_dat(n, nd, na, nai)
    W, tim = orthogonal_weight(nd + 1, n_e)
    pool = pp.ProcessingPool(nodes = 3)  # mp_cores)
    assert np.dot(W[0], W[1]) < 10**8  # sum(W[0]*W[1])
    assert abs(1 - sum(vec_w)) < 10**8

    k = 0
    A_j = A[:, k]
    idx_S1 = A_j == 1
    idx_S0 = ~idx_S1
    tmp_1 = DirectDist_bin(X_nA_y, idx_S1)
    tmp_3 = DirectDist_nonbin(X_nA_y, [idx_S0, idx_S1])
    tmp_4 = DirectDist_nonbin(X_nA_y, indices[k])
    tmp_7, _ = DirectDist_multiver(X_nA_y, indices)

    tmp_1, _ = tmp_1
    tmp_3, _ = tmp_3
    tmp_4, _ = tmp_4
    _, _, tmp_7 = tmp_7
    tmp_7 = tmp_7[: -1]
    # pdb.set_trace()
    # assert tmp_1[0] == tmp_3[0] >= tmp_4[0] == tmp_7[0][k]  # max
    # assert tmp_1[1] == tmp_3[1] == tmp_4[1] == tmp_7[1][k]  # avg
    assert tmp_1[0] == tmp_3[0] and tmp_4[0] == tmp_7[0][k]  # max
    assert tmp_1[1] == tmp_3[1] and tmp_4[1] == tmp_7[1][k]  # avg

    res_1 = ApproxDist_bin(X_nA_y, A_j, idx_S1, m1, m2)
    res_2 = ApproxDist_bin_revised(X_nA_y, idx_S1, m1, m2)  # A_j,
    res_4 = ApproxDist_nonbin(X_nA_y, A_j, m1, m2, n_e)
    res_5 = ApproxDist_nonbin_mpver(X_nA_y, A_j, m1, m2, n_e)
    res_6 = ApproxDist_nonbin_mpver(X_nA_y, A_j, m1, m2, n_e, pool)

    res_1, _ = res_1
    res_2, _ = res_2
    res_4, _ = res_4
    res_5, _ = res_5
    res_6, _ = res_6
    # pdb.set_trace()

    for i in [0, 1]:  # max, avg
        no_less_than_check(res_2[i], tmp_1[i])
        assert tmp_1[i] == tmp_3[i]
        no_less_than_check(res_4[i], tmp_4[i])
        no_less_than_check(res_5[i], tmp_4[i])
        no_less_than_check(res_6[i], tmp_4[i])
    no_less_than_check(res_1, tmp_1[0])
    assert tmp_7[0][k] == tmp_4[0]  # max
    assert tmp_7[1][k] == tmp_4[1]  # avg

    del W
    return


def test_approx_dist():
    m1, m2 = 3, 5
    compare_accele(2, m1, m2)
    compare_accele(3, m1, m2)
    compare_approx(2, m1, m2)
    compare_approx(3, m1, m2)
    compare_multiver(2, m1, m2)
    compare_multiver(3, m1, m2)

    return


# def subcomp_alternative(
#         X_nA_y, A_j, idx_S0, idx_S1, m1, m2, vec_w):
#     from pyfair.dr_hfm.dist_est_bin import (
#         sub_accelerator_smaler, subalt_accel_smaler,
#         sub_accelerator_larger, subalt_accel_larger,)
#
#     proj = [projector(ele, vec_w) for ele in X_nA_y]
#     idx_y_fx = np.argsort(proj)
#     pdb.set_trace()
#     return


def compare_alternative(nai, m1, m2):
    n, nd, na = 30, 4, 2
    X_nA_y, A, indices, vec_w = generate_dat(n, nd, na, nai)
    W = weight_generator(n_d=5)
    assert W.shape[0] == 5 + 1

    k = 0
    A_j = A[:, k]
    idx_S1 = A_j == 1
    idx_S0 = ~idx_S1

    tmp_1 = AcceleDist_bin(X_nA_y, A_j, idx_S0, idx_S1, m2, vec_w)
    tmp_3 = AcceleDist_bin_alter(X_nA_y, idx_S0, idx_S1, m2, vec_w)
    ans_1 = DirectDist_bin(X_nA_y, idx_S1)
    res_1 = ApproxDist_bin(X_nA_y, A_j, idx_S1, m1, m2)
    res_2 = ApproxDist_bin_revised(X_nA_y, idx_S1, m1, m2)  # A_j,
    res_3 = ApproxDist_bin_alter(X_nA_y, idx_S1, m1, m2)

    res_1, _ = res_1
    res_2, _ = res_2
    res_4, _ = res_3
    tmp_1, _ = tmp_1
    tmp_3, _ = tmp_3
    ans_1, _ = ans_1

    # def subcomp_alternative(X_nA_y, A_j, idx_S0, idx_S1,
    #                         idx_y_fx, ik, m2):
    def subcomp_alternative(idx_y_fx, ik):
        ans_js = sub_accelerator_smaler(
            X_nA_y, A_j, idx_S0, idx_S1, idx_y_fx, ik, m2)
        ans_jr = sub_accelerator_larger(
            X_nA_y, A_j, idx_S0, idx_S1, idx_y_fx, ik, m2)
        res_js, _ = subalt_accel_smaler(
            X_nA_y, idx_S0, idx_S1, idx_y_fx, ik, m2)
        res_jr, _ = subalt_accel_larger(
            X_nA_y, idx_S0, idx_S1, idx_y_fx, ik, m2)
        # return ans_js, ans_jr, res_js, res_jr
        return ans_js, res_js, ans_jr, res_jr

    # def subcomp_subaccel(idx_y_fx, ik):
    #     from pyfair.dr_hfm.dist_est_bin import set_belonging
    #     i_anchor = idx_y_fx[ik]
    #     X_yfx_anchor = X_nA_y[i_anchor]
    #     return min(min_js, min_jr), min_js_list, min_jr_list

    if res_4[0] < ans_1[0]:
        # nai=2|3 都有可能出现
        # subcomp_alternative(X_nA_y, A_j, idx_S0, idx_S1, m1, m2, vec_w)
        proj = [projector(ele, vec_w) for ele in X_nA_y]
        idx_y_fx = np.argsort(proj)
        # intermediate = subcomp_alternative(
        #     X_nA_y, A_j, idx_S0, idx_S1, idx_y_fx, 0, m2)
        intermediate = subcomp_alternative(idx_y_fx, 0)
        # intermediate = subcomp_subaccel(idx_y_fx, 1)
        pdb.set_trace()

    no_less_than_check(res_1, ans_1[0])
    for i in [0, 1]:  # max,avg
        no_less_than_check(res_2[i], ans_1[i])
        no_less_than_check(res_4[i], ans_1[i])
        # res_4:
        # (6.469993525578606 >= 6.826606523423045 or False)
        # (7.182977542299324 >= 7.502924673636918 or False)
    no_less_than_check(tmp_1[0], ans_1[0])
    no_less_than_check(tmp_3[0], ans_1[0])
    return


def test_alternative_bin():
    # from pyfair.dr_hfm.dist_est_bin import (
    #     AcceleDist_bin_alter, ApproxDist_bin_alter)
    m1, m2 = 3, 5
    compare_alternative(2, m1, m2)
    return
