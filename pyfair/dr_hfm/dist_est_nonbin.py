# coding: utf-8
# Usage: to approximate the distance quickly
#
# Author: Yj
# 2. Approximating Discrimination Within Models When Faced With Several
#    Non-Binary Sensitive Attributes [https://arxiv.org/abs/2408.06099]
#


import numpy as np
import numba

from pyfair.facil.utils_timer import fantasy_timer
from pyfair.dr_hfm.dist_drt import DistDirect_Euclidean
from pyfair.dr_hfm.dist_est_bin import projector


# ==========================================
# Estimated distance between sets


# ------------------------------------------
# Algorithm 3. Sub-routines


@numba.jit(nopython=True)
def set_belonging(A_anchor, Ai_j):
    return True if A_anchor == Ai_j else False


@numba.jit(nopython=True)
def sub_accelerator_smaler(X_yfx, Ai, idx_y_fx, i, m2):
    i_anchor = idx_y_fx[i]
    A_anchor = Ai[i_anchor]
    X_yfx_anchor = X_yfx[i_anchor]

    # Compute the distances d(anchor,\cdot) for at most m2 nearby
    # data points that meets a!=ai and g()<=g(xi,yi;w)
    j, num_j, min_js = i, 0, np.finfo(np.float32).max
    j = i - 1  # doesn't have to be compared with the anchor
    while num_j < m2:
        if j < 0:
            break

        idx_j = idx_y_fx[j]
        if set_belonging(A_anchor, Ai[idx_j]):
            j -= 1
            continue

        curr = DistDirect_Euclidean(X_yfx_anchor, X_yfx[idx_j])
        if curr < min_js:
            min_js = curr
        # Find the minimum among them, recorded as d_min^s

        num_j += 1
        j -= 1
    return min_js


@numba.jit(nopython=True)
def sub_accelerator_larger(X_yfx, Ai, idx_y_fx, i, m2):
    i_anchor = idx_y_fx[i]
    A_anchor = Ai[i_anchor]
    X_yfx_anchor = X_yfx[i_anchor]

    # Compute the distances d(anchor,\cdot) for at most m2 nearby
    # data points that meets a!=ai and g()>=g(xi,yi;w)
    j, num_j, min_jr = i, 0, np.finfo(np.float32).max
    j = i + 1  # doesn't have to be compared with the anchor
    n = len(X_yfx)
    while num_j < m2:
        if j >= n:
            break

        idx_j = idx_y_fx[j]
        if set_belonging(A_anchor, Ai[idx_j]):
            j += 1
            continue

        curr = DistDirect_Euclidean(X_yfx_anchor, X_yfx[idx_j])
        if curr < min_jr:
            min_jr = curr
        # Find the minimum among them, record as d_min^r

        num_j += 1
        j += 1
    return min_jr


# ------------------------------------------
# Algorithm 3. AcceleDist

# """ parameters
# X_nA_y: np.ndarray, size (n, 1+n_d)
# A_i: np.ndarray, size (n,)
#      indicating which group the instance belongs to,
#      corresponding to one single sensitive attribute
# m1 : scalar, number of repetition
# m2 : scalar, number of comparison
# vec_w: np.ndarray, size (1+n_d,)
# """


@fantasy_timer
def AcceleDist_nonbin(X_nA_y, A_j, m2, vec_w):
    proj = [projector(ele, vec_w) for ele in X_nA_y]
    idx_y_fx = np.argsort(proj)

    n = X_nA_y.shape[0]  # number of instances
    d_min = []
    for i in range(n):
        # Set the anchor data point (xi,yi) in this round

        min_js = sub_accelerator_smaler(X_nA_y, A_j, idx_y_fx, i, m2)
        min_jr = sub_accelerator_larger(X_nA_y, A_j, idx_y_fx, i, m2)

        # finally,
        tmp = min(min_js, min_jr)
        d_min.append(tmp)
    return max(d_min), sum(d_min)


# ------------------------------------------
# Algorithm 2. ApproxDist


@fantasy_timer
def orthogonal_weight(n_d, n_e=3):
    for _ in range(n_d):
        B = np.random.rand(n_d, n_d)
        tmp = np.linalg.det(B)
        if tmp != 0:
            break

    A_T = B.T.copy()  # schmidt
    tmp = A_T[0] / np.sqrt(np.dot(A_T[0], A_T[0]))
    eta = [tmp]
    for i in range(1, n_e):

        schmidt = np.zeros(n_d)
        for j in range(i):
            schmidt += np.dot(A_T[i], eta[j]) * eta[j]
        tmp = A_T[i] - schmidt
        tmp = tmp / np.sqrt(np.dot(tmp, tmp))

        eta.append(tmp)
    return np.array(eta)


@fantasy_timer
def ApproxDist_nonbin(X_nA_y, A_j, m1, m2, n_e=2):
    n, n_d = X_nA_y.shape  # n_d-1: number of non-sen att(s)
    # n_d-1: number of non-sensitive attributes
    d_max, d_avg = [], []
    for _ in range(m1):  # for j in

        # Take two orthogonal vectors $w_0$ and $w_1$ where each $w_k
        # \in [-1,+1]^{1+n_x} (k=\{0,1\})$
        # Or take three orthogonal vectors. Your choice.
        W, _ = orthogonal_weight(n_d, n_e)

        tmp = [AcceleDist_nonbin(
            X_nA_y, A_j, m2, W[k]) for k in range(n_e)]
        tmp, _ = zip(*tmp)
        t_max, t_avg = zip(*tmp)

        d_max.append(min(t_max))
        d_avg.append(min(t_avg))
    return min(d_max), min(d_avg) / float(n)


@fantasy_timer
def ApproxDist_nonbin_mpver(X_nA_y, A_j, m1, m2, n_e=2, pool=None):
    n, n_d = X_nA_y.shape
    d_max, d_avg = [], []
    X_nA_y_map = [X_nA_y] * n_e
    A_j_map = [A_j] * n_e
    m2_map = [m2] * n_e

    if pool is None:
        W = list(map(orthogonal_weight, [n_d] * m1, [n_e] * m1))
        W, _ = zip(*W)  # ignoring time cost
        for j in range(m1):
            tmp = list(map(AcceleDist_nonbin,
                           X_nA_y_map, A_j_map, m2_map, W[j]))
            tmp, _ = zip(*tmp)
            t_max, t_avg = zip(*tmp)
            d_max.append(min(t_max))
            d_avg.append(min(t_avg))

    else:
        W = pool.map(orthogonal_weight, [n_d] * m1, [n_e] * m1)
        W, _ = zip(*W)
        for j in range(m1):
            tmp = pool.map(AcceleDist_nonbin,
                           X_nA_y_map, A_j_map, m2_map, W[j])
            tmp, _ = zip(*tmp)
            t_max, t_avg = zip(*tmp)
            d_max.append(min(t_max))
            d_avg.append(min(t_avg))

    return min(d_max), min(d_avg) / float(n)


# ------------------------------------------
# Algorithm 1. ExtendDist


@fantasy_timer
def ExtendDist_multiver_mp(X_nA_y, A, m1, m2, n_e=3, pool=None):
    # n, n_a = A.shape  # n: number of instances
    _, n_a = A.shape  # n_a: number of sensitive attributes

    X_nA_y_map = [X_nA_y] * n_a
    A_i_map = [A[:, j].copy() for j in range(n_a)]
    m1_map = [m1] * n_a
    m2_map = [m2] * n_a
    ne_map = [n_e] * n_a

    if pool is None:
        tmp = list(map(ApproxDist_nonbin,
                       X_nA_y_map, A_i_map, m1_map, m2_map, ne_map))
    else:
        tmp = pool.map(ApproxDist_nonbin,
                       X_nA_y_map, A_i_map, m1_map, m2_map, ne_map)
    del X_nA_y_map, A_i_map, m1_map, m2_map, ne_map

    tmp, half_ut = zip(*tmp)
    d_max, d_avg = zip(*tmp)
    return max(d_max), sum(d_avg) / float(n_a), (
        d_max, d_avg, half_ut)


# ------------------------------------------
#
