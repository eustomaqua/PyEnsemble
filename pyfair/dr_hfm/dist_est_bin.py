# coding: utf-8
# Usage: to approximate the distance quickly
#
# Author: Yj
# 1. Does Machine Bring in Extra Bias in Learning? Approximating Fairness
#    in Models Promptly [https://arxiv.org/abs/2405.09251 arXiv]
#


import numpy as np
import numba

from pyfair.facil.utils_timer import fantasy_timer
from pyfair.dr_hfm.dist_drt import DistDirect_Euclidean


# ==========================================
# Estimated distance between sets


# ------------------------------------------
# Algorithm 2: Sub-routes

# """ parameters
# # X, A, y, fx: np.ndarray
# # idx_S1: np.ndarray of `np.bool_`
# # m1, m2: scalar, hyperparameters
#
# X : np.ndarray, size (n, n_d)
# A : np.ndarray, size (n, n_a)
# y : np.ndarray, size (n,), true labels
# fx: np.ndarray, size (n,), prediction of one classifier
# idx_S1: np.ndarray, size (n,)
#       whether the instance belongs to the privileged group (if
#       True)
# m1: scalar, number of repetition
# m2: scalar, number of comparison
# vec_w : np.ndarray, size (1+n_d,)
# """


@numba.jit(nopython=True)
def set_belonging(idx_S0, idx_S1, i_anchor, idx_j):
    if idx_S0[i_anchor] == idx_S0[idx_j]:
        return True
    if idx_S1[i_anchor] == idx_S1[idx_j]:
        return True
    return False


@numba.jit(nopython=True)
def sub_accelerator_smaler(X_yfx, A, idx_S0, idx_S1, idx_y_fx,
                           i, m2):
    i_anchor = idx_y_fx[i]  # anchor's location after projection
    A_anchor = A[i_anchor]
    X_yfx_anchor = X_yfx[i_anchor]

    # Compute the distance d(anchor,\cdot) for at most m2 nearby
    # data points that meets a!=ai and g()<=g(xi,yi;w)
    j, num_j, min_js = i, 0, np.finfo(np.float32).max
    j = i - 1  # doesn't have to be compared with the anchor
    while num_j < m2:
        if j < 0:
            break

        idx_j = idx_y_fx[j]
        if set_belonging(idx_S0, idx_S1, i_anchor, idx_j):
            j -= 1
            continue

        curr = DistDirect_Euclidean(X_yfx_anchor, X_yfx[idx_j])
        if curr < min_js:
            min_js = curr

        num_j += 1
        j -= 1
    # Find the minimum among them, recorded as d_min^s
    # del A_anchor
    return min_js


@numba.jit(nopython=True)
def sub_accelerator_larger(X_yfx, A, idx_S0, idx_S1, idx_y_fx,
                           i, m2):
    i_anchor = idx_y_fx[i]  # anchor's location after projection
    A_anchor = A[i_anchor]
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
        if set_belonging(idx_S0, idx_S1, i_anchor, idx_j):
            j += 1
            continue

        curr = DistDirect_Euclidean(X_yfx_anchor, X_yfx[idx_j])
        if curr < min_jr:
            min_jr = curr

        num_j += 1
        j += 1
    # Find the minimum among them, recorded as d_min^r
    return min_jr


# ------------------------------------------
# Algorithm 2. AcceleDist


# Projection using \mathbf{w}
#
@numba.jit(nopython=True)
def projector(element, vec_w):
    return float(np.dot(element, vec_w))


@fantasy_timer
def AcceleDist_bin(X_and_yddot, A, idx_S0, idx_S1, m2, vec_w):
    # Project data points onto a one-dimensional space
    proj = [projector(ele, vec_w) for ele in X_and_yddot]
    idx_y_fx = np.argsort(proj)

    n = X_and_yddot.shape[0]  # number of instances
    d_min = []
    for i in range(n):
        # Set the anchor data point (xi,yi) in this round

        min_js = sub_accelerator_smaler(
            X_and_yddot, A, idx_S0, idx_S1, idx_y_fx, i, m2)
        min_jr = sub_accelerator_larger(
            X_and_yddot, A, idx_S0, idx_S1, idx_y_fx, i, m2)

        # finally,
        tmp = min(min_js, min_jr)
        d_min.append(tmp)
    # return max(d_min)
    return max(d_min), sum(d_min)


# ------------------------------------------
# Algorithm 1. ApproxDist


@numba.jit(nopython=True)
def weight_generator(n_d):
    vec_w = np.zeros(1 + n_d)
    tmp = 1.
    for i in range(1 + n_d):
        vec_w[i] = np.random.uniform(-tmp, tmp)
        tmp -= np.abs(vec_w[i])
    vec_w[n_d] = 1. - np.sum(np.abs(vec_w[: -1]))
    return vec_w


# """
# @fantasy_timer
# def ApproxDist_bin(X_and_yddot, A, idx_S1, m1, m2):
#     idx_S0 = ~idx_S1
#     n_d = X_and_yddot.shape[1]  # n,n_d= X_and_yddot.shape
#     d_max = []
#     for _ in range(m1):  # for k in
#         vec_w = weight_generator(n_d - 1)
#         tmp, _ = AcceleDist_bin(
#             X_and_yddot, A, idx_S0, idx_S1, m2, vec_w)
#         d_max.append(tmp[0])
#     return min(d_max)  # float


# @fantasy_timer
# def ApproxDist_bin_revised(X_and_yddot, A, idx_S1, m1, m2):
#     idx_S0 = ~idx_S1
#     n, n_d = X_and_yddot.shape
#     d_max, d_avg = [], []
#     for _ in range(m1):  # for k in
#         vec_w = weight_generator(n_d - 1)
#         tmp, _ = AcceleDist_bin(
#             X_and_yddot, A, idx_S0, idx_S1, m2, vec_w)
#         d_max.append(tmp[0])
#         d_avg.append(tmp[1])
#     # return min(d_max)  # float
#     return min(d_max), min(d_avg) / float(n)
# """


# @fantasy_timer
# def ApproxDist_bin(X_nA_y, A_j, non_sa, m1, m2):
@fantasy_timer
def ApproxDist_bin(X_nA_y, A_j, idx_S1, m1, m2):
    idx_S0 = ~idx_S1       # idx_sa = ~non_sa
    n_d = X_nA_y.shape[1]  # n,n_d= X_nA_y.shape
    d_max = []
    for _ in range(m1):    # for k in
        vec_w = weight_generator(n_d - 1)
        tmp, _ = AcceleDist_bin(
            # X_nA_y, A_j, idx_sa, non_sa, m2, vec_w)
            X_nA_y, A_j, idx_S0, idx_S1, m2, vec_w)
        d_max.append(tmp[0])
    return min(d_max)      # float


# @fantasy_timer
# def ApproxDist_bin_revised(X_nA_y, A_j, non_sa, m1, m2):
@fantasy_timer
def ApproxDist_bin_revised(X_nA_y, idx_S1, m1, m2):
    A_j = idx_S1.astype('int')  # non_sa.astype('int')
    idx_S0 = ~idx_S1       # idx_sa = ~non_sa
    n, n_d = X_nA_y.shape
    d_max, d_avg = [], []
    for _ in range(m1):    # for k in
        vec_w = weight_generator(n_d - 1)
        tmp, _ = AcceleDist_bin(
            # X_nA_y, A_j, idx_sa, non_sa, m2, vec_w)
            X_nA_y, A_j, idx_S0, idx_S1, m2, vec_w)
        d_max.append(tmp[0])
        d_avg.append(tmp[1])
    # return min(d_max)    # float
    return min(d_max), min(d_avg) / float(n)


# ------------------------------------------
#
# ------------------------------------------
# Alternative forms:
#   AcceleDist_bin, ApproxDist_bin


@numba.jit(nopython=True)
def subalt_accel_smaler(X_yfx, idx_S0, idx_S1, idx_y_fx,
                        i, m2, i_anchor, X_yfx_anchor):
    # Compute the distance d(anchor,\cdot) for at most m2 nearby
    # data points that meets a!=ai and g()<=g(xi,yi;w)
    j, num_j, tmp_list = i - 1, 0, []  # No comparison with anchor
    min_js = float(np.finfo(np.float32).max)  # or min_js_list
    while num_j < m2:
        if j < 0:
            break
        idx_j = idx_y_fx[j]
        if set_belonging(idx_S0, idx_S1, i_anchor, idx_j):
            j -= 1
            continue
        curr = DistDirect_Euclidean(X_yfx_anchor, X_yfx[idx_j])
        if curr < min_js:
            min_js = curr
            tmp_list.append(curr)    # min_js_list.append(curr)
        num_j += 1
        j -= 1
    # Find the minimum among them, recorded as d_min^s
    return min_js, tmp_list


@numba.jit(nopython=True)
def subalt_accel_larger(X_yfx, idx_S0, idx_S1, idx_y_fx,
                        i, m2, i_anchor, X_yfx_anchor):
    # Compute the distances d(anchor,\cdot) for at most m2 nearby
    # data points that meets a!=ai and g()>=g(xi,yi;w)
    j, num_j, tmp_list = i + 1, 0, []  # No comparison with anchor
    min_jr = float(np.finfo(np.float32).max)  # or min_jr_list
    n = len(X_yfx)
    while num_j < m2:
        if j >= n:
            break
        idx_j = idx_y_fx[j]
        if set_belonging(idx_S0, idx_S1, i_anchor, idx_j):
            j += 1
            continue
        curr = DistDirect_Euclidean(X_yfx_anchor, X_yfx[idx_j])
        if curr < min_jr:
            min_jr = curr
            tmp_list.append(min_jr)  # min_jr_list.append(min_jr)
        num_j += 1
        j += 1
    # Find the minimum among them, recorded as d_min^r
    return min_jr, tmp_list


@fantasy_timer
def AcceleDist_bin_alter(X_and_yddot, idx_S0, idx_S1, m2, vec_w):
    # Project data points onto a one-dimensional space
    proj = [projector(ele, vec_w) for ele in X_and_yddot]
    idx_y_fx = np.argsort(proj)
    n = X_and_yddot.shape[0]       # number of instances
    d_min = []
    for i in range(n):
        i_anchor = idx_y_fx[i]  # anchor's location aft projection
        X_yfx_anchor = X_and_yddot[i_anchor]
        # Set the anchor data point (xi,yi) in this round
        min_js, _ = subalt_accel_smaler(
            X_and_yddot, idx_S0, idx_S1, idx_y_fx, i, m2,
            i_anchor, X_yfx_anchor)
        min_jr, _ = subalt_accel_larger(
            X_and_yddot, idx_S0, idx_S1, idx_y_fx, i, m2,
            i_anchor, X_yfx_anchor)
        tmp = min(min_js, min_jr)  # finally,
        d_min.append(tmp)
    return max(d_min), sum(d_min)  # ,d_min  # return max(d_min)


@fantasy_timer
def ApproxDist_bin_alter(X_nA_y, Aj_nsa, m1, m2):
    idx_S0 = ~Aj_nsa       # Aj_nsa i.e. idx_S1
    n, n_d = X_nA_y.shape  # n_d = X_nA_y.shape[1]
    d_max, d_avg = [], []  # d_max = []
    for _ in range(m1):    # for k in
        vec_w = weight_generator(n_d - 1)
        tmp, _ = AcceleDist_bin_alter(
            X_nA_y, idx_S0, Aj_nsa, m2, vec_w)
        # tmp, _ = AcceleDist_bin(
        #     X_nA_y, Aj_nsa.astype('int'),
        #     idx_S0, Aj_nsa, m2, vec_w)
        d_max.append(tmp[0])
        d_avg.append(tmp[1])
    # return min(d_max)    # float
    return min(d_max), min(d_avg) / float(n)


# ------------------------------------------
#
