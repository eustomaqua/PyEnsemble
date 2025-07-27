# coding: utf-8
#
# Reference:
# Taha AA, Hanbury A. An efficient algorithm for calculating the
# exact Hausdorff distance. IEEE transactions on pattern analysis
# and machine intelligence. 2015 Mar 3;37(11):2153-63.
#


import numpy as np
import numba
from pyfair.dr_hfm.dist_drt import DistDirect_Euclidean
from pyfair.facil.utils_timer import fantasy_timer


# ==========================================
# An efficient algorithm for calculating the exact Hausdorff distance


# Algorithm 1. NAIVEHDD
# Straightfowardly computes the directed Hausdorff distance

@numba.jit(nopython=True)
def NaiveHDD(A, B):
    cmax = 0
    for ele_x in A:
        cmin = float(np.finfo(np.float32).max)
        for ele_y in B:
            d = DistDirect_Euclidean(ele_x, ele_y)
            if d < cmin:
                cmin = d
        if cmin > cmax:
            cmax = cmin
    return cmax


# Early Breaking

# Algorithm 3. RANDOMIZE
# Finds a random order of a given point set

def HDD_randomize(S):
    m = len(S)
    ind = list(range(m))
    for p in range(m):
        q = np.random.choice(ind)
        if q == p:
            continue
        tmp = ind[p]
        ind[p] = ind[q]
        ind[q] = tmp
    return ind


# Algorithm 2. EARLYBREAK
# Computes the directed HDD using the Early Break technique and
# the Random Sampling

def HDD_earlybreak(A, B):
    cmax = 0
    Er_ind = HDD_randomize(A)
    Br_ind = HDD_randomize(B)
    Er = A[Er_ind]
    Br = B[Br_ind]

    for ele_x in Er:
        cmin = float(np.finfo(np.float32).max)
        for ele_y in Br:
            d = DistDirect_Euclidean(ele_x, ele_y)
            if d < cmin:
                cmin = d
            if d < cmax:
                break

        if cmin > cmax:
            cmax = cmin
    return cmax


# ==========================================
# NaiveHDD & EarlyBreak (aka. EffHDD)


@fantasy_timer
def EffHD_bin(X_nA_y, idx_Si):
    Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
    half_1 = HDD_earlybreak(Sj, Sj_c)
    half_2 = HDD_earlybreak(Sj_c, Sj)
    return max(half_1, half_2)


@fantasy_timer
@numba.jit(nopython=True)
def Naive_bin(X_nA_y, idx_Si):
    Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
    half_1 = NaiveHDD(Sj, Sj_c)
    half_2 = NaiveHDD(Sj_c, Sj)
    return max(half_1, half_2)


@fantasy_timer
def EffHD_nonbin(X_nA_y, idx_Sjs):
    cmax = 0
    for idx_Si in idx_Sjs:
        Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
        half_1 = HDD_earlybreak(Sj, Sj_c)
        if half_1 > cmax:
            cmax = half_1
    return cmax


@fantasy_timer
def EffHD_multivar(X_nA_y, idx_Ai_Sj):
    half_mid = [EffHD_nonbin(
        X_nA_y, idx_Sjs) for idx_Sjs in idx_Ai_Sj]
    half_mid, half_ut = zip(*half_mid)
    return max(half_mid), (half_mid, half_ut)


@fantasy_timer
# @numba.jit(nopython=True)  # @numba.njit
def Naive_nonbin(X_nA_y, idx_Sjs):
    cmax = 0
    for idx_Si in idx_Sjs:
        Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
        half_1 = NaiveHDD(Sj, Sj_c)
        if half_1 > cmax:
            cmax = half_1
    return cmax


@fantasy_timer
def Naive_multivar(X_nA_y, idx_Ai_Sj):
    half_mid = [Naive_nonbin(
        X_nA_y, idx_Sjs) for idx_Sjs in idx_Ai_Sj]
    half_mid, half_ut = zip(*half_mid)
    return max(half_mid), (half_mid, half_ut)
