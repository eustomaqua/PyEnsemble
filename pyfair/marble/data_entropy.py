# coding: utf-8
#
# Target:
#   Some calculation of entropy (existing methods)
#


from copy import deepcopy
import gc

import numpy as np
from pathos import multiprocessing as pp

from pyfair.facil.utils_const import (
    DTY_FLT, DTY_INT, check_zero, random_seed_generator)
gc.enable()


# =======================================
#  previously on ``distributed.py''
# =======================================


#
# ----------- Convert data -----------
#
# Input : list
# Output: list, not np.ndarray
#


# minimum description length
#
def binsMDL(data, nb_bin=5):  # bins5MDL
    # Let `U' be a set of size `d' of labelled instances
    # accompanied by a large set of features `N' with
    # cardinality `n', represented in a `dxn' matrix.

    data = np.array(data, dtype=DTY_FLT)
    d = data.shape[0]  # number of samples
    n = data.shape[1]  # number of features

    for j in range(n):  # By Feature
        fmin = np.min(data[:, j])
        fmax = np.max(data[:, j])
        fgap = (fmax - fmin) / nb_bin
        trans = data[:, j]

        idx = (data[:, j] == fmin)
        trans[idx] = 0
        pleft = fmin
        pright = fmin + fgap

        for i in range(nb_bin):
            idx = ((data[:, j] > pleft) & (data[:, j] <= pright))
            trans[idx] = i
            pleft += fgap
            pright += fgap

        data[:, j] = deepcopy(trans)  # trans.copy()
        del fmin, fmax, fgap, trans, pleft, pright, idx
    data = np.array(data, dtype=DTY_INT)
    del d, n  # , i, j
    gc.collect()
    # return data.copy()  # np.ndarray
    return data.tolist()  # list


#
# ----------- Probability of Discrete Variable -----------
#


# probability of one vector
#
def prob(X):
    X = np.array(X)
    vX = np.unique(X).tolist()
    dX = len(vX)

    px = np.zeros(dX)
    for i in range(dX):
        px[i] = np.mean(X == vX[i])
    px = px.tolist()

    # BUG! no i if X == []
    # i = None
    del X, dX, i
    # gc.collect()
    return deepcopy(px), deepcopy(vX)  # list


# joint probability of two vectors
#
def jointProb(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    vX = np.unique(X).tolist()
    vY = np.unique(Y).tolist()
    dX = len(vX)
    dY = len(vY)

    pxy = np.zeros((dX, dY))
    for i in range(dX):
        for j in range(dY):
            pxy[i, j] = np.mean((X == vX[i]) & (Y == vY[j]))
    pxy = pxy.tolist()

    # i = j = None
    del dX, dY, X, Y, i, j
    # gc.collect()
    return deepcopy(pxy), deepcopy(vX), deepcopy(vY)  # list


#
# ----------- Shannon Entropy -----------
#
# calculate values of entropy
# H(.) is the entropy function and
# p(.,.) is the joint probability
#


# for a scalar value
#
def H(p):
    if p == 0.:
        return 0.
    # return (-1.) * p * np.log2(p)
    return -p * np.log2(p)


# H(X), H(Y): for one vector
#
def H1(X):
    px, _ = prob(X)
    ans = 0.
    for i in px:
        ans += H(i)
    # i = None
    del px, i
    return ans


# H(X, Y): for two vectors
#
def H2(X, Y):
    pxy, _, _ = jointProb(X, Y)
    ans = 0.
    for i in pxy:
        for j in i:
            ans += H(j)
    # i = j = None
    del pxy, i, j
    return ans


# ----------------------------------


# =======================================
#  Inspired by zadeh2017diversity
# =======================================


# I(.;.) is the mutual information function
# I(X; Y)
#
def I(X, Y):
    if (X == []) or (Y == []):
        return 0.0
    #   #
    px, _ = prob(X)
    py, _ = prob(Y)
    pxy, _, _ = jointProb(X, Y)
    dX, dY = len(px), len(py)
    ans = 0.
    for i in range(dX):
        for j in range(dY):
            if pxy[i][j] == 0.:
                ans += 0.
            else:
                ans += pxy[i][j] * np.log2(pxy[i][j] / px[i] / py[j])
            # ? 若 px[i]是0, pxy[i,..] 一定是0；反之则未必？
    # i = j = None
    del px, py, pxy, dX, dY, i, j
    return ans


# MI(X, Y):
# The normalized mutual information of two discrete
# random variables X and Y
#
def MI(X, Y):
    # return I(X, Y) / np.sqrt(H1(X) * H1(Y))
    tem = np.sqrt(H1(X) * H1(Y))
    return I(X, Y) / check_zero(tem)


# VI(X, Y):
# The normalized variation of information of two discrete
# random variables X and Y
#
def VI(X, Y):
    # return 1. - I(X, Y) / np.max([H2(X, Y), 1e-18])
    return 1. - I(X, Y) / check_zero(H2(X, Y))


# For two feature vectors like p and q, and the class label vector L,
# define DIST(p,q) as follows:
#
def DIST(X, Y, L, lam):  # lambda
    if X == Y:  # list
        return 0.
    tem = (MI(X, L) + MI(Y, L)) / 2.
    return lam * VI(X, Y) + (1. - lam) * tem


#
# S \subset or \subseteq N, N is the set of all features and |S|=k.
# We want to maximize the following objective function (as the
# objective of diversity maximization problem)
# for `S` \subset `N` and |S|=k


def DIV1(S, L, lam):
    S = np.array(S)
    k = S.shape[1]
    # '''
    # ans = 0.
    # for i in range(k):
    #     for j in range(k):
    #         ans += DIST(S[:, i].tolist(), S[:, j].tolist(), L, lam)
    # ans /= 2.
    # '''
    ans = [[DIST(S[:, i].tolist(), S[:, j].tolist(), L, lam)
            for j in range(k)] for i in range(k)]
    ans = np.sum(ans) / 2.
    del S, k
    return ans


def DIV2(S, L, lam):
    S = np.array(S)
    k = S.shape[1]
    # '''
    # ans1, ans2 = 0., 0.
    # for i in range(k):
    #     for j in range(k):
    #         ans1 += VI(S[:, i].tolist(), S[:, j].tolist())
    # ans1 *= lam/2.
    # for i in range(k):
    #     ans2 += MI(S[:, i].tolist(), L)
    # ans2 *= (1.-lam)*(k-1.)/2.
    # ans = ans1 + ans2
    # '''
    ans1 = [[VI(S[:, i].tolist(), S[:, j].tolist())
             for j in range(k)] for i in range(k)]
    ans1 = np.sum(ans1)
    ans2 = [MI(S[:, i].tolist(), L) for i in range(k)]
    ans2 = np.sum(ans2)
    ans = ans1 * lam / 2. + ans2 * (1. - lam) * (k - 1.) / 2.
    del S, k, ans1, ans2
    return ans


# ----------------------------------
#  DDisMI v.s. Greedy
# ----------------------------------


# ----------- Algorithm Greedy -----------
#
# T: set of points/features
# k: number of selected features
#


def _dist_sum(p, S, L, lam):
    S = np.array(S)
    n = S.shape[1]
    ans = 0.
    for i in range(n):
        ans += DIST(p, S[:, i].tolist(), L, lam)
    del S, n, i
    return ans


# T is the set of points/features;
# S = [True,False] represents this one is in S or not,
# and S is selected features.
#
def _arg_max_p(T, S, L, lam):
    T = np.array(T)
    S = np.array(S)

    all_q_in_S = T[:, S].tolist()
    idx_p_not_S = np.where(np.logical_not(S))[0]  # np.where(S == False)[0]
    if len(idx_p_not_S) == 0:
        del T, S, all_q_in_S, idx_p_not_S
        return -1  # idx = -1

    ans = [_dist_sum(T[:, i].tolist(), all_q_in_S, L, lam)
           for i in idx_p_not_S]
    idx_p = ans.index(np.max(ans))
    idx = idx_p_not_S[idx_p]

    del T, S, all_q_in_S, idx_p_not_S, idx_p, ans
    return idx  # np.int64


# T: set of points/features
# k: number of selected features
#
def Greedy(T, k, L, lam):
    T = np.array(T)
    n = T.shape[1]
    S = np.zeros(n, dtype='bool')  # np.bool
    p = np.random.randint(0, n)
    S[p] = True
    for _ in range(1, k):  # for i in range(1,k):
        idx = _arg_max_p(T, S, L, lam)
        if idx > -1:
            S[idx] = True  # #1
    S = S.tolist()
    del T, n, p
    return deepcopy(S)  # list


# ----------- Algorithm DDisMI -----------
#  previously on ``thin_entropy.py''
#  import distributed as db
#
# N: set of features (d samples, n features, dxn matrix)
# k: number of selected features
# m: number of machines
#


def _choose_proper_platform(nb, pr):
    m = int(np.round(np.sqrt(1. / pr)))
    k = np.max([int(np.round(nb * pr)), 1])
    while k * m >= nb:
        m = np.max([m - 1, 1])
        if m == 1:
            break
    # m = np.max([m, 2])
    return k, m


def _randomly_partition(n, m):
    _, prng = random_seed_generator('fixed_tseed')  # rndsed,
    # rndsed = renew_fixed_tseed()
    # prng = renew_random_seed(rndsed)

    tem = np.arange(n)
    prng.shuffle(tem)
    # idx = np.zeros(n, dtype=np.int8)-1  # init  # np.ones(n)-2
    idx = np.zeros(n, dtype=DTY_INT)  # initial

    if n % m != 0:
        # 底和顶 floors and ceilings
        floors = int(np.floor(n / float(m)))
        ceilings = int(np.ceil(n / float(m)))
        # 模：二元运算 modulus and mumble 含糊说话
        modulus = n - m * floors
        mumble = m * ceilings - n
        # mod: n % m

        for k in range(modulus):
            ij = tem[(k * ceilings): ((k + 1) * ceilings)]
            idx[ij] = k
        ijt = ceilings * modulus  # as follows:
        # range(modulus, modulus+mumble)  # m = modulus+mumble
        for k in range(mumble):
            ij = tem[(k * floors + ijt): ((k + 1) * floors + ijt)]
            idx[ij] = k + modulus

        del floors, ceilings, modulus, mumble, k, ij, ijt
    else:
        ijt = int(n / m)
        for k in range(m):
            ij = tem[(k * ijt): ((k + 1) * ijt)]
            idx[ij] = k
        del ijt, ij, k

    # idx = idx.tolist()
    # return deepcopy(idx)
    return idx.tolist()  # list


# Group/Machine i-th
#
def _find_idx_in_sub(i, Tl, N, k, L, lam):
    sub_idx_in_N = np.where(Tl == i)[0]  # or np.argwhere(Tl == i).T[0]
    sub_idx_greedy = Greedy(N[:, (Tl == i)].tolist(), k, L, lam)
    sub_idx_greedy = np.where(sub_idx_greedy)[0]  # or:
    # sub_idx_greedy np.where(np.array(sub_idx_greedy) == True)[0]
    ans = sub_idx_in_N[sub_idx_greedy]
    del sub_idx_in_N, sub_idx_greedy
    return deepcopy(ans)  # np.ndarray


# def DDisMI_MultiPool(N, k, m, L, lam):
def DDisMI(N, k, m, L, lam):
    N = np.array(N)
    n = N.shape[1]
    Tl = _randomly_partition(n = n, m = m)
    Tl = np.array(Tl)
    Sl = np.zeros(n, dtype=DTY_INT) - 1  # init

    # define lambda function
    # concurrent selection
    pool = pp.ProcessingPool(nodes = m)
    sub_idx = pool.map(_find_idx_in_sub, list(range(m)), [Tl] * m,
                       [N] * m, [k] * m, [L] * m, [lam] * m)
    del pool, Tl

    for i in range(m):
        Sl[sub_idx[i]] = i
    del sub_idx
    sub_all_in_N = np.where(Sl != -1)[0]
    sub_all_greedy = Greedy(N[:, (Sl != -1)].tolist(), k, L, lam)
    sub_all_greedy = np.where(sub_all_greedy)[0]  # or:
    # sub_all_greedy = np.where(np.array(sub_all_greedy) == True)

    final_S = np.zeros(n, dtype='bool')
    final_S[sub_all_in_N[sub_all_greedy]] = 1  # ?? check needed  # checked
    del sub_all_in_N, sub_all_greedy

    div_temS = DIV1(N[:, final_S].tolist(), L, lam)
    div_Sl = [DIV1(N[:, (Sl == i)].tolist(), L, lam) for i in range(m)]
    if np.sum(np.array(div_Sl) > div_temS) >= 1:
        tem_argmax_l = div_Sl.index(np.max(div_Sl))
        final_S = (Sl == tem_argmax_l)
        del tem_argmax_l

    del div_temS, div_Sl, N, n, m, Sl
    final_S = final_S.tolist()
    gc.collect()
    return deepcopy(final_S)


# ----------------------------------
#
# If you want to do ``Serial Execution'', just to do:
# S = Greedy(N, k, L, lam)
#


# =======================================
#  Inspired by margineantu1997pruning
# =======================================
#
# KL distance between two probablity distributions p and q:
# KL_distance = scipy.stats.entropy(p, q)
#
