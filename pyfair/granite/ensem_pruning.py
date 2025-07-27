# coding: utf-8
#
# Target:
#   Existing pruning methods in ensemble learning
#
# Including:
#   1) Ranking based / Ordering based
#   2) Clustering based
#   3) Optimization based
#   4) Other
#   5) Ameliorate of mine (my own version)
#   6) Composable Core-sets, Diversity Maximization
#


from copy import deepcopy
import gc
import sys
import time
import numpy as np
from pympler.asizeof import asizeof

from pyfair.facil.utils_const import (
    check_zero, DTY_FLT, DTY_INT, DTY_BOL, judge_transform_need,
    random_seed_generator)
from pyfair.facil.utils_remark import AVAILABLE_NAME_PRUNE

from pyfair.facil.ensem_voting import plurality_voting
# from pyfair.senior.ensem_diversity import (
#     kappa_statistic_multiclass, Kappa_Statistic_multi)
from pyfair.marble.diver_pairwise import (
    kappa_statistic_multiclass, Kappa_Statistic_multi)
gc.enable()


# ----------------------------------
#  General
# ----------------------------------
# return:
#   yo :  individual classifiers in the pruned sub-ensemble
#   P  :  indices of the pruned sub-ensemble
#   seq:  order / sequence / succession of pruned 顺序
#


# =========================================
# \citep{margineantu1997pruning}
#
# Pruning Adaptive Boosting (ICML-97)
# [multi-class classification, AdaBoost]
# =========================================


# ----------------------------------
# Early Stopping
#   works for [multi-class]
# ----------------------------------

def Early_Stopping(yt, nb_cls, nb_pru):
    yo = yt[: nb_pru]
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    P[: nb_pru] = True
    seq = np.arange(nb_pru)
    return yo, P.tolist(), seq.tolist()


# ----------------------------------
# KL-divergence Pruning
#   works for [multi-class]
# ----------------------------------
#
# KL_distance between two probability distribution p and q
# KL_distance = scipy.stats.entropy(p, q)

# data_entropy.py --- Inspired by margineantu1997pruning
#
# the KL distance between two probability distributions p and q is:
# D(p || q) =
#
# softmax regression
#

def _softmax(y):
    return np.exp(y) / np.sum(np.exp(y), axis=0)


def _KLD(p, q):
    # p = np.array(p, dtype=DTY_FLT)
    # q = np.array(q, dtype=DTY_FLT)
    if np.sum(p) != 1.0:
        # tem = np.sum(p)
        # p /= check_zero(tem)
        tem = check_zero(np.sum(p))
        p = [i / tem for i in p]
    if np.sum(q) != 1.0:
        # tem = np.sum(q)
        # q /= check_zero(tem)
        tem = check_zero(np.sum(q))
        q = [i / tem for i in q]
    #   #   #
    ans = 0.
    n = len(p)  # = len(q)
    for i in range(n):
        tem = p[i] / check_zero(q[i])
        tem = p[i] * np.log(check_zero(tem))
        # ?? tem = p[i] * np.log2(check_zero(tem))
        ans += tem
    return float(ans)


# X, Y = classification vectors
#
def _KLD_vectors(X, Y):
    vXY = np.unique(X + Y).tolist()
    dXY = len(vXY)
    # vXY = np.concatenate([X, Y]).tolist()
    # vXY, dXY = judge_need_transform_or_not(vXY)
    # if dXY == 1:
    #     dXY = 2
    # vXY = np.unique(vXY); dXY = len(vXY)
    #   #   #
    px = np.zeros(dXY, dtype=DTY_FLT)
    py = np.zeros(dXY, dtype=DTY_FLT)
    X, Y = np.array(X), np.array(Y)
    for i in range(dXY):
        px[i] = np.mean(X == vXY[i])  # np.equal(X, vXY[i])
        py[i] = np.mean(Y == vXY[i])  # np.equal(Y, vXY[i])
    px = px.tolist()
    py = py.tolist()
    del X, Y, dXY, vXY
    ans = _KLD(px, py)
    del px, py
    return ans


def _JU_set_of_vectors(U):
    ans = 0.
    u = len(U)
    for i in range(u - 1):
        for j in range(i + 1, u):
            ans += _KLD_vectors(U[i], U[j])
    del i, j, u
    return ans


def _U_next_idx(yt, P):
    # not_in_p = np.where(np.array(P) == False)[0]
    not_in_p = np.where(np.logical_not(P))[0]
    yt = np.array(yt)
    ansJ = []
    for i in not_in_p:
        ansP = deepcopy(P)  # P.copy()
        ansP[i] = True
        ansU = yt[ansP].tolist()  # yt[ansP == True]
        ansJ.append(_JU_set_of_vectors(ansU))
        del ansP, ansU
    idx = ansJ.index(np.max(ansJ))  # int
    # del ansJ, yt
    ans = not_in_p[idx]  # np.integer
    # return ans, ansJ  # obtain different id by np.integer
    return int(ans)


def KL_divergence_Pruning(yt, nb_cls, nb_pru):
    P = np.zeros(nb_cls, dtype=DTY_BOL).tolist()
    P[0] = True
    seq = [0]
    while np.sum(P) < nb_pru:
        idx = _U_next_idx(yt, P)
        P[idx] = True
        seq.append(idx)
        del idx
    yo = np.array(yt)[P].tolist()  # yo = yt[P]
    return yo, deepcopy(P), deepcopy(seq)  # list


# ----------------------------------
# KL-divergence Pruning
# Modification of mine (with softmax)
# ----------------------------------

# KL distance between two vectors X and Y:
#
# the most important difference between KLD_vectors(X, Y) and KLD_pq(X, Y) is that:
#     KLD_vectors(X, Y) uses unique values in X, Y, however,
#     KLD_pq(X, Y) uses softmax and therefore, even the previous same values in
#         "unique values" would be different as well.
#     That's why they are different at all
#

def _KLD_pq(X, Y):
    p = _softmax(X).tolist()
    q = _softmax(Y).tolist()
    return _KLD(p, q)


def _J(U):
    ans = 0.
    u = len(U)
    for i in range(u - 1):
        for j in range(i + 1, u):
            ans += _KLD_pq(U[i], U[j])
    return ans


def _KL_find_next(yt, P):
    # not_in_p = np.where(P == False)[0]
    not_in_p = np.where(np.logical_not(P))[0]
    yt = np.array(yt)
    ansJ = []
    for i in not_in_p:
        ansP = deepcopy(P)  # P.copy()
        ansP[i] = True
        ansU = yt[ansP].tolist()  # yt[ansP == True]
        ansJ.append(_J(ansU))
        del ansP, ansU
    idx = ansJ.index(np.max(ansJ))
    del ansJ, yt
    ans = not_in_p[idx]  # Notice the position of p
    return int(ans)


def KL_divergence_Pruning_modify(yt, nb_cls, nb_pru):
    P = [True] + [False for _ in range(nb_cls - 1)]
    seq = [0]  # [False] * (nb_cls - 1)
    while np.sum(P) < nb_pru:
        idx = _KL_find_next(yt, P)
        P[idx] = True
        seq.append(idx)
    yo = np.array(yt)[P].tolist()  # yt[P]
    return yo, deepcopy(P), deepcopy(seq)  # list


#
# Updated on Feb 2nd 2020:
#   Do not use softmax in KL-divergence Pruning.
#   Better not to do this.
#


# ----------------------------------
# Kappa Pruning [binary classification]
# ! not multi-class
# now works on multi-class
# ----------------------------------

# def Kappa(nb_y, nb_c, ha, hb):
#     dY = nb_c  # number of labels
#     m = nb_y  # number of instances / samples

# def KappaMulti(ha, hb, y):
#     ....__annotations__
#     return ans, theta1, theta2


# using `kappa_statistic_multiclass`
def Kappa_Pruning_kuncheva(y, yt, nb_cls, nb_pru):
    m = len(y)  # initial
    Kij = np.zeros(shape=(nb_cls, nb_cls), dtype=DTY_FLT)
    Kij += sys.maxsize  # sys.maxint  # Kij += np.inf
    for i in range(nb_cls - 1):
        for j in range(i + 1, nb_cls):
            Kij[i, j] = kappa_statistic_multiclass(yt[i], yt[j], y, m)
    # upper triangular / triangle matrix
    # the lowest $\kappa$
    P = np.zeros(nb_cls, dtype=DTY_BOL).tolist()
    seq = []
    nb_p = 0
    while nb_p < nb_pru:
        idx = np.where(Kij == np.min(Kij))
        try:
            row, col = idx[0][0], idx[1][0]  # np.integer
        except Exception as e:
            print("Kappa_Pruning --\n nb_p {}\n idx {}\n "
                  "Kij {}".format(nb_p, idx, Kij))
            raise e
        if nb_p + 1 == nb_pru:
            P[row] = True
            seq.append(int(row))
            Kij[row, :] = sys.maxsize
            Kij[:, row] = sys.maxsize
            nb_p += 1
        else:
            P[row] = True
            P[col] = True
            seq.extend([int(row), int(col)])
            Kij[row, :] = sys.maxsize
            Kij[:, col] = sys.maxsize
            Kij[:, row] = sys.maxsize
            Kij[col, :] = sys.maxsize
            nb_p += 2
        del idx, row, col
    yo = np.array(yt)[P].tolist()  # yt[P == True]
    return yo, deepcopy(P), deepcopy(seq)  # list


# using `Kappa_Statistic_multi`
def Kappa_Pruning_zhoubimu(y, yt, nb_cls, nb_pru):
    m = len(y)  # = float(len(y))  # number of instances / samples
    Kij = np.zeros(shape=(nb_cls, nb_cls), dtype=DTY_FLT)
    Kij += sys.maxsize  # sys.maxint  # Kij += np.inf
    for i in range(nb_cls - 1):
        for j in range(i + 1, nb_cls):
            ans, _, _ = Kappa_Statistic_multi(yt[i], yt[j], y, m)
            Kij[i, j] = ans
    # upper triangular / triangle matrix
    # the lowest \kappa
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    seq = []
    nb_p = 0
    while nb_p < nb_pru:
        idx = np.where(Kij == np.min(Kij))
        try:
            row, col = idx[0][0], idx[1][0]  # np.integer
        except Exception as e:
            print("Kappa_Pruning -- nb_cls {} nb_pru {} nb_p {}"
                  "".format(nb_cls, nb_pru, nb_p))
            print("Kappa_Pruning -- y, yt, {} \n{:23s}{}".format(y, '', yt))
            print("Kappa_Pruning -- idx {}".format(idx))
            print("Kappa_Pruning -- Kij \n{:20s}{}".format('', Kij))
            raise e
        else:
            pass
        finally:
            pass
        #   #   #
        if nb_p + 1 == nb_pru:
            P[row] = True
            seq.append(int(row))
            Kij[row, :] = sys.maxsize
            Kij[:, row] = sys.maxsize
            nb_p += 1
        else:
            P[row] = True
            P[col] = True
            seq.extend([int(row), int(col)])
            Kij[row, :] = sys.maxsize
            Kij[:, col] = sys.maxsize
            Kij[:, row] = sys.maxsize
            Kij[col, :] = sys.maxsize
            nb_p += 2
        del idx, row, col
    # yo = np.array(yt)[P]
    # return yo.tolist(), P.tolist(), deepcopy(seq)
    yo = np.array(yt)[P].tolist()
    return yo, P.tolist(), deepcopy(seq)  # list


# ----------------------------------
# Reduce-Error Pruning with Backfitting
# [multi-class classification]
# ----------------------------------


def Reduce_Error_Pruning_with_Backfitting():
    pass


# =========================================
# \citep{martine2006pruning}
#
# Pruning in Ordered Bagging Ensembles (ICML-06)
# [multi-class classification, Bagging]
# =========================================


# ----------------------------------
# Ordering Bagging Ensembles
# Orientation Ordering [mult-class classification]
# ----------------------------------


# a, b are vectors
#
def _angle(a, b):
    a, b = np.array(a), np.array(b)
    # dot product, scalar product
    prod = np.sum(a * b)  # $a \cdot b$  # np.ndarray
    # or: prod = np.dot(a, b)  # also works for list
    # norm / module
    len1 = np.sqrt(np.sum(a * a))  # $|a|, |b|$
    len2 = np.sqrt(np.sum(b * b))
    # $\cos(\theta)$ \in [-1,1]
    cos_theta = prod / check_zero(len1 * len2)
    if not (-1. <= cos_theta <= 1.):
        cos_theta = max(-1., cos_theta)
        cos_theta = min(cos_theta, 1.)
    # nan robust # RuntimeWarning: invalid value encountered in arccos
    theta = np.arccos(cos_theta)
    del a, b, prod, len1, len2, cos_theta
    return float(theta)


# $\mathbf{c}_t$, as the $N_{tr}$-dimensional vector
# the signature vector of the classifier $h_t$, for the dataset
# $L_{tr}$ composed of $N_{tr}$ examples
#
def _signature_vector(ht, y):
    y, ht = np.array(y), np.array(ht)
    ct = 2. * (y == ht) - 1.
    ans = ct.tolist()
    del y, ht, ct
    return deepcopy(ans)  # list


# $c_{ti}$ is equal to +1 if $h_t$ (the t-th unit in the ensemble)
# correctly classifies the i-th example of $L_{tr}$
# $\mathbf{c}_{ens}$, the average signature vector of the ensemble is
#
def _average_signature_vector(yt, y):
    ct = [_signature_vector(ht, y) for ht in yt]
    cens = np.mean(ct, axis=0)  # np.sum(..)/nb_cls
    return cens.tolist(), deepcopy(ct)  # list


# This study presents an ordering criterion based on the orientation
# of the signature vector of the individual classifiers with respect
# to a reference direction.
#
# This direction, coded in a reference vector, $\mathbf{c}_{ref}$, is
# the projection of the first quadrant diagonal onto the hyper-plane
# defined by $\mathbf{c}_{fens}$.
#
def _reference_vector(nb_y, cens):
    oc = np.ones(shape=nb_y)  # $\mathbf{o}$
    cens = np.array(cens)  # $\mathbf{c}_{ens}$
    lam = check_zero(np.sum(cens * cens))
    lam = -1. * np.sum(oc * cens) / lam  # $\lambda$
    cref = oc + lam * cens  # $\mathbf{c}_{ref}$
    # perpendicular: np.abs(np.sum(cref * cens) - 0.) < 1e-6 == True
    #   #   #
    # $\mathbf{c}_{ref}$ becomes unstable when the vectors that define
    # the projection (i.e., $\mathbf{c}_{ref}$ and the diagonal of the
    # first quadrant) are close to each other.
    flag = _angle(cref.tolist(), oc.tolist())
    ans = cref.tolist()
    del oc, cens, lam, cref
    return deepcopy(ans), flag


# The classifiers are ordered by increasing values of the angle between
# the signature vectors of the individual classifiers and the reference
# vector $\mathbf{c}_{ref}$
#
def Orientation_Ordering_Pruning(y, yt):
    nb_y = len(y)  # number of samples / instances
    cens, ct = _average_signature_vector(yt, y)
    cref, flag = _reference_vector(nb_y, cens)
    theta = [_angle(i, cref) for i in ct]
    P = np.array(theta) < (np.pi / 2.)
    P = P.tolist()
    seq = np.array(np.argsort(theta), dtype=DTY_INT).tolist()
    del nb_y, cens, ct, cref, theta
    #   #
    if np.sum(P) == 0:
        tem = np.random.randint(len(P))
        P[tem] = True
        seq = [tem]
    else:
        seq = seq[: np.sum(P)]
    yo = np.array(yt)[P].tolist()  # yt[P]
    return yo, deepcopy(P), deepcopy(seq), flag  # list


# ----------------------------------
# ----------------------------------


# =========================================
# \citep{martinez2009analysis}
#
# An Analysis of Ensemble Pruning Techniques Based on Ordered Aggregation
# (TPAMI) [multi-class classification, Bagging]
# =========================================


# ----------------------------------
# Reduce-Error Pruning
# ----------------------------------
#
# Note that:
#   need to use a pruning set, subdivided from training set,
#   with a sub-training set
#

def Reduce_Error_Pruning(y, yt, nb_cls, nb_pru):
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    yt = np.array(yt)
    #
    # first
    err = np.mean(np.not_equal(yt, y), axis=1)
    # or: err = np.mean(yt != y, axis=1)
    idx = err.argmin()  # argmax()
    P[idx] = True
    seq = [int(idx)]  # np.integer
    #
    # next
    while np.sum(P) < nb_pru:
        # find the next idx
        # not_in_p = np.where(P == False)[0]
        not_in_p = np.where(np.logical_not(P))[0]
        anserr = []
        for i in not_in_p:
            temP = P.copy()
            temP[i] = True
            temyt = yt[temP].tolist()
            temfens = plurality_voting(temyt)  # y,
            # temerr = np.mean(temfens != y, axis=0)
            temerr = np.mean(np.not_equal(temfens, y), axis=0)
            anserr.append(temerr)
            del temP, temyt, temfens, temerr
        #   #
        idx = anserr.index(np.min(anserr))  # int
        P[not_in_p[idx]] = True
        seq.append(int(not_in_p[idx]))
        del anserr, idx, not_in_p
    #   #   #
    yo = np.array(yt)[P].tolist()
    del yt, err
    return yo, P.tolist(), deepcopy(seq)  # list


# ----------------------------------
# Complementarity Measure
# ----------------------------------
# \citep{martinez2004aggregation, cao2018optimizing}
#
# at the u-th iteration,
#   s_u = \argmax_k \sum_{(x,y)\in D} \mathbb{I}(
#                           H_{S_{u-1}}(x) \neq y && y==h_k(x)
#         )
# where the index k in L_{u-1} and S_u = S_{u-1} \cup {s_u}
#

def Complementarity_Measure_Pruning(y, yt, nb_cls, nb_pru):
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    err = np.mean(np.not_equal(yt, y), axis=1)  # yt != y
    idx = err.argmin()
    P[idx] = True
    seq = [int(idx)]  # first
    yt = np.array(yt)
    # next
    while np.sum(P) < nb_pru:
        S_u = np.where(P)[0]  # pruned sub-   # current selected individuals
        L_u = np.where(np.logical_not(P))[0]  # current left classifiers in ensemble
        fens = plurality_voting(yt[S_u].tolist())  # y,
        hk_x = [np.logical_and(np.not_equal(fens, y),
                               np.equal(y, yt[k])) for k in L_u]
        # hk_x = [np.logical_and(fens != y, y == yt[k]) for ..]  # [[nb_y] len(L_u)]
        hk_x = [np.sum(k) for k in hk_x]
        k = np.argsort(hk_x)[-1]
        #   #   #
        idx = L_u[k]  # Notice here!!
        P[idx] = True  # P[k] = True
        seq.append(int(idx))  # seq.append(int(k))
    #   #   #
    yo = np.array(yt)[P].tolist()  # yt[P == True]
    return yo, P.tolist(), deepcopy(seq)  # list


# ----------------------------------
# Concurrency thinning
# ----------------------------------


# ----------------------------------
# Margin Distance Minimization
# ----------------------------------


# ----------------------------------
# Boosting-based Ordering [multi-class, Bagging]
# \citep{martinez2007using}
# Using Boosting to Prune Bagging Ensemble
# ----------------------------------


# =========================================
# \citep{tsoumakas2009ensemble}
#
# An Ensemble Pruning Primer
# =========================================


# ----------------------------------
# ----------------------------------


# =========================================
# \citep{}
# indyk2014composable, aghamolaei2015diversity, abbassi2013diversity
#
# Composable Core-sets for Diversity and Coverage Maximization
# Diversity Maximization via Composable Coresets
# Diversity Maximization Under Matroid Constraints
# =========================================

# def pruning_methods(name_func, para_func):
#     return name_func(*para_func)
#
# Remark:
#     specially for dt.DIST in DDisMI
#     using Kappa statistic: K, theta1, theta2 = Kappa(ha, hb, y)
#         K = 0, different; K = 1, completely the same
#
#     K = 1, if the two classifiers totally agree with each other
#               (completely the same);
#     K = 0, if the two classifiers agree by chance;
#     K < 0, is a rare case where the agreement is even less than
#               what is expected by chance.
#


# ----------------------------------
# GMM(S, k)
# ----------------------------------
#
# Input :  S , a set of points; k, size of the subset
# Output:  S', a subset of S of size k
# Operations:
# 1.    S' <-- An arbitrary point p
# 2.    for i = 2, ..., k do
# 3.        find p \in S\S' which maximize min_{q\in S'} dist(p, q)
# 4.        S' <-- S' \cup {p}
#

def _GMM_Kappa_sum(p, S, y):
    m = len(y)
    # ans = [Kappa(y, p, q) for q in S]
    ans = []
    for q in S:
        # tem, _, _ = Kappa_Statistic_multi(p, q, y, m)
        tem = kappa_statistic_multiclass(p, q, y, m)
        ans.append(tem)
    tem = np.sum(ans)  # np.float64  # isinstance(tem, float)
    return float(tem)


def GMM_Algorithm(y, yt, nb_cls, nb_pru):
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    rndsed, prng = random_seed_generator()

    idx = prng.randint(nb_cls)
    P[idx] = True
    seq = [idx]  # int
    #
    for _ in range(1, nb_pru):
        # find_max_p
        all_q_in_S = np.array(yt)[P].tolist()  # yt[P]
        # idx_p_not_S = np.where(P == False)[0]
        idx_p_not_S = np.where(np.logical_not(P))[0]
        if len(idx_p_not_S) == 0:
            idx = -1
        else:
            ans = [_GMM_Kappa_sum(yt[j], all_q_in_S, y) for j in idx_p_not_S]
            idx_p = ans.index(np.max(ans))
            idx = idx_p_not_S[idx_p]  # np.integer
            del ans, idx_p
        del all_q_in_S, idx_p_not_S
        # find_max_p
        if idx > -1:
            P[idx] = True
            seq.append(int(idx))
    #   #   #
    del rndsed, prng, idx
    yo = np.array(yt)[P].tolist()
    return yo, P.tolist(), deepcopy(seq)  # list


# ----------------------------------
# Local Search Algorithm
# ----------------------------------
#
# Input :  S , a set of points; k, size of the subset
# Output:  S', a subset of S of size k
# Operations:
# 1.    S' <-- An arbitrary set of k points which contains the two
#              farthest points
# 2.    while there exists p\in S\S' and p'\in S' such that ____ do
#           such that:  div(S'\{p'} \cup {p}) > div(S')(1+ \epsilon /k)
#               or      div(S'\{p'} \cup {p}) > div(S')(1+ \epsilon /n)
# 3.        S' <-- S' \ {p'} \cup {p}
#

def _LocalSearch_kappa_sum(S, y):
    n = len(S)
    Kij = np.zeros(shape=(n, n))
    m = len(y)  # number of instances
    for i in range(n - 1):
        for j in range(i + 1, n):
            # ans, _, _ = Kappa_Statistic_multi(S[i], S[j], y, m)
            ans = kappa_statistic_multiclass(S[i], S[j], y, m)
            Kij[i, j] = ans
    # upper triangle matrix
    ans = np.sum(Kij) / (n * (n - 1.) / 2.)
    del n, Kij
    return float(ans)


# """
# def _LCS_sub_get_index(nb_cls, nb_pru, row, col):
#     rndsed, prng = random_seed_generator()
#     idx2 = np.arange(nb_cls).tolist()
#     prng.shuffle(idx2)
#     idx3 = idx2[: nb_pru]
#     del rndsed, prng

#     if (row not in idx3) and (col not in idx3):
#         idx3 = idx3[: -2]
#     elif (row in idx3) and (col in idx3):
#         idx3.remove(row)
#         idx3.remove(col)
#     elif (row in idx3) and (col not in idx3):
#         idx3.remove(row)
#         idx3 = idx3[: -1]
#     elif (col in idx3) and (row not in idx3):
#         idx3.remove(col)
#         idx3 = idx3[: -1]
#     return idx3, idx2
# """


def _LCS_sub_get_index(nb_cls, nb_pru, row, col):
    _, prng = random_seed_generator()
    idx2 = list(range(nb_cls))
    prng.shuffle(idx2)
    idx3 = idx2[: nb_pru]
    del prng

    if (row in idx3) and (col in idx3):
        idx3.remove(row)
        idx3.remove(col)
    elif row in idx3:
        idx3.remove(row)
        idx3 = idx3[: -1]
    elif col in idx3:
        idx3.remove(col)
        idx3 = idx3[: -1]
    else:
        idx3 = idx3[: -2]
    return idx3, idx2


# def LCS_sub_choose_renew(y, yt, nb_pru, epsilon, S_within, S_without):
def _LCS_sub_idx_renew(y, yt, nb_pru, epsilon, S_within, S_without):
    flag = False  # whether exists (p, q)?
    div_b4 = _LocalSearch_kappa_sum(yt[S_within].tolist(), y)
    for p in S_within:
        idx_p = S_within.index(p)
        for q in S_without:
            tem_q = deepcopy(S_within)
            tem_q[idx_p] = q
            div_af = _LocalSearch_kappa_sum(yt[tem_q].tolist(), y)
            # if div_af > div_b4 * (1. + epsilon / nb_pru):
            #     pass
            # if div_af > div_b4 * (1. + epsilon / nb_cls):
            #     pass
            #   #
            if div_af > div_b4 * (1. + epsilon / nb_pru):
                flag = True
                S_within = deepcopy(tem_q)
                # seqseq.remove(p)
                # seqseq.append(q)  # int
                tem_p = deepcopy(S_without)
                idx_q = S_without.index(q)
                tem_p[idx_q] = p
                S_without = deepcopy(tem_p)
                del tem_p, idx_q
                break
        if flag:
            break
        # could use these two /\ or not, better not, but will take longer time
    return p, q, S_within, S_without, flag


def Local_Search(y, yt, nb_cls, nb_pru, epsilon):
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    m = len(y)  # number of instances
    #   #
    # an abritrary set of k points which contains the two
    # farthest points
    Kij = np.zeros(shape=(nb_cls, nb_cls))
    Kij += sys.maxsize
    for i in range(nb_cls - 1):
        for j in range(i + 1, nb_cls):
            # ans, _, _ = Kappa_Statistic_multi(yt[i], yt[j], y, m)
            ans = kappa_statistic_multiclass(yt[i], yt[j], y, m)
            Kij[i, j] = ans
    # upper triangular matrix
    idx1 = np.where(Kij == np.min(Kij))
    row = idx1[0][0]
    col = idx1[1][0]
    P[row] = True
    P[col] = True
    seq = [int(row), int(col)]  # np.integer
    del Kij, i, j, ans, idx1  # , row, col
    #   #
    # an abritrary set of k points which contains the two
    # farthest points
    idx4, _ = _LCS_sub_get_index(nb_cls, nb_pru, row, col)
    for i in idx4:
        P[i] = True
        if i not in seq:
            seq.append(int(i))  # np.integer
        # otherwise np.alen(seq) might >= np.sum(P)
    # del rndsed, prng, idx2, idx3, idx4, row, col
    #   #
    # while there exists p\in S\S'
    nb_cnt = np.sum(P) * (nb_cls - np.sum(P))
    yt = np.array(yt)
    S_within = np.where(P)[0].tolist()  # np.where(P == True)[0]..
    S_without = np.where(np.logical_not(P))[0].tolist()  # P==False
    seqseq = np.array(seq, dtype=DTY_INT).tolist()
    while nb_cnt >= 0:  # nb_count
        p, q, S_within, S_without, flag = _LCS_sub_idx_renew(
            y, yt, nb_pru, epsilon, S_within, S_without)
        if flag:
            seqseq.remove(p)
            seqseq.append(q)
        if not flag:
            break
        nb_cnt -= 1
    # end while
    del nb_cnt, S_without
    PP = np.zeros(nb_cls, dtype=DTY_BOL)
    PP[S_within] = True
    yo = np.array(yt)[PP].tolist()
    del S_within
    return yo, PP.tolist(), deepcopy(seqseq)


# ----------------------------------
# ----------------------------------


# =========================================
# \citep{li2012diversity}
#
# Diversity Regularized Ensemble Pruning
# =========================================
#
# H = \{ h_i(\mathbf{x}) \}, i=1,\dots,n,
#                          h_i:\mathcal{X}\mapsto\{-1,+1\}
# S = \{ (\mathbf{x}_k, y_k) \}, k=1,\dots,m
#
# Note that: y\in {-1,+1}, transform!
#


# ----------------------------------
# DREP
# [binary classification] only
# ----------------------------------
#
# Algorithm 1 The DREP method
# Input : ensemble to be pruned $H = \{h_i(\bm{x})\}_{i=1}^n$,
#         validation data set   $S = \{(\bm{x}_i, y_i)\}_{i=1}^m$,
#         and tradeoff parameter $\rho \in (0, 1)$
# Output: pruned ensemble $H^*$
#
#  1.   initialize H^* <-- \emptyset
#  2.   h(x) <-- the classifier in H with the lowest error on S
#  3.   H^* <-- \{h(x)\} and H <-- H\{h(x)}
#  4.   repeat
#  5.       for each h'(x) in H do
#  6.           compute d_{h'} <-- \diff(h', H^*) based on (9)
#  7.       end for
#  8.       sort classifiers h'(x) 's in H in the ascending order of d_{h'} 's
#  9.       \Gamma <-- the first ceil(\rho \cdot |H|) classifiers in the sorted list
# 10.       h(x) <-- the classifier in \Gamma which most reduces the error of H^* on S
# 11.       H^* <-- {h(x)} and H <-- H \ {h(x)}
# 12.   until the error of H^* on S cannot be reduced
#
# Correct: 11.  H^* <-- H^* \cup {h(x)} and ....
#


# $f(\mathbf{x};H) = \frac{1}{n} \sum_{1\leqslant i\leqslant n}
#                       h_i(\mathbf{x})$
#
def _DREP_fxH(yt):
    yt = np.array(yt)
    if yt.ndim == 1:
        yt = np.array([yt.tolist()])
    fens = np.mean(yt, axis=0)
    del yt

    fens = np.sign(fens)
    rndsed, prng = random_seed_generator()

    tie = [np.sum(fens == i) for i in [0, 1, -1]]
    if tie[1] > tie[2]:
        fens[fens == 0] = 1
    elif tie[1] < tie[2]:
        fens[fens == 0] = -1
    else:
        fens[fens == 0] = prng.randint(2) * 2 - 1

    del rndsed, prng, tie
    return fens.tolist()  # list


# $\diff(h_i, h_j) = \frac{1}{m} \sum_{1\leqslant i\leqslant m}
#                       h_i(\mathbf{x}_k) h_j(\mathbf{x}_k)$
#
def _DREP_diff(hi, hj):
    ans = np.mean(np.array(hi) * np.array(hj))
    # ans = np.mean(hi * hj)
    return float(ans)


def _DREP_sub_find_idx(tr_y, tr_yt, rho, P):
    # 5. for each h'(x) in H do
    # 6.     compute d_h' <-- \diff(h', H*) based on Eq.(9)
    # 7. end for
    hstar = np.array(tr_yt)[P].tolist()
    hstar = _DREP_fxH(hstar)
    all_q_in_S = np.where(np.logical_not(P))[0]  # P==False
    dhstar = [_DREP_diff(tr_yt[q], hstar) for q in all_q_in_S]
    #
    # 8. sort classifiers h'(x) 's in H in the ascending order
    #    of d_h' 's
    # 9. \Gamma <-- the first ceil(\rho * |H|) classifiers in
    #    the sorted list
    dhidx = np.argsort(dhstar).tolist()  # sort in ascending order
    tradeoff = int(np.ceil(rho * len(all_q_in_S)))
    gamma = dhidx[: tradeoff]  # index in Gamma
    gamma = [all_q_in_S[q] for q in gamma]
    #
    # 10. h(x) <-- the classifier in \Gamma which most reduces
    #     the error of H* on the training set
    errHstar = np.mean(np.array(hstar) != tr_y)
    ## errHstar = np.mean(np.not_equal(hstar, tr_y))
    idx = np.where(P)[0].tolist()  # P==True
    # errNew = [np.mean(DREP_fxH(tr_yt[idx + [p]]) != tr_y) for p in gamma]
    tr_yt = np.array(tr_yt)
    errNew = [_DREP_fxH(tr_yt[idx + [p]].tolist()) for p in gamma]
    errNew = [np.mean(np.not_equal(p, tr_y)) for p in errNew]
    errIdx = errNew.index(np.min(errNew))
    #
    # 11. H* <-- {h(x)} and H <-- H\{h(x)}
    if errNew[errIdx] <= errHstar:
        # P[gamma[errIdx]] = True
        # seq.append(gamma[errIdx])  # np.integer type
        flag = True
        return gamma[errIdx], flag
    # else:
    #     flag = False
    #   #
    del hstar, all_q_in_S, dhstar, dhidx
    del tr_yt, errHstar, errNew, tradeoff
    return -1, False


# $\rho \in (0, 1)$
#
def DREP_Pruning(y, yt, nb_cls, rho):
    vY = np.concatenate([[y], yt]).reshape(-1).tolist()
    vY, dY = judge_transform_need(vY)
    # '''
    # if dY > 2:
    #     raise UserWarning("DREP only works for binary classification."
    #                       " Check np.unique(y) please.")
    # elif dY == 2:
    #     # pass
    #     tr_y = [i * 2 - 1 for i in y]
    #     tr_yt = (np.array(yt) * 2 - 1).tolist()
    # elif dY == 1 and len(vY) == 2:
    #     ## y = [(i + 1) // 2 for i in y]
    #     ## yt = np.array((np.array(yt) + 1) / 2, dtype=DTY_INT).tolist()
    #     tr_y, tr_yt = y, yt
    # #   #   #
    # '''
    if dY > 2:
        raise UserWarning(
            "DREP only works for binary classification."
            " Check np.unique(y) please.")
    tr_y, tr_yt = y, yt
    if dY == 2:
        tr_y = [i * 2 - 1 for i in y]
        tr_yt = (np.array(yt) * 2 - 1).tolist()

    # 1. initialize H* <-- \emptyset
    # 2. h(x) <-- the classifier in H(original ensemble) with the lowest
    #    error on the training set
    # 3. H* <-- {h(x)} and H <-- H\{h(x)}
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    ## accpl = [np.mean(h != y) for h in yt]
    accpl = [np.mean(np.not_equal(h, y)) for h in yt]
    idx = accpl.index(np.min(accpl))
    P[idx] = True
    seq = [idx]  # int type
    # 4. repeat
    #   #
    flag = True  # whether the error of H* on S can be reduced
    nb_cnt = int(np.ceil(rho * nb_cls))  # nb_count
    ## tr_y = np.array(y) * 2 - 1
    ## tr_yt = np.array(yt) * 2 - 1
    while nb_cnt > 0:  # >=
        errIdx, flag = _DREP_sub_find_idx(tr_y, tr_yt, rho, P)
        if flag:
            P[errIdx] = True
            seq.append(int(errIdx))  # np.integer type
        #   #
        nb_cnt -= 1
        if not flag:  # flag==False
            break
    # 12. until the error of H* on the training set cannot be reduced
    #   #
    yo = np.array(yt)[P].tolist()
    del idx, flag, accpl, tr_y, tr_yt
    return yo, P.tolist(), deepcopy(seq)  # list


# ----------------------------------
# DREP modify
# to make it work for [multi-class]
# ----------------------------------


def _drep_multi_modify_diff(ha, hb):
    ans = np.equal(ha, hb) * 2 - 1
    ans = np.mean(ans)
    return float(ans)


def _drep_multi_modify_findidx(y, yt, rho, P):
    hstar = np.array(yt)[P].tolist()
    hstar = plurality_voting(hstar)  # y,
    all_q_in_S = np.where(np.logical_not(P))[0]
    dhstar = [_drep_multi_modify_diff(yt[q],
                                      hstar) for q in all_q_in_S]
    dhidx = np.argsort(dhstar).tolist()  # ascending order sort
    tradeoff = int(np.ceil(rho * len(all_q_in_S)))
    gamma = dhidx[: tradeoff]  # index in Gamma
    gamma = [all_q_in_S[q] for q in gamma]
    errHstar = np.mean(np.not_equal(hstar, y))
    idx = np.where(P)[0].tolist()
    yt = np.array(yt)
    # errNew = [plurality_voting(y, yt[idx + [p]].tolist()) for p in gamma]
    errNew = [plurality_voting(yt[idx + [p]].tolist()) for p in gamma]
    errNew = [np.mean(np.not_equal(p, y)) for p in errNew]
    errIdx = errNew.index(np.min(errNew))
    if errNew[errIdx] <= errHstar:
        return gamma[errIdx], True
    del hstar, all_q_in_S, dhstar, dhidx, tradeoff, gamma
    del errHstar, idx, yt, errNew, errIdx
    return -1, False


def drep_multi_modify_pruning(y, yt, nb_cls, rho):
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    accpl = [np.mean(np.not_equal(h, y)) for h in yt]
    idx = accpl.index(np.min(accpl))  # the lowest error on S
    P[idx] = True
    seq = [idx]  # int type
    # repeat
    #   #
    nb_cnt = int(np.ceil(rho * nb_cls))
    flag = True
    while nb_cnt > 0:
        errIdx, flag = _drep_multi_modify_findidx(y, yt, rho, P)
        if flag:
            P[errIdx] = True
            seq.append(int(errIdx))
        nb_cnt -= 1
        if not flag:
            break
        # 12. until the error of H* on the training set cannot
        #     be reduced
    # end while
    yo = np.array(yt)[P].tolist()
    del idx, accpl, flag
    return yo, P.tolist(), deepcopy(seq)  # list


# ----------------------------------
# ----------------------------------


# =========================================
# \citep{qian2015pareto}
#
# Pareto Ensemble Pruning (PEP)  (AAAI-15)
# [binary/multiclassification, Bagging]
# =========================================


# ----------------------------------
# assume
# ----------------------------------
#
# $D = \{ (\mathbf{x}_i, y_i) \}_{i=1}^m$   data set, with $m$ samples
# $H = \{ h_j \}_{j=1}^n$                   set of trained individual
#                                           classifiers, with $n$ ones
# $H_{\mathbf{s}}$ with $\mathbf{s}=\{0,1\}^n$  pruned ensemble, with a
#                                               selector vector
#     $|\mathbf{s}| = \sum_{j=1}^n s_j$         minimize the size of
#                                               $H_{\mathbf{s}}$
#
# ${\arg\min}_{ \mathbf{s}\in\{0,1\}^n } \bigg(
#           f(H_{\mathbf{s}}), |\mathbf{s}|
#     \bigg)$                   bi-objective ensemble pruning problem
# Note that: y\in {-1,+1},  transform!
#


# a pruned ensemble $H_{\mathbf{s}}$ is composited as
# $H_\mathbf{s}(\bm{x}) = \argmax_{y\in\mathcal{Y}} \sum_{i=1}^n
#                             s_i\cdot \mathbb{I}(h_i(\bm{x}) == y)$
#
# multiclassification
#
def _PEP_Hs_x(y, yt, s):
    vY = np.unique(np.vstack([y, yt])).tolist()
    s = np.transpose([s])  # np.array([s]).T  # np.array(np.mat(s).T)
    #   #
    # method 1:
    vote = [np.sum(s * np.equal(yt, i), axis=0).tolist() for i in vY]
    # method 2:
    # yt = yt[np.squeeze(s)]
    # vote = [np.sum(yt == i, axis=0).tolist() for i in vY]
    #   #
    loca = np.array(vote).argmax(axis=0)
    Hsx = [vY[i] for i in loca]
    del vY, s, vote, loca
    return deepcopy(Hsx)


# define the difference of two classifiers as
# [binary only]
#
def _PEP_diff_hihj(hi, hj):
    hi = np.array(hi)
    hj = np.array(hj)
    ans = np.mean((1. - hi * hj) / 2.)
    return float(ans)


# and the error of one classifier as
# [binary only]
#
def _PEP_err_hi(y, hi):
    y = np.array(y)
    hi = np.array(hi)
    ans = np.mean((1. - hi * y) / 2.)
    return float(ans)


# both of them (i.e., \diff(hi, hj), \err(hi)) belong to [0, 1].
# If \diff(hi, hj) = 0 (or 1), hi and hj always make the same
# (or opposite) predictions;
# if \err(hi) = 0 (or 1), hi always make the right (or wrong)
# prediction.
# We hope: (1) \diff larger, (2) \err smaller
#


# the validation error is calculated as
# binary [I think multiclassification works as well]
#
def _PEP_f_Hs(y, yt, s):
    Hsx = _PEP_Hs_x(y, yt, s)
    # ans = np.mean(Hsx != y)
    ans = np.mean(np.not_equal(Hsx, y))
    return float(ans), Hsx


# modify to multi-class
# ----------------------------------
# def PEP_diff_hihj()
# def PEP_err_hi()


def _pep_multi_modify_diff_hihj(ha, hb):
    ans = np.equal(ha, hb) * 2 - 1
    ans = np.mean((1. - ans) / 2.)
    return float(ans)


def _pep_multi_modify_err_hi(y, hc):
    ans = np.equal(hc, y) * 2 - 1
    ans = np.mean((1. - ans) / 2.)
    return float(ans)


# ----------------------------------
# performance objective $f$
# ----------------------------------


# def PEP_objective_performance():
#     pass


# ----------------------------------
# evaluation criterion $$eval$
# ----------------------------------


# ----------------------------------
# OEP, SEP
# ----------------------------------


# generate $\mathbf{s'}$ by flipping each bit of $\mathbf{s}$
# with prob.$\frac{1}{n}$
#
def _PEP_flipping_uniformly(s):
    n = len(s)
    pr = np.random.uniform(size=n)  # \in [0, 1]
    pr = (pr < 1. / n)  # <=
    # s', sprime
    sp = [1 - s[i] if pr[i] else s[i] for i in range(n)]
    del n, pr
    return deepcopy(sp)  # list


# Simple Ensemble Pruning
# ----------------------------------
#
# Algorithm 4 (SEP).
# Given a set of trained classifiers H = {h_i}_{i=1}^n and an
# objective f: 2^H -> \mathbb{R}, it contains:
#
# 1. \mathbf{s} = randomly selected from \{0, 1\}^n.
# 2. Repeated until the termination condition is met:
# 3.    Generate \mathbf{s}' by flippling each bit of \mathbf{s}
#       with prob.\frac{1}{n}.
# 4.    if $f(H_{\mathbf{s}'}) \leqslant f(H_\mathbf{s})$ then
#       \mathbf{s} = \mathbf{s}'.
# 5. Output \mathbf{s}.
#

def PEP_SEP(y, yt, nb_cls, rho):
    # 1. \mathbf{s} = randomly selected from \{0, 1\}^n.
    tem_s = np.random.uniform(size=nb_cls)  # \in [0, 1)
    tem_i = tem_s <= rho
    if np.sum(tem_i) == 0:
        tem_i[np.random.randint(nb_cls)] = True
    # 1. \mathbf{s} = randomly selected from \{0, 1\}^n.
    s = np.zeros(nb_cls, dtype=DTY_INT)
    s[tem_i] = 1
    s = s.tolist()
    del tem_s, tem_i
    # what if: 万一 tem_i = [] 呢？   solved
    #
    # 2. repeat until: the termination condition is met.
    nb_pru = int(np.ceil(rho * nb_cls))
    nb_cnt = nb_pru  # counter
    while nb_cnt >= 0:
        # 3. Generate \mathbf{s}' by flipping each bit of
        #    \mathbf{s} with prob.\frac{1}{n}.
        sp = _PEP_flipping_uniformly(s)
        f1, _ = _PEP_f_Hs(y, yt, sp)
        f2, _ = _PEP_f_Hs(y, yt, s)
        # 4. if f(H_{\mathbf{s}'}) <= f(H_\mathbf{s}) then
        #    \mathbf{s} = \mathbf{s}'.
        if f1 <= f2:
            s = deepcopy(sp)  # sp.copy()
        nb_cnt = nb_cnt - 1
        del sp, f1, f2
        if np.sum(s) > nb_pru:
            break
    #   #   #
    if np.sum(s) == 0:
        s[np.random.randint(nb_cls)] = 1
    # /\ this is for robust
    #   #
    yo = np.array(yt)[np.array(s) == 1].tolist()  # yt[s==1]
    # 5. output: \mathbf{s}.
    P = np.array(s, dtype=DTY_BOL)
    seq = np.where(np.array(s) == 1)[0]
    del nb_cnt, s
    return yo, P.tolist(), seq.tolist()  # list


# Ordering-based Ensemble Pruning
# ----------------------------------
#
# Algorithm 3 (OEP).
# Given trained classifiers $H = \{h_i\}_{i=1}^n$, an objective
# $f: 2^H -> \mathbb{R}$ and a criterion $eval$, it contains:
#
# 1. Let H^S = \emptyset, H^U = {h_1, h_2, ..., h_n}.
# 2. Repeat until H^U = \emptyset:
# 3.     h^* = \argmin_{h\in H^U} f(H^S \cup {h}).
# 4.     H^S = H^S \cup {h^*}, H^U = H^U - {h^*}.
# 5. Let H^S = {h_1^*, ..., h_n^*}, where h_i^* is the classifier
#    added in the i-th iteration.
# 6. Let k = \argmin_{1\leqslant i\leqslant n} eval({h_1^*, ..., h_i^*}).
# 7. Output {h_1^*, h_2^*, ..., h_k^*}.
#

def PEP_OEP(y, yt, nb_cls):
    # 1. Let H^S = \emptyset, H^U = {h_1, h_2, ..., h_n}.
    Hs = np.zeros(nb_cls, dtype=DTY_BOL).tolist()
    ordered_idx = []
    # 2. repeat until H^U = \emptyset:
    while np.sum(Hs) < nb_cls:
        # 3. h^* = \argmin_{h\in H^U} f(H^S \cup {h}).
        Hu_idx = np.where(np.logical_not(Hs))[0]  # Hs==False
        Hu_idx = Hu_idx.tolist()
        obj_f = []
        for h in Hu_idx:
            # obtain H^S \cup {h}
            tem_s = deepcopy(Hs)  # Hs.copy()
            tem_s[h] = True
            # compute f(H^S \cup {h})
            tem_ans, _ = _PEP_f_Hs(y, yt, tem_s)
            obj_f.append(tem_ans)
            del tem_s, tem_ans
        # find the index of the minimal value
        idx_f = obj_f.index(np.min(obj_f))
        idx_f = Hu_idx[idx_f]
        # ordered_idx is the equivalent of `seq`
        # 4. H^S = H^S \cup {h^*}, H^U = H^U - {h^*}.
        ordered_idx.append(idx_f)
        Hs[idx_f] = True
        del Hu_idx, obj_f, idx_f
    #   #
    # 5. Let H^S = {h_1^*, ..., h_n^*}, where h_i^* is the
    #    classifier added in the i-th iteration.
    del Hs
    # 6. Let k = \argmin_{1\leq i\leq n} eval({h_1^*, ..., h_i^*}).
    obj_eval = []
    for h in range(1, nb_cls + 1):
        tem_s = np.zeros(nb_cls, dtype=DTY_INT)
        tem_s[ordered_idx[: h]] = 1
        tem_ans, _ = _PEP_f_Hs(y, yt, tem_s.tolist())
        obj_eval.append(tem_ans)
        del tem_s, tem_ans
    # 6. Let k = \argmin_{1\leq i\leq n} eval({h_1^*, ..., h_i^*}).
    idx_k = obj_eval.index(np.min(obj_eval))
    # (2) idx_k = np.argsort(obj_eval)[0]
    # (3) idx_k = np.where(np.array(obj_eval) == np.min(obj_eval))[0]
    # (3) idx_k = idx_k[-1]  # no!  we want the size smaller!
    #
    # 7. output {h_1^*, h_2^*, ..., h_k^*}.
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    P[ordered_idx[: (idx_k + 1)]] = True  # Notice!!
    seq = ordered_idx[: (idx_k + 1)]  # np.integer
    del obj_eval, idx_k
    yo = np.array(yt)[P].tolist()
    return yo, P.tolist(), deepcopy(seq)  # list


# ----------------------------------
# Domination
# ----------------------------------


# bi-objective
#
def _PEP_bi_objective(y, yt, s):
    fHs, Hsx = _PEP_f_Hs(y, yt, s)
    s_ab = np.sum(s)  # absolute  # np.integer type
    ans = (fHs, int(s_ab))
    del fHs, Hsx, s_ab
    return deepcopy(ans)


# the objective vector:
# $\mathbf{g}(\mathbf{s}) = (g_1, g_2)$
# $\mathbf{g}: \mathcal{S} to \mathbb{R}^2$
#
# for two solutions
# $\mathbf{s}, \mathbf{s'} \in \mathcal{S}$
#


# (1) s weakly dominates s' if
#           g1(s) <= g1(s') and g2(s) <= g2(s')
#
def _PEP_weakly_dominate(g_s1, g_s2):
    if (g_s1[0] <= g_s2[0]) and (g_s1[1] <= g_s2[1]):
        return True
    return False


# (2) s dominates s' if
#           s \succeq_{g} s' and
#           either g1(s) < g1(s') or g2(s) < g2(s')
#
def _PEP_dominate(g_s1, g_s2):
    if _PEP_weakly_dominate(g_s1, g_s2):
        if g_s1[0] < g_s2[0]:
            return True
        elif g_s1[1] < g_s2[1]:
            return True
    return False


# ----------------------------------
# VDS, PEP
# ----------------------------------


# VDS Subroutine
# ----------------------------------
#
# Algorithm 2 (VDS Subroutine).
# Given a pseudo-Boolean function f and a solution \mathbf{s},
# it contains:
#
# 1. Q = \emptyset, L = \emptyset.
# 2. Let N(\cdot) denote the set of neighbor solutions of a
# binary vector with Hamming distance 1.
# 3. While V_\mathbf{s} = {
#                   \bm{y} \in N(\mathbf{s}) |
#                   (y_i \neq s_i \Rightarrow i \notin L)
#               } \neq \emptyset
# 4.     Choose \bm{y} \in V_\mathbf{s} with the minimal f value.
# 5.     Q = Q \cup {\bm{y}}.
# 6.     L = L \cup {i | y_i \neq s_i}.
# 7.     \mathbf{s} = \bm{y}.
# 8. Output Q.
#

def _PEP_VDS(y, yt, nb_cls, s):
    QL = np.zeros(nb_cls, dtype=DTY_BOL)
    sp = deepcopy(s)  # s.copy()  # $\mathbf{s}$
    # 1. Q = \emptyset, L = \emptyset.
    Q, L = [], []
    # 2. Let N(\cdot) denote the set of neighbor solutions of a
    #    binary vector with Hamming distance 1.
    # 3. While V_mathbf{s} = {
    #               \bm{y} \in N(\mathbf{s}) |
    #               y_i \neq s_i \Rightarrow i \notin L
    #          } \neq \emptyset
    while np.sum(QL) < nb_cls:
        # 2. Let N(\cdot) denote the set of neighbor solution with
        #    Hamming distance 1.
        Ns = [deepcopy(sp) for i in range(nb_cls)]
        # Ns = [sp.tolist() for i in range(nb_cls)]
        for i in range(nb_cls):
            Ns[i][i] = 1 - Ns[i][i]
        Ns = np.array(Ns)
        #
        # \bm{y} in N(\mathbf{s})
        idx_Vs = np.where(np.logical_not(QL))[0]  # QL==False
        Vs = Ns[idx_Vs].tolist()
        #
        # 4. Choose \bm{y} \in V_\mathbf{s} with the minimal f value.
        obj_f = [_PEP_f_Hs(y, yt, i)[0] for i in Vs]
        idx_f = obj_f.index(np.min(obj_f))
        yp = Vs[idx_f]  # $\mathbf{y}$
        #
        # 5. Q = Q \cup {\bm{y}}.
        # 6. L = L \cup {i | y_i \neq s_i}.
        Q.append(deepcopy(yp))  # Q.append(yp.copy())
        L.append(int(idx_Vs[idx_f]))  # otherwise, np.integer
        QL[idx_Vs[idx_f]] = True
        #
        # 7. \mathbf{s} = \bm{y}.
        sp = deepcopy(yp)  # yp.copy()
        del Ns, idx_Vs, Vs, obj_f, idx_f, yp
    del QL
    # 8. Output Q.
    return deepcopy(Q), deepcopy(L)  # np.ndarray, int. type


# Pareto Ensemble Pruning
# ----------------------------------
#
# Algorithm 1 (PEP).
# Given a set of trained classifiers $H = {h_i}_{i=1}^n$,
# an objective $f: 2^H \to \mathbb{R}$ and an evaluation
# criterion $eval$, it contains:
#
#  1. Let g(\mathbf{s}) = (f(H_\mathbf{s}), |\mathbf{s}|)
#     be the bi-objective.
#  2. Let \mathbf{s} = randomly selected from {0, 1}^n and
#     P = \{\mathbf{s}\}.
#  3. Repeat
#  4.     Select \mathbf{s} \in P uniformly at random.
#  5.     Generate \mathbf{s}' by flipping each bit of \mathbf{s}
#         with prob.\frac{1}{n}.
#  6.     if \nexists \mathbf{z} \in P such that
#                                   \mathbf{z} \succ_g \mathbf{s}'
#  7.         P = (P - \{
#                           \mathbf{z} \in P |
#                           \mathbf{s}' \succeq_g \mathbf{z}
#                 \}) \cup \{\mathbf{s}'\}.
#  8.         Q = VDS(f, \mathbf{s}').
#  9.         for \mathbf{q} \in Q
# 10.             if \nexists \mathbf{z} \in P such that
#                                    \mathbf{z} \succ_g \mathbf{q}
# 11.                 P = (P - \{
#                                  \mathbf{z} \in P |
#                                  \mathbf{q} \succeq_g \mathbf{z}
#                         \}) \cup \{\mathbf{q}\}.
# 12. Output \argmin_{\mathbf{s} \in P} eval(\mathbf{s}).
#

def PEP_PEP(y, yt, nb_cls, rho):
    # 1. Let g(\mathbf{s}) = (f(H_\mathbf{s}), |\mathbf{s}|) be the
    #    bi-objective.
    # 2. Let \mathbf{s} = randomly selected from {0, 1}^n and
    #    P = \{\mathbf{s}\}.
    s = np.random.randint(2, size=nb_cls).tolist()
    P = [deepcopy(s)]  # P = [s.copy()]
    # 3. repeat
    nb_cnt = int(np.ceil(rho * nb_cls))  # counter
    while nb_cnt > 0:
        # 4. Select \mathbf{s} \in P uniformly at random.
        idx = np.random.randint(len(P))
        s0 = P[idx]
        # 5. Generate \mathbf{s}' by flipping each bit of \mathbf{s}
        #    with prob.\frac{1}{n}.
        sp = _PEP_flipping_uniformly(s0)
        g_sp = _PEP_bi_objective(y, yt, sp)
        # 6. if \nexists \mathbf{z} \in P such that
        #    \mathbf{z} \succ_g \mathbf{s}'
        flag1 = False
        for z1 in P:
            g_z1 = _PEP_bi_objective(y, yt, z1)
            if _PEP_dominate(g_z1, g_sp):
                flag1 = True
                break
        del g_z1, z1
        #   #
        if not flag1:
            # 7. P= (P-\{
            #               \mathbf{z} \in P |
            #               \mathbf{s}' \succeq_s \mathbf{z}
            #    \}) \cup \{\mathbf{s}'\}.
            idx1 = []
            # for i in range(len(P)):
            for i, _ in enumerate(P):
                g_z2 = _PEP_bi_objective(y, yt, P[i])
                if _PEP_weakly_dominate(g_sp, g_z2):
                    idx1.append(i)
            for i in idx1[:: -1]:
                del P[i]
            P.append(deepcopy(sp))  # sp.copy()
            del g_z2, i, idx1
            #
            # 8. Q = VDS(f, \mathbf{s}').
            Q, _ = _PEP_VDS(y, yt, nb_cls, sp)
            # 9. for \mathbf{q} \in Q
            for q in Q:
                g_q = _PEP_bi_objective(y, yt, q)
                # 10. if \nexists \mathbf{z} \in P such that
                #     \mathbf{z} \succ_g \mathbf{q}
                flag3 = False
                for z3 in P:
                    g_z3 = _PEP_bi_objective(y, yt, z3)
                    if _PEP_dominate(g_z3, g_q):
                        flag3 = True
                        break
                del g_z3, z3
                if not flag3:
                    # 11. P=(P- \{
                    #               \mathbf{z} \in P |
                    #               \mathbf{q} \succeq_g \mathbf{z}
                    #     \}) \cup \{\mathbf{q}\}.
                    idx3 = []
                    # for j in range(len(P)):
                    for j, _ in enumerate(P):
                        g_z4 = _PEP_bi_objective(y, yt, P[j])
                        if _PEP_weakly_dominate(g_q, g_z4):
                            idx3.append(j)
                    for j in idx3[:: -1]:
                        del P[j]
                    P.append(deepcopy(q))  # q.copy()
                    del g_z4, j, idx3
                del flag3, g_q
            del q, Q
        del flag1, g_sp, sp, s0, idx
        nb_cnt = nb_cnt - 1
        # end of this iteration
    del nb_cnt, s
    #
    # 12. Output \argmin_{\mathbf{s} \in P} eval(\mathbf{s}).
    obj_eval = [_PEP_f_Hs(y, yt, t)[0] for t in P]  # si/t
    idx_eval = obj_eval.index(np.min(obj_eval))
    s = P[idx_eval]
    del P, obj_eval, idx_eval
    if np.sum(s) == 0:
        s[np.random.randint(nb_cls)] = 1
    P = np.array(s, dtype=DTY_BOL)
    seq = np.where(np.array(s) == 1)[0]  # np.int64
    yo = np.array(yt)[P].tolist()
    del s
    return yo, P.tolist(), seq.tolist()  # list


def PEP_PEP_modify(y, yt, nb_cls, rho):
    nb_pru = int(np.ceil(rho * nb_cls))
    s = np.random.randint(2, size=nb_cls).tolist()
    P = [deepcopy(s)]  # s.copy()
    del s
    nb_cnt = nb_pru  # counter
    while nb_cnt > 0:
        idx = np.random.randint(len(P))
        s = P[idx]
        sp = _PEP_flipping_uniformly(s)
        g_sp = _PEP_bi_objective(y, yt, sp)

        flag1 = False
        for z1 in P:
            g_z1 = _PEP_bi_objective(y, yt, z1)
            if _PEP_dominate(g_z1, g_sp):
                flag1 = True
                break
        if not flag1:
            idx1 = []
            for i, _ in enumerate(P):  # for i in range(len(P)):
                g_z1 = _PEP_bi_objective(y, yt, P[i])
                if _PEP_weakly_dominate(g_sp, g_z1):
                    idx1.append(i)
            for i in idx1[:: -1]:
                del P[i]
            P.append(deepcopy(sp))
            del i, idx1

            Q, _ = _PEP_VDS(y, yt, nb_cls, sp)
            for q in Q:
                g_q = _PEP_bi_objective(y, yt, q)
                flag3 = False
                for z3 in P:
                    g_z3 = _PEP_bi_objective(y, yt, z3)
                    if _PEP_dominate(g_z3, g_q):
                        flag3 = True
                        break
                if not flag3:
                    idx3 = []
                    # for j in range(len(P)):
                    for j, _ in enumerate(P):
                        g_z3 = _PEP_bi_objective(y, yt, P[j])
                        if _PEP_weakly_dominate(g_q, g_z3):
                            idx3.append(j)
                    for j in idx3[:: -1]:
                        del P[j]
                    P.append(deepcopy(q))
                    del j, idx3

                del g_z3, z3, flag3, g_q
            del q, Q
        del g_z1, z1, flag1, g_sp, sp, s, idx
        nb_cnt = nb_cnt - 1
        # end of this iteration
        obj_eval = [_PEP_f_Hs(y, yt, si)[0] for si in P]
        idx_eval = obj_eval.index(np.min(obj_eval))
        s_ef = P[idx_eval]  # se/sf, s_eventually, s_finally
        del obj_eval, idx_eval
        if (np.sum(s_ef) <= nb_pru) and (np.sum(s_ef) > 0):
            break
    P_ef = np.array(s_ef, dtype=DTY_BOL)
    if np.sum(P_ef) == 0:
        P_ef[np.random.randint(nb_cls)] = True
    seq = np.where(P_ef)[0]  # np.int64
    yo = np.array(yt)[P_ef].tolist()  # yo = yt[P_ef]
    del s_ef, nb_cnt, P, nb_pru
    return yo, P_ef.tolist(), seq.tolist()


# ----------------------------------
# PEP split up
# ----------------------------------


def _pep_pep_split_up_nexists(y, yt, P_set, sp_q):
    g_sp_q = _PEP_bi_objective(y, yt, sp_q)
    # flag = False
    for z in P_set:
        g_z = _PEP_bi_objective(y, yt, z)
        if _PEP_dominate(g_z, g_sp_q):
            # flag = True
            # break
            return True, deepcopy(z)
    # return flag
    return False, []


def _pep_pep_refresh_weakly_domi(y, yt, P_set, sp_q):
    # all_z_in_P = list(range(len(P_set)))
    g_sp_q = _PEP_bi_objective(y, yt, sp_q)
    all_z_in_P = [_PEP_bi_objective(y, yt, i) for i in P_set]
    all_z_in_P = [_PEP_weakly_dominate(g_sp_q, i) for i in all_z_in_P]
    idx = np.where(np.logical_not(all_z_in_P))[0]
    P_set = np.array(P_set)[idx].tolist()
    P_set.append(deepcopy(sp_q))
    return P_set, idx.tolist()


def pep_pep_integrate(y, yt, nb_cls, rho):
    # 1. Let $g(\mathbf{s}) = (f(|H_\mathbf{s}|), |\mathbf{s}|)$
    #    be the bi-objective.
    # 2. Let $\mathbf{s}$ = randomly selected from $\{0,1\}^n$ and
    #    $P = \{\mathbf{s}\}$.
    s = np.random.randint(2, size=nb_cls).tolist()
    P = [deepcopy(s)]
    # 3. Repeat
    # 4.    Select $\mathbf{s} \in P$ uniformly at random.
    # 5.    Generate $\mathbf{s'}$ by flipping each bit of
    #       $\mathbf{s}$ with prob.$\frac{1}{n}$.
    nb_cnt = int(np.ceil(rho * nb_cls))
    while nb_cnt > 0:
        idx = np.random.randint(len(P))
        s0 = P[idx]
        sp = _PEP_flipping_uniformly(s0)
        fg1, _ = _pep_pep_split_up_nexists(y, yt, P, sp)
        if not fg1:
            P, _ = _pep_pep_refresh_weakly_domi(y, yt, P, sp)
            Q, _ = _PEP_VDS(y, yt, nb_cls, sp)
            for q in Q:
                fg3, _ = _pep_pep_split_up_nexists(y, yt, P, q)
                if not fg3:
                    P, _ = _pep_pep_refresh_weakly_domi(y, yt, P, q)
        nb_cnt -= 1
    # 6.    if $\nexists z\in P$ such that $z \succ_g s'$
    # 7.        P=(P-{z\in P| s' \succeq_g z}) \cup {s'}
    # 8.        Q = VDS(f,s')
    # 9.        for q \in Q
    # 10.           if \nexists z\in P such that z \succ_g q
    # 11.               P=(P-{z\in P| q \succeq_g z}) \cup {q}
    obj_eval = [_PEP_f_Hs(y, yt, st)[0] for st in P]
    idx_eval = obj_eval.index(np.min(obj_eval))
    s = P[idx_eval]
    # 12. Output $\argmin_{\mathbf{s} \in P} eval(\mathbf{s})$
    del P, obj_eval, idx_eval
    if np.sum(s) == 0:
        s[np.random.randint(nb_cls)] = 1
    PP = np.array(s, dtype=DTY_BOL)
    seq = np.where(np.array(s) == 1)[0]  # np.int64
    yo = np.array(yt)[PP].tolist()
    return yo, PP.tolist(), seq.tolist()


# def pep_pep_integrate_modify(y, yt, nb_cls, rho):
def pep_pep_re_modify(y, yt, nb_cls, rho):
    nb_pru = int(np.ceil(rho * nb_cls))
    s = np.random.randint(2, size=nb_cls).tolist()
    P = [deepcopy(s)]
    nb_cnt = nb_pru
    while nb_cnt > 0:
        idx = np.random.randint(len(P))
        s = P[idx]
        sp = _PEP_flipping_uniformly(s)
        fg1, _ = _pep_pep_split_up_nexists(y, yt, P, sp)
        if not fg1:
            P, _ = _pep_pep_refresh_weakly_domi(y, yt, P, sp)
            Q, _ = _PEP_VDS(y, yt, nb_cls, sp)
            for q in Q:
                fg3, _ = _pep_pep_split_up_nexists(y, yt, P, q)
                if not fg3:
                    P, _ = _pep_pep_refresh_weakly_domi(y, yt, P, q)
        nb_cnt = nb_cnt - 1
        obj_eval = [_PEP_f_Hs(y, yt, si)[0] for si in P]
        idx_eval = obj_eval.index(np.min(obj_eval))
        s_ef = P[idx_eval]  # eventually, finally
        del obj_eval, idx_eval
        # if (np.sum(s_ef) <= nb_pru) and (np.sum(s_ef) > 0):
        if 0 < np.sum(s_ef) <= nb_pru:
            break
    if np.sum(s_ef) == 0:
        s_ef[np.random.randint(nb_cls)] = 1
    P_ef = np.array(s_ef, dtype=DTY_BOL)
    seq = np.where(P_ef)[0]  # np.int64
    yo = np.array(yt)[P_ef].tolist()
    del s_ef, nb_cnt, P, nb_pru
    return yo, P_ef.tolist(), seq.tolist()


# ----------------------------------
# ----------------------------------


# =========================================
# Valuation Codes
# =========================================

# def pruning_methods(name_func, *para_func):
#     return name_func(*para_func)


# prelim.py
# obtain pruning
# ----------------------------------
# import core.ensem_pruning as dp
# import core.ensem_pruorder as dp


def contrastive_pruning_methods(name_pru, nb_cls, nb_pru,
                                y_val, y_cast, epsilon, rho):
    # since = time.time()  # y_insp, y_pred, coef, clfs
    if name_pru == 'KP':
        name_pru = 'KPz'
    assert name_pru in AVAILABLE_NAME_PRUNE

    if name_pru == "ES":
        ys_cast, P, seq = Early_Stopping(y_cast, nb_cls, nb_pru)
    elif name_pru == "KL":
        ys_cast, P, seq = KL_divergence_Pruning(y_cast, nb_cls, nb_pru)
    elif name_pru == "KL+":
        ys_cast, P, seq = KL_divergence_Pruning_modify(
            y_cast, nb_cls, nb_pru)
    elif name_pru == "KPk":
        ys_cast, P, seq = Kappa_Pruning_kuncheva(
            y_val, y_cast, nb_cls, nb_pru)
    elif name_pru == "KPz":
        ys_cast, P, seq = Kappa_Pruning_zhoubimu(
            y_val, y_cast, nb_cls, nb_pru)

    elif name_pru == "RE":
        ys_cast, P, seq = Reduce_Error_Pruning(
            y_val, y_cast, nb_cls, nb_pru)
    elif name_pru == "CM":
        ys_cast, P, seq = Complementarity_Measure_Pruning(
            y_val, y_cast, nb_cls, nb_pru)
    elif name_pru == "OO":
        ys_cast, P, seq, flag = Orientation_Ordering_Pruning(
            y_val, y_cast)

    elif name_pru in ["GMA", "GMM"]:  # == "GMA":
        ys_cast, P, seq = GMM_Algorithm(
            y_val, y_cast, nb_cls, nb_pru)
    elif name_pru == "LCS":
        ys_cast, P, seq = Local_Search(
            y_val, y_cast, nb_cls, nb_pru, epsilon)

    elif name_pru == "DREP":  # works for binary classification only
        vY = np.concatenate([
            [y_val], y_cast]).reshape(-1).tolist()
        vY, dY = judge_transform_need(vY)
        if dY > 2:
            raise UserWarning(
                "DREP works for binary classification only.")
        ys_cast, P, seq = DREP_Pruning(y_val, y_cast, nb_cls, rho)
    elif name_pru == "drepm":
        ys_cast, P, seq = drep_multi_modify_pruning(
            y_val, y_cast, nb_cls, rho)

    elif name_pru == "SEP":
        ys_cast, P, seq = PEP_SEP(y_val, y_cast, nb_cls, rho)
    elif name_pru == "OEP":
        ys_cast, P, seq = PEP_OEP(y_val, y_cast, nb_cls)
    elif name_pru == "PEP":
        ys_cast, P, seq = PEP_PEP(y_val, y_cast, nb_cls, rho)
    elif name_pru == "PEP+":
        ys_cast, P, seq = PEP_PEP_modify(y_val, y_cast, nb_cls, rho)

    elif name_pru == "pepre":
        ys_cast, P, seq = pep_pep_integrate(
            y_val, y_cast, nb_cls, rho)
    elif name_pru == "pepr+":
        ys_cast, P, seq = pep_pep_re_modify(
            y_val, y_cast, nb_cls, rho)
    else:
        # raise UserWarning("Error occurred in `contrastive_pruning_methods`.")
        # propose for `ustc_diversity`
        # P = dd.prune_diversity(y_val, y_cast, nb_pru, tradeoff)
        # ys_cast = np.array(y_cast)[np.array(P)].tolist()
        #
        raise UserWarning("Error occurred in `contrastive_pruning_methods`.")

    if name_pru != "OO":
        flag = None
    # tim_elapsed = time.time() - since
    return ys_cast, P, seq, flag


def contrastive_pruning_according_validation(
        name_pru, nb_cls, nb_pru, y_val, y_cast, epsilon, rho,
        y_insp, y_pred, coef, clfs):
    since = time.time()
    ys_cast, P, seq, flag = contrastive_pruning_methods(
        name_pru, nb_cls, nb_pru, y_val, y_cast, epsilon, rho)
    time_elapsed = time.time() - since

    ys_pred = np.array(y_pred)[P].tolist()
    ys_insp = np.array(y_insp)[P].tolist() if len(
        y_insp) > 0 else []
    # if we only have trn/tst with val, then
    # ys_cast is ys_insp and ys_insp=[] (for val)

    opt_coef = np.array(coef)[P].tolist()  # coef[P]
    # opt_clfs = np.array(clfs)[P].tolist()
    opt_clfs = [cj for i, cj in zip(P, clfs) if i]
    space_cost__ = asizeof(opt_clfs) + asizeof(opt_coef)

    return opt_coef, opt_clfs, ys_insp, ys_cast, ys_pred, \
        time_elapsed, space_cost__, P, seq, flag

# Notice that:
#   If we use `y_trn, y_val, y_tst`, then no doubts
#   If we use `y_trn, y_tst` without `y_val`, then:
#       _, _, [], ys_insp, ys_pred, _, _, P, seq, flag = \
#           contrastive_pruning_according_validation(
#               _, _, _, y_trn, y_insp, _, _, [], y_pred, _, _)
#       ys_cast = []
#
