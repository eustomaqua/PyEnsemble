# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()

import numpy as np

from pyensemble.utils_const import DTY_FLT
from pyensemble.utils_const import DTY_BOL
from pyensemble.utils_const import check_zero



#----------------------------------
# data_entropy.py
#----------------------------------
#
# data_entropy.py ---- Inspired by margineantu1997pruning
# KL distance between two probability distributions p and q:
# KL_distance = scipy.stats.entropy(p, q)


# the KL distance between two probability distributions p and q is:
# D(p || q) =
#

def KLD(p, q):
    p = np.array(p, dtype=DTY_FLT)
    q = np.array(q, dtype=DTY_FLT)
    if np.sum(p) != 1.0:
        tem = np.sum(p)
        p /= check_zero(tem)
    if np.sum(q) != 1.0:
        tem = np.sum(q)
        q /= check_zero(tem)
    ans = 0.;   n = len(p)
    for i in range(n):
        tem = p[i] / check_zero(q[i])
        tem = p[i] * np.log(check_zero(tem))
        ans += tem
    return ans



#==================================
# \citep{margineantu1997pruning}
#
# Pruning Adaptive Boosting (ICML-97)  [multi-class classification, AdaBoost]
#==================================



#----------------------------------
# KL-divergence Pruning
# works for [mutli-class]
#----------------------------------
#
# KL distance between two probability distributions p and q:
# KL_distance = scipy.stats.entropy(p, q)


# X, Y = classification vectors
#

def KLD_vectors(X, Y):
    vXY = np.unique(np.concatenate([X, Y])).tolist()
    dXY = len(vXY)
    px = np.zeros(dXY); py = np.zeros(dXY)
    X = np.array(X);    Y = np.array(Y)
    for i in range(dXY):
        px[i] = np.mean(X == vXY[i])
        py[i] = np.mean(Y == vXY[i])
    px = px.tolist();   py = py.tolist()
    del i, X, Y, dXY
    ans = KLD(px, py)
    del px, py
    gc.collect()
    return ans


def JU_set_of_vectors(U):
    ans = 0.;   u = len(U)
    for i in range(u - 1):
        for j in range(i + 1, u):
            ans += KLD_vectors(U[i], U[j])
    del i, j, u
    gc.collect()
    return ans


def U_next_idx(yt, P):
    P = np.array(P)
    # not_in_p = np.where(P == False)[0]
    not_in_p = np.where(np.logical_not(P))[0]
    #
    yt = np.array(yt);  ansJ = []
    for i in not_in_p:
        ansP = deepcopy(P)
        ansP[i] = True
        # ansU = yt[ansP == True].tolist()
        ansU = yt[ansP].tolist()
        ansJ.append(JU_set_of_vectors(ansU))
        del ansP, ansU
    idx = ansJ.index(np.max(ansJ))
    del ansJ, yt, P
    gc.collect()
    return not_in_p[idx]


def KL_divergence_Pruning(yt, nb_cls, nb_pru):
    P = np.zeros(nb_cls, dtype=DTY_BOL).tolist()
    P[0] = True
    while np.sum(P) < nb_pru:
        idx = U_next_idx(yt, P)
        P[idx] = True
        del idx
    yo = np.array(yt)[P].tolist()
    return deepcopy(yo), deepcopy(P)


