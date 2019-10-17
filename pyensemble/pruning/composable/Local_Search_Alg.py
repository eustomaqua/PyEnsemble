# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()
import time
import sys
import numpy as np

from pyensemble.utils_const import DTY_BOL, GAP_MID, GAP_INF
from pyensemble.pruning.ranking_based.Kappa_Pruning import KappaMulti


#----------------------------------
# Local Search Algorithm
#----------------------------------
#
# Input:    S, a set of points; k, size of the subset
# Output:   S', a subset of S of size k
#   1.  S' <-- An arbitrary set of k points which contains the two farthest points
#   2.  while there exists p\in S\S' and p'\in S' such that div(S'\{p'} \cup {p}) >= div(S')(1+\epsilon/n) do
#   3.      S' <-- S' \ {p'} \cup {p}


def LocalSearch_kappa_sum(S, y):
    n = len(S)
    Kij = np.zeros(shape=(n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            ans, _, _ = KappaMulti(S[i], S[j], y)
            Kij[i, j] = ans
    # upper triangular matrix
    ans = np.sum(Kij) / (n * (n - 1.) / 2.)
    del n, Kij
    gc.collect()
    return ans


def Local_Search(yt, y, nb_cls, nb_pru, epsilon):
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    # an abritrary set of k points which contains the two farthest points
    #
    Kij = np.zeros(shape=(nb_cls, nb_cls))
    Kij += sys.maxsize
    for i in range(nb_cls - 1):
        for j in range(i + 1, nb_cls):
            ans, _, _ = KappaMulti(yt[i], yt[j], y)
            Kij[i, j] = ans
    # upper triangular matrix
    idx1 = np.where(Kij == np.min(Kij))
    row = idx1[0][0];   col = idx1[1][0]
    P[row] = True;      P[col] = True
    del Kij, i,j, ans, idx1  # row, col
    #
    randseed = int(time.time() * GAP_MID % GAP_INF)
    prng = np.random.RandomState(randseed)
    idx2 = np.arange(nb_cls)
    prng.shuffle(idx2)
    idx3 = idx2[: nb_pru]
    if (row in idx3) and (col in idx3):
        idx4 = idx3  # idx2[: nb_pru]
    elif (row in idx3) and (col not in idx3):
        idx4 = idx3[: -1]
    elif (row not in idx3) and (col in idx3):
        idx4 = idx3[: -1]
    elif (row not in idx3) and (col not in idx3):
        idx4 = idx3[: -2]
    else:
        pass
    for i in idx4:
        P[i] = True
    del randseed, prng, idx2,idx3,idx4, row,col  # i,
    #
    # while there exists p\in S\S'
    nb_count = np.sum(P) * (len(P) - np.sum(P))  # nb_cls = len(P)
    yt = np.array(yt)
    S_within  = np.where(P ==  True)[0].tolist()
    S_without = np.where(P == False)[0].tolist()
    while nb_count >= 0:
        flag = False  # whether exists (p, q)?
        div_b4 = LocalSearch_kappa_sum(yt[S_within].tolist(), y)
        for p in S_within:
            idx_p = S_within.index(p)
            for q in S_without:
                tem_q = deepcopy(S_within)
                tem_q[idx_p] = q
                div_af = LocalSearch_kappa_sum(yt[tem_q].tolist(), y)
                if div_af > div_b4 * (1. + epsilon / nb_cls):
                    flag = True
                    S_within = deepcopy(tem_q)
                    tem_p = deepcopy(S_without)
                    idx_q = S_without.index(q)
                    tem_p[idx_q] = p
                    S_without = deepcopy(tem_p)
                    del tem_p, tem_q
                    break
            #   #   #
            if flag == True:
                break
        #   #   #
        if flag == False:
            break
    #   #   #
    # end while
    del nb_count, S_without  # S_within,
    #
    PP = np.zeros(nb_cls, dtype=DTY_BOL)
    PP[S_within] = True
    yo = np.array(yt)[PP].tolist()
    PP = PP.tolist()
    del S_within
    gc.collect()
    return deepcopy(yo), deepcopy(PP)


