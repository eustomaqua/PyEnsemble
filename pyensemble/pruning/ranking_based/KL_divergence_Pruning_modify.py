# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()

import numpy as np

from pyensemble.utils_const import DTY_BOL



#----------------------------------
# data_entropy.py
#----------------------------------
#

from pyensemble.pruning.ranking_based.KL_divergence_Pruning import KLD


# softmax regression
#
def softmax(y):
    return np.exp(y) / np.sum(np.exp(y), axis=0)



#==================================
# Modification of mine (with softmax)
#==================================


#----------------------------------
# KL-divergence Pruning
#----------------------------------


# KL distance between two vectors X and Y:
#
def KLD_pq(X, Y):
    # return stats.entropy(p, q)  # default: base=e
    p = softmax(X)
    q = softmax(Y)
    ans = KLD(p, q)
    del p, q
    gc.collect()
    return ans


def J(U):
    ans = 0.;   u = len(U)
    for i in range(u - 1):
        for j in range(i + 1, u):
            ans += KLD_pq(U[i], U[j])
    return ans


def KL_find_next(yt, P):
    P = np.array(P)
    # not_in_p = np.where(P == False)[0]
    not_in_p = np.where(np.logical_not(P))[0]
    #
    yt = np.array(yt)
    ansJ = []
    for i in not_in_p:
        ansP = deepcopy(P)
        ansP[i] = True
        # ansU = yt[ansP == True].tolist()
        ansU = yt[ansP].tolist()
        ansJ.append( J(ansU) )
        del ansP, ansU
    idx = ansJ.index( np.max(ansJ) )
    del ansJ, yt, P
    #
    gc.collect()
    return not_in_p[idx]  # Notice the position of P



#
def KL_divergence_Pruning_modify(yt, nb_cls, nb_pru):
    # P = [False] * nb_cls;   P[0] = True
    P = np.zeros(nb_cls, dtype=DTY_BOL).tolist()
    P[0] = True
    while np.sum(P) < nb_pru:
        idx = KL_find_next(yt, P)
        P[idx] = True
    # yo = np.array(yt)[np.array(P) == True].tolist()
    yo = np.array(yt)[P].tolist()
    return deepcopy(yo), deepcopy(P)


