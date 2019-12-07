# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()
import time
import numpy as np

from pyensemble.utils_const import DTY_FLT, DTY_BOL, GAP_MID, GAP_INF



#==================================
# \citep{li2012diversity}
#
# Diversity Regularized Ensemble Pruning (DREP)  [Binary, Bagging]
#==================================
#
# H = \{ h_i(\mathbf{x}) \}, i=1,\dots,n,  h_i:\mathcal{X}\mapsto\{-1,+1\}
# S = \{ (\mathbf{x}_k, y_k) \}, k=1,\dots,m
# Note that:  y \in {-1, +1},  transform!


#----------------------------------
# DREP
#----------------------------------


# $f(\mathbf{x};H) = \frac{1}{n} \sum_{1\leqslant i\leqslant n} h_i(\mathbf{x})$
#
def DREP_fxH(yt):
    yt = np.array(yt)
    if yt.ndim == 1:
        # bug!  # yt = np.array(np.mat(yt).T).tolist()
        yt = np.array([yt])
    fens = np.mean(yt, axis=0)  # .tolist()
    del yt
    #
    fens = np.sign(fens)
    randseed = int(time.time() * GAP_MID % GAP_INF)
    prng = np.random.RandomState(randseed)
    # fens = [prng.randint(2)*2-1 if i==0. else i for i in fens]
    #
    tie = [np.sum(fens == i) for i in [0, 1, -1]]
    if tie[1] > tie[2]:
        fens[fens == 0] = 1
    elif tie[1] < tie[2]:
        fens[fens == 0] = -1
    else:
        fens[fens == 0] = prng.randint(2) * 2 - 1
    fens = fens.tolist()
    #
    gc.collect()
    return deepcopy(fens)


# $\diff(h_i, h_j) = \frac{1}{m} \sum_{1\leqslant k\leqslant m}
#                    h_i(\mathbf{x}_k) h_j(\mathbf{x}_k)$
#
def DREP_diff(hi, hj):
    tem = np.array(hi) == np.array(hj)
    tem = np.array(tem, dtype=DTY_FLT) * 2 - 1
    return np.mean(tem)
    # works for multi-class now



# \rho \in (0, 1)
#
def DREP_Pruning(yt, y, nb_cls, rho):
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    accpl = [np.mean(np.array(h) != np.array(y)) for h in yt]
    idx = accpl.index( np.min(accpl) )
    P[idx] = True
    #
    flag = True  # whether the error of H* on S can be reduced
    nb_count = int(np.ceil(rho * nb_cls))  # = nb_pru
    #
    uY = np.unique(y).tolist()
    if len(uY) == 2 and 0 in uY and 1 in uY:
        tr_y = (np.array(y) * 2 - 1).tolist()
        tr_yt = (np.array(yt) * 2 - 1).tolist()
    elif len(uY) == 2 and -1 in uY and 1 in uY:
        tr_y = deepcopy(y)
        tr_yt = deepcopy(yt)
    else:  # len(uY) > 2
        tr_y = (np.array(y) - len(uY) // 2).tolist()
        tr_yt = (np.array(yt) - len(uY) // 2).tolist()
    del uY
    #
    while nb_count > 0:  # >=
        hstar = np.array(tr_yt)[P].tolist()
        hstar = DREP_fxH(hstar)
        # all_q_in_S = np.where(P == False)[0]
        all_q_in_S = np.where(np.logical_not(P))[0]
        dhstar = [ DREP_diff(tr_yt[q], hstar)  for q in all_q_in_S]
        dhidx = np.argsort(dhstar).tolist()  # sort in the ascending order
        tradeoff = int(np.ceil(rho * len(all_q_in_S)))
        gamma = dhidx[: tradeoff]  # index in Gamma
        gamma = [ all_q_in_S[q]  for q in gamma]
        #
        errHstar = np.mean(np.array(hstar) != np.array(tr_y))
        # idx = np.where(P == True)[0].tolist()
        idx = np.where(P)[0].tolist()
        errNew = [ np.mean(
            np.array(
                DREP_fxH(np.array(tr_yt)[idx+[p]].tolist())
            ) != np.array(tr_y)
        ) for p in gamma]
        errIdx = errNew.index( np.min(errNew) )
        if errNew[errIdx] <= errHstar:
            P[gamma[errIdx]] = True
            flag = True
        else:
            flag = False
        #
        del hstar,all_q_in_S,dhstar,dhidx,tradeoff, errHstar,errNew,errIdx
        nb_count -= 1
        # if flag == False:
        if not flag:
            break
    #   #   #
    yo = np.array(yt)[P].tolist()
    P = P.tolist()
    del idx,flag,accpl, tr_y,tr_yt
    gc.collect()
    return deepcopy(yo), deepcopy(P)


