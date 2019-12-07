# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()
import time
import numpy as np

from pyensemble.utils_const import DTY_BOL, GAP_MID, GAP_INF
from pyensemble.pruning.ranking_based.Kappa_Pruning import KappaMulti


#----------------------------------
# GMM(S, k)
#----------------------------------
#
# Input:    S, a set of points; k, size of the subset
# Output:   S', a subset of S of size k
#   1.  S' <-- An arbitrary point p
#   2.  for i = 2,...,k do
#   3.      find p \in S\S' which maximizes min_{q\in S'} dist(p,q)
#   4.      S' <-- S' \cup {p}


def GMM_Kappa_sum(p, S, y):
    # ans = [KappaMulti(p, q, y)[0] for q in S]
    ans = []
    for q in S:
        tem, _, _ = KappaMulti(p, q, y)
        ans.append(tem)
    tem = np.sum(ans)
    return tem


def GMM_Algorithm(yt, y, nb_cls, nb_pru):
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    randseed = int(time.time() * GAP_MID % GAP_INF)
    prng = np.random.RandomState(randseed)
    idx = prng.randint(nb_cls)
    # idx = np.random.randint(nb_cls)
    P[idx] = True
    #
    for i in range(1, nb_pru):
        # find_max_p
        all_q_in_S = np.array(yt)[P].tolist()
        # idx_p_not_S = np.where(P == False)[0]
        idx_p_not_S = np.where(np.logical_not(P))[0]
        if len(idx_p_not_S) == 0:
            idx = -1
        else:
            ans = [GMM_Kappa_sum(yt[j], all_q_in_S, y) for j in idx_p_not_S]
            idx_p = ans.index( np.max(ans) )
            idx = idx_p_not_S[idx_p]
            del ans, idx_p
        del all_q_in_S, idx_p_not_S
        # fine_max_p
        if idx > -1:
            P[idx] = True
    #   #   #
    P = P.tolist()
    del randseed, prng, idx
    # del idx
    yo = np.array(yt)[P].tolist()
    gc.collect()
    return deepcopy(yo), deepcopy(P)


