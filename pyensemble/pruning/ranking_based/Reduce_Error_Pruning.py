# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()

import numpy as np

from pyensemble.utils_const import DTY_BOL
from pyensemble.classify.voting import plurality_voting



#==================================
# \citep{martinez2009analysis}
#
# An Analysis of Ensemble Pruning Techniques Based on Ordered Aggregation
# (TPAMI)  [multi-class classification, Bagging]
#==================================



#----------------------------------
# Reduce-Error Pruning
#----------------------------------


# need to use a pruning set, subdivided from training set, with a sub-training set
#
def Reduce_Error_Pruning(yt, y, nb_cls, nb_pru):
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    yt = np.array(yt)   # y = np.array(y)
    #
    # first
    err = np.mean(yt != np.array(y), axis=1)
    idx = err.argmin()  # argmax()
    P[idx] = True
    #
    # next
    while np.sum(P) < nb_pru:
        # find the next idx
        # not_in_p = np.where(P == False)[0]
        not_in_p = np.where(np.logical_not(P))[0]
        anserr = []
        for i in not_in_p:
            temP = deepcopy(P)
            temP[i] = True
            temyt = yt[temP].tolist()
            temfens = plurality_voting(y, temyt)
            temerr = np.mean(np.array(temfens) != np.array(y), axis=0)
            anserr.append( temerr )
            del temP, temyt, temfens, temerr
        #   #
        idx = anserr.index( np.min(anserr) )
        P[ not_in_p[idx] ] = True
        del anserr, idx, not_in_p
    #   #
    yo = yt[P].tolist()
    P = P.tolist()
    gc.collect()
    return deepcopy(yo), deepcopy(P)


