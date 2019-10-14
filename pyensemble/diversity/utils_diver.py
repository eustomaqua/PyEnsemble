# coding: utf8
# Aim to: diversity measures in ensembles (existing methods)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from copy import deepcopy
# import time
import numpy as np

# from pyensemble.utils_const import DTY_FLT
from pyensemble.utils_const import DTY_INT
# from pyensemble.utils_const import check_zero


# -------------------------------------------
#   General
# -------------------------------------------


# pairwise measures


def contingency_table(hi, hj):
    if not len(hi) == len(hj):  # number of instances/samples
        raise AssertionError( 
            "These two individual classifiers have two different shapes.")
    vY = np.unique(np.concatenate([hi, hj])).tolist()
    #
    if len(vY) == 2 and 0 in vY and 1 in vY:
        hi = np.array(hi, dtype=DTY_INT) * 2 - 1
        hj = np.array(hj, dtype=DTY_INT) * 2 - 1
    else:  # len(vY) == 2 and -1 in vY and 1 in vY
        hi = np.array(hi, dtype=DTY_INT)
        hj = np.array(hj, dtype=DTY_INT)
    #   #
    a = np.sum((hi == 1) & (hj == 1))
    b = np.sum((hi == 1) & (hj == -1))
    c = np.sum((hi == -1) & (hj == 1))
    d = np.sum((hi == -1) & (hj == -1))
    #
    return a, b, c, d


def multiclass_contingency_table(ha, hb, y):
    vY = np.unique(np.concatenate([y, ha, hb]))
    dY = len(vY)  # L, number of classes/labels
    ha = np.array(ha);  hb = np.array(hb)
    #
    # construct a contingency table
    Cij = np.zeros(shape=(dY, dY))
    for i in range(dY):
        for j in range(dY):
            Cij[i, j] = np.sum((ha == vY[i]) & (hb == vY[j]))
    #   #   #
    return Cij.tolist()



# non-pairwise measures

def number_individuals_correct(yt, y):
    y = np.array(y, dtype=DTY_INT)
    yt = np.array(yt, dtype=DTY_INT)
    rho_x = np.sum(yt == y, axis=0)
    return rho_x.tolist()

