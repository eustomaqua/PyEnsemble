# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()
import sys

import numpy as np

from pyensemble.utils_const import DTY_FLT
from pyensemble.utils_const import DTY_BOL
from pyensemble.utils_const import check_zero



#==================================
# \citep{margineantu1997pruning}
#
# Pruning Adaptive Boosting (ICML-97)  [multi-class classification, AdaBoost]
#==================================



#----------------------------------
# Kappa Pruning [binary classification]
# ! not multi-class
# now works on multi-class
#----------------------------------


# def Kappa(nb_y, nb_c, ha, hb):
#     dY = nb_c  # dY = nb_lab  # nb_label
#     m = nb_y   # number of instances / samples


def KappaMulti(ha, hb, y):
    vY = np.unique(np.concatenate([y, ha, hb]))
    dY = len(vY)  # number of labels / classes
    ha = np.array(ha);  hb = np.array(hb)
    #
    # construct a contingency table
    Cij = np.zeros(shape=(dY, dY))
    for i in range(dY):
        for j in range(dY):
            Cij[i, j] = np.sum((ha == vY[i]) & (hb == vY[j]))
    m = len(y)  # number of instances / samples
    #
    c_diagonal = [Cij[i][i] for i in range(dY)]  # Cij[i, i]
    theta1 = np.sum(c_diagonal) / float(m)
    c_row_sum = [np.prod(
        [(Cij[i, i] + Cij[i, j]) for j in range(dY) if (j != i)]
    ) for i in range(dY)]
    c_col_sum = [np.prod(
        [(Cij[i, j] + Cij[j, j]) for i in range(dY) if (i != j)]
    ) for j in range(dY)]
    theta2 = np.sum(np.multiply(c_row_sum, c_col_sum)) / (float(m) ** 2)
    #
    ans = (theta1 - theta2) / check_zero(1. - theta2)
    del dY, ha,hb, Cij, m
    del c_row_sum, c_col_sum, c_diagonal
    gc.collect()
    return ans, theta1, theta2


def Kappa_Pruning(yt, y, nb_cls, nb_pru):
    # initial
    Kij = np.zeros(shape=(nb_cls, nb_cls), dtype=DTY_FLT)
    Kij += sys.maxsize  # sys.maxint
    #
    for i in range(nb_cls - 1):
        for j in range(i + 1, nb_cls):
            ans, _, _ = KappaMulti(yt[i], yt[j], y)
            Kij[i, j] = ans
    # upper triangular / triangle matrix
    #
    # the lowest \kappa
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    nb_p = 0
    while nb_p < nb_pru:
        idx = np.where(Kij == np.min(Kij))
        try:
            row = idx[0][0];    col = idx[1][0]
        except Exception as e:
            print("Kappa_Pruning -- nb_cls {} nb_pru {} nb_p {}".format(
                nb_cls, nb_pru, nb_p))
            print("Kappa_Pruning -- y, yt, {} \n{:23s}{}".format(y, '', yt))
            print("Kappa_Pruning -- idx {}".format(idx))
            print("Kappa_Pruning -- Kij \n{:20s}{}".format('', Kij))
            raise e
        else:
            pass
        finally:
            pass
        #   #
        if nb_p + 1 == nb_pru:
            P[row] = True
            Kij[row, :] = sys.maxsize
            Kij[:, row] = sys.maxsize
            nb_p += 1
        else:
            P[row] = True;  P[col] = True
            Kij[row, :] = sys.maxsize
            Kij[:, col] = sys.maxsize
            Kij[:, row] = sys.maxsize
            Kij[col, :] = sys.maxsize
            nb_p += 2
        del idx, row, col
    #   #
    yo = np.array(yt)[P == True].tolist()
    P = P.tolist()
    del nb_p, Kij
    gc.collect()
    return deepcopy(yo), deepcopy(P)


