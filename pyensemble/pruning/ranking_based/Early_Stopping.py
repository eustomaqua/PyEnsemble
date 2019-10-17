# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import numpy as np

from pyensemble.utils_const import DTY_BOL



#==================================
# \citep{margineantu1997pruning}
#
# Pruning Adaptive Boosting (ICML-97)  [multi-class classification, AdaBoost]
#==================================



#----------------------------------
# Early Stopping
# works for [multi-class]
#----------------------------------


def Early_Stopping(yt, nb_cls, nb_pru):
    yo = yt[: nb_pru]
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    P[: nb_pru] = True
    P = P.tolist()
    return deepcopy(yo), deepcopy(P)



#----------------------------------
# KL-divergence Pruning
# works for [mutli-class]
#----------------------------------
#
# KL distance between two probability distributions p and q:
# KL_distance = scipy.stats.entropy(p, q)


#----------------------------------
# Kappa Pruning [binary classification]
# ! not multi-class
# now works on multi-class
#----------------------------------


# def Kappa(nb_y, nb_c, ha, hb):
#     dY = nb_c  # dY = nb_lab  # nb_label
#     m = nb_y   # number of instances / samples


