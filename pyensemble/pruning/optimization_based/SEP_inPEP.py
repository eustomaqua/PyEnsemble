# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()
import numpy as np

from pyensemble.utils_const import DTY_INT, DTY_BOL
from pyensemble.pruning.utils_inPEP import PEP_f_Hs
from pyensemble.pruning.utils_inPEP import PEP_flipping_uniformly



#----------------------------------
# OEP, SEP
#----------------------------------


# Simple Ensemble Pruning
#
def PEP_SEP(yt, y, nb_cls, rho):
    # nb_cls = len(yt)  # n
    # s = np.random.randint(2, size=nb_cls).tolist()
    tem_s = np.random.uniform(size=nb_cls)  # \in [0, 1]
    tem_i = (tem_s <= rho)
    if np.sum(tem_i) == 0:
        tem_i[ np.random.randint(nb_cls) ] = True
    s = np.zeros(nb_cls, dtype=DTY_INT)
    s[tem_i] = 1
    s = s.tolist()
    del tem_s, tem_i
    #
    nb_pru = int(np.ceil(rho * nb_cls))
    nb_count = nb_pru
    while nb_count >= 0:
        sp = PEP_flipping_uniformly(s)
        f1, _ = PEP_f_Hs(y, yt, sp)
        f2, _ = PEP_f_Hs(y, yt, s)
        if f1 <= f2:
            s = deepcopy(sp)
        #   #
        nb_count = nb_count - 1  # nb_count -= 1
        del sp, f1, f2
        if np.sum(s) > nb_pru:
            break
    #   #   #
    yo = np.array(yt)[np.array(s) == 1].tolist()
    P = np.array(s, dtype=DTY_BOL).tolist()
    del nb_count, s
    gc.collect()
    return deepcopy(yo), deepcopy(P)


