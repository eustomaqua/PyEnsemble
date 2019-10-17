# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()
import numpy as np

from pyensemble.utils_const import DTY_BOL, DTY_INT
from pyensemble.pruning.utils_inPEP import PEP_f_Hs



#----------------------------------
# OEP, SEP
#----------------------------------


# Ordering-based Ensemble Pruning
#
def PEP_OEP(yt, y, nb_cls):
    # nb_cls = len(yt)
    Hs = np.zeros(nb_cls, dtype=DTY_BOL).tolist()
    ordered_idx = []
    while np.sum(Hs) < nb_cls:
        Hu_idx = np.where(np.array(Hs) == False)[0].tolist()
        obj_f = []
        for h in Hu_idx:
            tem_s = deepcopy(Hs)
            tem_s[h] = True  # tem_idx
            tem_ans, _ = PEP_f_Hs(y, yt, tem_s)
            obj_f.append(tem_ans)
            del tem_s, tem_ans
        idx_f = obj_f.index( np.min(obj_f) )
        idx_f = Hu_idx[idx_f]
        ordered_idx.append( idx_f )
        Hs[idx_f] = True
        del Hu_idx, obj_f, idx_f
    del Hs
    #
    obj_eval = []
    for h in range(1, nb_cls + 1):
        tem_s = np.zeros(nb_cls, dtype=DTY_INT)
        tem_s[ordered_idx[: h]] = 1
        tem_ans, _ = PEP_f_Hs(y, yt, tem_s.tolist())
        obj_eval.append(tem_ans)
        del tem_s, tem_ans
    idx_k = obj_eval.index( np.min(obj_eval) )
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    P[ ordered_idx[: (idx_k + 1)] ] = True  # Notice!!
    del obj_eval, idx_k  #, nb_cls
    #
    yo = np.array(yt)[P].tolist()
    P = P.tolist()
    gc.collect()
    return deepcopy(yo), deepcopy(P)


