# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()
import numpy as np

from pyensemble.utils_const import DTY_BOL
from pyensemble.pruning.utils_inPEP import PEP_f_Hs
from pyensemble.pruning.utils_inPEP import PEP_flipping_uniformly
from pyensemble.pruning.utils_inPEP import PEP_bi_objective
from pyensemble.pruning.utils_inPEP import PEP_dominate
from pyensemble.pruning.utils_inPEP import PEP_weakly_dominate
from pyensemble.pruning.optimization_based.PEP_inPEP import PEP_VDS



#----------------------------------
# PEP_inPEP
#
# Modification of mine (to speed up)
#----------------------------------


def PEP_PEP_modify(yt, y, nb_cls, rho):
    # nb_cls = len(yt)
    nb_pru = int(np.ceil(rho * nb_cls))
    s = np.random.randint(2, size=nb_cls).tolist()
    P = [ deepcopy(s) ]
    del s
    #
    nb_count = nb_pru
    while nb_count > 0:
        idx = np.random.randint( len(P) )
        s = P[idx]
        sp = PEP_flipping_uniformly(s)
        g_sp = PEP_bi_objective(y, yt, sp)
        #
        flag1 = False
        for z1 in P:
            g_z1 = PEP_bi_objective(y, yt, z1)
            if PEP_dominate(g_z1, g_sp):
                flag1 = True
                break
        if not flag1:
            idx1 = []
            for _ in range(len(P)):
                g_z1 = PEP_bi_objective(y, yt, P[i])
                if PEP_weakly_dominate(g_sp, g_z1):
                    idx1.append(i)
            for i in idx1[::-1]:
                del P[i]
            P.append( deepcopy(sp) )
            del i, idx1
            #
            Q, _ = PEP_VDS(y, yt, nb_cls, sp)
            for q in Q:
                g_q = PEP_bi_objective(y, yt, q)
                flag3 = False
                for z3 in P:
                    g_z3 = PEP_bi_objective(y, yt, z3)
                    if PEP_dominate(g_z3, g_q):
                        flag3 = True
                        break
                if not flag3:
                    idx3 = []
                    for _ in range(len(P)):
                        g_z3 = PEP_bi_objective(y, yt, P[j])
                        if PEP_weakly_dominate(g_q, g_z3):
                            idx3.append(j)
                    for j in idx3[::-1]:
                        del P[j]
                    P.append( deepcopy(q) )
                    del j, idx3
                #   #
                del g_z3, z3, flag3, g_q
            del q, Q
        del g_z1, z1, flag1, g_sp, sp, s, idx
        #
        nb_count = nb_count - 1
        # end of this iteration
        #
        obj_eval = [ PEP_f_Hs(y, yt, sj)[0]  for sj in P]
        idx_eval = obj_eval.index( np.min(obj_eval) )
        s_ef = P[idx_eval]  # se/sf, s_eventually, s_finally
        del obj_eval, idx_eval
        if (np.sum(s_ef) <= nb_pru) and (np.sum(s_ef) > 0):
            break
    #   #
    P_ef = np.array(s_ef, dtype=DTY_BOL)
    if np.sum(P_ef) == 0:
        P_ef[ np.random.randint(nb_cls) ] = True
    yo = np.array(yt)[P_ef].tolist()
    P_ef = P_ef.tolist()
    del s_ef, nb_count, P, nb_pru  #, nb_cls
    gc.collect()
    return deepcopy(yo), deepcopy(P_ef)


