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



#----------------------------------
# VDS, PEP
#----------------------------------


# VDS Subroutine
#
def PEP_VDS(y, yt, nb_cls, s):
    # nb_cls = len(s)
    QL = np.zeros(nb_cls, dtype=DTY_BOL)
    sp = deepcopy(s)  # $\mathbf{s}$
    # initial
    Q = [];     L = []
    # Let N(.) denote the set of neighbor solutions of a binary vector with Hamming distance 1.
    # Ns = [deepcopy(sp) for i in range(nb_cls)]
    # for i in range(nb_cls):
    #     Ns[i][i] = 1 - Ns[i][i]
    # Ns = np.array(Ns)
    while np.sum(QL) < nb_cls:
        # Let N(.) denote the set of neighbor solutions of a binary vector with Hamming distance 1.
        Ns = [deepcopy(sp) for i in range(nb_cls)]
        for i in range(nb_cls):
            Ns[i][i] = 1 - Ns[i][i]
        Ns = np.array(Ns)
        #
        # idx_Vs = np.where(QL == False)[0]
        idx_Vs = np.where(np.logical_not(QL))[0]
        Vs = Ns[idx_Vs].tolist()
        # an objective $f: 2^H \mapsto \mathbb{R}$
        obj_f = [PEP_f_Hs(y, yt, i)[0] for i in Vs]  # obj_f = [PEP_f_Hs(y, yt, s)[0] for i in Vs]
        idx_f = obj_f.index( np.min(obj_f) )
        yp = Vs[idx_f]  # $\mathbf{y}$
        Q.append( deepcopy(yp) )
        L.append( idx_Vs[idx_f] )
        QL[ idx_Vs[idx_f] ] = True
        sp = deepcopy(yp)
        #
        del Ns, idx_Vs, Vs, obj_f, idx_f, yp
    del QL  #, nb_cls
    gc.collect()
    return deepcopy(Q), deepcopy(L)


# Pareto Ensemble Pruning
#
def PEP_PEP(yt, y, nb_cls, rho):
    # nb_cls = len(yt)
    s = np.random.randint(2, size=nb_cls).tolist()
    P = [];     P.append( deepcopy(s) )
    #
    nb_count = int(np.ceil(rho * nb_cls))  # = nb_pru
    while nb_count > 0:
        idx = np.random.randint( len(P) )
        s0 = P[idx]
        sp = PEP_flipping_uniformly(s0)
        g_sp = PEP_bi_objective(y, yt, sp)
        #
        flag1 = False
        for z1 in P:
            g_z1 = PEP_bi_objective(y, yt, z1)
            # if PEP_dominate(g_z1, g_sp) == True:
            if PEP_dominate(g_z1, g_sp):
                flag1 = True
                del g_z1
                break
            else:
                del g_z1
        del z1
        #
        # if flag1 == False:
        if not flag1:
            idx1 = []
            for _ in range(len(P)):
                g_z2 = PEP_bi_objective(y, yt, P[i])
                # if PEP_weakly_dominate(g_sp, g_z2) == True:
                if PEP_weakly_dominate(g_sp, g_z2):
                    idx1.append(i)
                del g_z2
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
                    # if PEP_dominate(g_z3, g_q) == True:
                    if PEP_dominate(g_z3, g_q):
                        flag3 = True
                        del g_z3
                        break
                    else:
                        del g_z3
                del z3
                #
                # if flag3 == False:
                if not flag3:
                    idx3 = []
                    for _ in range(len(P)):
                        g_z4 = PEP_bi_objective(y, yt, P[j])
                        # if PEP_weakly_dominate(g_q, g_z4) == True:
                        if PEP_weakly_dominate(g_q, g_z4):
                            idx3.append(j)
                        del g_z4
                    for j in idx3[::-1]:
                        del P[j]
                    P.append( deepcopy(q) )
                    del j, idx3
                #   #
                del flag3, g_q
            del q, Q
        del flag1, g_sp, sp, s0, idx
        nb_count = nb_count - 1
        # end of this iteration
    del nb_count, s
    #
    obj_eval = [PEP_f_Hs(y, yt, t)[0] for t in P]
    idx_eval = obj_eval.index( np.min(obj_eval) )
    s = P[idx_eval]
    del P, obj_eval, idx_eval
    P = np.array(s, dtype=DTY_BOL)
    if np.sum(P) == 0:
        P[ np.random.randint(nb_cls) ] = True
    P = P.tolist()
    yo = np.array(yt)[P].tolist()
    del s  #, nb_cls
    gc.collect()
    return deepcopy(yo), deepcopy(P)


