# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()
import numpy as np

from pyensemble.utils_const import DTY_FLT



#==================================
# \citep{qian2015pareto}
#
# Pareto Ensemble Pruning (PEP)  (AAAI-15)  [Binary/ multi-class classification, Bagging]
#==================================
#



#----------------------------------
# assume
#----------------------------------
#
# $D = \{(\mathbf{x}_i, y_i)\} _{i=1}^m$        data set, with $m$ samples
# $H = \{h_j\} _{j=1}^n$                        set of trained individual classifiers, with $n$ ones
# $H_{\mathbf{s}}$ with $\mathbf{s} = \{0, 1\}^n$   pruned sub-ensemble, with a selector vector
#       $|\mathbf{s}| = \sum_{j=1}^n s_j$           minimize the size of $H_{\mathbf{s}}$
#
# ${\arg\min}_{ \mathbf{s} \in \{0, 1\}^n } \bigg( f(H_{\mathbf{s}}), |\mathbf{s| \bigg)$
#                                                   bi-objective ensemble pruning problem
# Note that: y\in {-1, +1},  transform!



# a pruned ensemble $H_{\mathbf{s}}$ is composited as
# multi-class classification
#
def PEP_Hs_x(y, yt, s):
    vY = np.unique(np.vstack((y, yt)))
    dY = len(vY)
    yt = np.array(yt)
    s = np.transpose([s])  # s = np.array(np.mat(s).T)
    #
    vote = [np.sum(s*(yt==i), axis=0).tolist() for i in vY]
    loca = np.array(vote).argmax(axis=0)
    Hsx = [vY[i] for i in loca]
    #
    del vY,dY, yt,s, vote,loca
    gc.collect()
    return deepcopy(Hsx)


# define the difference of two classifiers as
# [works for multi-class, after modification]
#
def PEP_diff_hihj(hi, hj):
    tem = np.array(hi) == np.array(hj)
    tem = np.array(tem, dtype=DTY_FLT) * 2 - 1
    return np.mean( (1. - tem) / 2. )

# and the error of one classifier as
# [works for multi-class now]
#
def PEP_err_hi(y, hi):
    tem = np.array(hi) == np.array(y)
    tem = np.array(tem, dtype=DTY_FLT) * 2 - 1
    return np.mean( (1. - tem) / 2. )

# both of them (i.e., \diff(hi,hj), \err(hi)) belong to [0, 1].
# If \diff(hi,hj) = 0 (or 1), hi and hj always make the same (or opposite) prediction;
# If \err(hi) = 0 (or 1), hi always make the right (or wrong) prediction.
# We hope: (1) \diff larger, (2) \err small


# the validation error is calculated as
# binary  [I think multi-class classification]
#
def PEP_f_Hs(y, yt, s):
    Hsx = PEP_Hs_x(y, yt, s)
    ans = np.mean(np.array(Hsx) != np.array(y))
    return ans, deepcopy(Hsx)



#----------------------------------
# performance objective $f$
#----------------------------------


# def PEP_objective_performance():
#     pass



#----------------------------------
# evaluation criterion $eval$
#----------------------------------



#----------------------------------
# OEP, SEP
#----------------------------------


# generate $\mathbf{s'}$ by flipping each bit of $\mathbf{s}$ with prob.$\frac{1}{n}$
#
def PEP_flipping_uniformly(s):
    n = len(s)
    pr = np.random.uniform(size=n)  # \in [0, 1]
    pr = (pr < 1. / n)  # <=
    # s', sprime
    sp = [1-s[i] if pr[i] else s[i] for i in range(n)]
    del n, pr
    gc.collect()
    return deepcopy(sp)



#----------------------------------
# Domination
#----------------------------------


# bi-objective
#
def PEP_bi_objective(y, yt, s):
    fHs, Hsx = PEP_f_Hs(y, yt, s)
    s_ab = np.sum(s)  # absolute
    ans = (fHs, s_ab)
    del fHs, Hsx, s_ab
    gc.collect()
    return deepcopy(ans)


#' ''
# the objective vector:
# $\mathbf{g}(\mathbf{s}) = (g_1, g_2)$
# $\mathbf{g}: \mathcal{S} \to \mathbb{R}^2$
#
# for two solutions
# $\mathbf{s}, \mathbf{s'} \in \mathcal{S}$
#' ''

# (1) s weakly dominate s'  if g1(s)<=g1(s') and g2(s)<=g2(s')
#
def PEP_weakly_dominate(g_s1, g_s2):
    if (g_s1[0] <= g_s2[0]) and (g_s1[1] <= g_s2[1]):
        return True
    return False

# (2) s dominate s'  if s \succeq_{g} s'  and either g1(s)<g1(s') or g2(s)<g2(s')
#
def PEP_dominate(g_s1, g_s2):
    # if PEP_weakly_dominate(g_s1, g_s2) == True:
    if PEP_weakly_dominate(g_s1, g_s2):
        if g_s1[0] < g_s2[0]:
            return True
        elif g_s1[1] < g_s2[1]:
            return True
        else:
            return False
    return False


