# coding: utf8
# Aim to: diversity measures in ensembles (existing methods)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pyensemble.utils_const import DTY_FLT
from pyensemble.utils_const import DTY_INT
from pyensemble.utils_const import check_zero

from pyensemble.diversity.utils_diver import number_individuals_correctly



#-------------------------------------------
# Non-pairwise Measures [multi-class classification]
#-------------------------------------------
#
# nb_cls = len(yt)
# m = len(y) = len(yt[0])



# Kohavi-Wolpert Variance
# the larger the kw value, the larger the diversity
#

def Kohavi_Wolpert_Variance_multiclass(yt, y, m, nb_cls):
    rho_x = number_individuals_correctly(yt, y)
    rho_x = np.array(rho_x, dtype=DTY_FLT)
    return np.sum(rho_x * (nb_cls - rho_x)) / (float(m) * nb_cls ** 2)



# Interrater agreement
# =1, totally agree; \leqslant 0, even less than what is expected by chance
#

def Interrater_agreement_multiclass(yt, y, m, nb_cls):
    y = np.array(y, dtype=DTY_INT)
    yt = np.array(yt, dtype=DTY_INT)
    p_bar = np.sum(np.sum(yt == y, axis=1)) / (float(m) * nb_cls)
    rho_x = np.sum(yt == y, axis=0)
    numerator = np.sum(rho_x * (nb_cls - rho_x)) / float(nb_cls)
    denominator = m * (nb_cls - 1.) * p_bar * (1. - p_bar)
    return 1. - numerator / check_zero(denominator)



# Entropy [works for multi-class classification]
#

def Entropy_cc_multiclass(yt, y, m, nb_cls):
    yt = np.array(yt, dtype=DTY_INT)
    uY = np.unique(np.concatenate([[y], yt]))
    ans = np.zeros(m, dtype=DTY_FLT)
    for i in uY:
        P_y_xk = np.sum(yt == i, axis=0) / float(nb_cls)
        tem = list(map(check_zero, P_y_xk))
        tem = (-1. * P_y_xk) * np.log(tem)
        ans += tem
    return np.sum(ans) / float(m)


def Entropy_sk_multiclass(yt, y, m, nb_cls):
    rho_x = number_individuals_correctly(yt, y)
    rho_x = np.array(rho_x, dtype=DTY_FLT)
    tmp = list(map(min, rho_x, nb_cls - rho_x))
    denominator = nb_cls - np.ceil(nb_cls / 2.)
    return np.sum(tmp) / float(m) / check_zero(denominator)



# Difficulty
# the smaller the theta value, the larger the diversity
#

def Difficulty_multiclass(yt, y, nb_cls):
    y = np.array(y, dtype=DTY_INT)
    yt = np.array(yt, dtype=DTY_INT)
    X = np.sum(yt == y, axis=0) / float(nb_cls)
    return np.var(X)



# Generalized Diversity
# \in [0, 1];   the diversity is minimized when gd=0
#

def Generalized_Diversity_multiclass(yt, y, m, nb_cls):
    y = np.array(y, dtype=DTY_INT)
    yt = np.array(yt, dtype=DTY_INT)
    failing = np.sum(yt != y, axis=0)
    # failing = np.sum(yt != y, axis=0) / nb_cls * nb_cls
    #
    pi = [-1.]
    for i in range(1, nb_cls + 1):
        tem = np.sum(failing == i) / float(m)
        pi.append(tem)
    #   #
    p_1 = 0.
    for i in range(1, nb_cls + 1):
        p_1 += pi[i] * i / nb_cls
    p_2 = 0.
    for i in range(1, nb_cls + 1):
        p_2 += pi[i] * (i * (i - 1.) / nb_cls / (nb_cls - 1.))
    #   #
    return 1. - p_2 / check_zero(p_1)



# Coincident Failure
#

def Coincident_Failure_multiclass(yt, y, m, nb_cls):
    y = np.array(y, dtype=DTY_INT)
    yt = np.array(yt, dtype=DTY_INT)
    failing = np.sum(yt != y, axis=0)
    pi = []
    for i in range(nb_cls + 1):
        tem = np.sum(failing == i) / float(m)
        pi.append(tem)
    #   #
    if pi[0] == 1.:
        return 0.
    if pi[0] < 1.:
        ans = 0.
        for i in range(1, nb_cls + 1):
            ans += pi[i] * (nb_cls - i) / (nb_cls - 1.)
        #   #
        return ans / check_zero(1. - pi[0])
    return

