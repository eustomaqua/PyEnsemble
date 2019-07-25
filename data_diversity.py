# coding: utf8
# Aim to: diversity measures in ensembles, (existing methods)


# Including:
#
# self.pairwise    = ['Disagreement', 'Q_statistic', 'Correlation', 'K_statistic', 'Double_fault']
# self.nonpairwise = ['KWVariance', 'Interrater', 'EntropyCC', 'EntropySK', 'Difficulty', 'Generalized', 'CoinFailure']
#
#   1) pairwise measure
#       Disagreement        |   Disagreement_Measure_multiclass(hi, hj,    m)   Disagreement_Measure_binary(   hi, hj,    m)
#       Q_statistic         |                                                   Q_Statistic_binary(            hi, hj      )
#       Correlation         |                                                   Correlation_Coefficient_binary(hi, hj      )
#       K_statistic         |   Kappa_Statistic_multiclass(     hi, hj, y, m)   Kappa_Statistic_binary(        hi, hj,    m)
#       Double_fault        |   Double_Fault_Measure_multiclass(hi, hj, y, m)   Double_Fault_Measure_binary(   hi, hj, y, m)
#   2) non-pairwise measure
#       KWVariance          |   Kohavi_Wolpert_Variance_multiclass( yt, y, m, nb_cls)
#       Interrater          |   Interrater_agreement_multiclass(    yt, y, m, nb_cls)
#       EntropyCC           |   Entropy_cc_multiclass(              yt, y, m, nb_cls)
#       EntropySK           |   Entropy_sk_multiclass(              yt, y, m, nb_cls)
#       Difficulty          |   Difficulty_multiclass(              yt, y,    nb_cls)
#       Generalized         |   Generalized_Diversity_multiclass(   yt, y, m, nb_cls)
#       CoinFailure         |   Coincident_Failure_multiclass(      yt, y, m, nb_cls)
#   3) utils
#       for pairwise        |   multiclass_contingency_table(   ha, hb, y)      contingency_table(          hi, hj)
#       for non-pairwise    |   number_individuals_correctly(      yt,  y)
#



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
# garbage collector
import gc
gc.enable()

import time
import numpy as np 


from utils_constant import DTY_FLT
from utils_constant import DTY_INT
# from utils_constant import DTY_BOL

# from utils_constant import CONST_ZERO
# from utils_constant import INF_GAP  # 1e-12
# from utils_constant import GAP_NAN
from utils_constant import check_zero



#===================================
# Diversity Measures
#===================================
#
# hi, hj:   list, not np.ndarray, with shape [m,]
# y \in {0, 1} -> {-1, +1}
# yt:       list, [[m,] nb_cls]
# 
# m = nb_y



#-----------------------------------
# General
#-----------------------------------


# pairwise measures


def contingency_table(hi, hj):
    # m = len(hi)   # = len(hj)
    # hi, hj \in {0, 1}
    # 
    # hi = np.array(hi, dtype=DTY_INT) * 2 - 1
    # hj = np.array(hj, dtype=DTY_FLT) * 2 - 1
    # 
    vY = np.unique(np.concatenate([hi, hj])).tolist()
    # if vY == [0, 1]:
    if len(vY) == 2 and 0 in vY and 1 in vY:
        hi = np.array(hi, dtype=DTY_INT) * 2 - 1
        hj = np.array(hj, dtype=DTY_INT) * 2 - 1
        # vY = np.unique(np.concatenate([hi, hj])).tolist()
    else:
        hi = np.array(hi, dtype=DTY_INT)
        hj = np.array(hj, dtype=DTY_INT)
    #   #
    a = np.sum((hi == 1) & (hj == 1))
    b = np.sum((hi == 1) & (hj == -1))
    c = np.sum((hi == -1) & (hj == 1))
    d = np.sum((hi == -1) & (hj == -1))
    # 
    return a, b, c, d  #, m


def multiclass_contingency_table(ha, hb, y):
    # vY = np.unique(y);  dY = len(vY)    # L
    vY = np.unique(np.concatenate([y, ha, hb]))
    dY = len(vY)
    ha = np.array(ha);  hb = np.array(hb)
    #
    # construct a contingency table
    Cij = np.zeros(shape=(dY, dY))
    for i in range(dY):
        for j in range(dY):
            Cij[i, j] = np.sum((ha == vY[i]) & (hb == vY[j]))
    #   #   #
    # m = len(y)  # number of instances / samples
    # return deepcopy(Cij)
    return Cij.tolist()  # , m



# non-pairwise measures

def number_individuals_correctly(yt, y):
    y = np.array(y, dtype=DTY_INT)
    yt = np.array(yt, dtype=DTY_INT)
    rho_x = np.sum(yt == y, axis=0)
    return rho_x.tolist()



#-----------------------------------
# Pairwise Measures [multi-class classification]
#   except `Q-statistic, Correlation-Coefficient` [binary classification]
#-----------------------------------


# Disagreement
# \in [0, 1];   the larger the value, the larger the diversity
#
def Disagreement_Measure_multiclass(hi, hj, m):
    # m = len(hi)  # = len(hj)
    hi = np.array(hi, dtype=DTY_INT)
    hj = np.array(hj, dtype=DTY_INT)
    return np.sum(hi != hj) / float(m)

def Disagreement_Measure_binary(hi, hj, m):
    # m = len(hi)  # = len(hj)
    a, b, c, d = contingency_table(hi, hj)
    return (b + c) / float(m)



# Q statistic
# [only works for binary classification]
# \in [-1, 1];  different / independent (=0) / similar predictions
#

def Q_Statistic_binary(hi, hj):
    a, b, c, d = contingency_table(hi, hj)
    tem = a * d + b * c
    return (a * d - b * c) / check_zero(tem)



# Correlation Coefficient
# [only works for binary classification]
# similar with Q-statistic;     |\rho_ij| \geqslant |Q_ij|
# 

def Correlation_Coefficient_binary(hi, hj):
    a, b, c, d = contingency_table(hi, hj)
    denominator = (a + b) * (a + c) * (c + d) * (b + d)
    denominator = np.sqrt(denominator)
    return (a * d - b * c) / check_zero(denominator)



# \kappa Statistic
# =1, totally agree; =0, agree by chance; <0, rare case, less than expected by chance
#
def Kappa_Statistic_multiclass(hi, hj, y, m):
    # m = len(y)  # = len(hi) = len(hj)
    # vY = np.unique(y);  dY = len(vY)  # L
    vY = np.unique(np.concatenate([y, hi, hj]))
    dY = len(vY)
    Cij = multiclass_contingency_table(hi, hj, y)
    Cij = np.array(Cij, dtype=DTY_FLT)
    # m = len(y)  # number of instances / samples
    #
    c_diagonal = [Cij[i][i] for i in range(dY)]  # Cij[i, i]
    theta1 = np.sum(c_diagonal) / float(m)
    #
    c_row_sum = [np.prod([Cij[i,i] + Cij[i,j] for j in range(dY) if j!=i]) for i in range(dY)]
    c_col_sum = [np.prod([Cij[i,j] + Cij[j,j] for i in range(dY) if i!=j]) for j in range(dY)]
    theta2 = np.sum(np.multiply(c_row_sum, c_col_sum)) / (float(m) ** 2)
    #
    ans = (theta1 - theta2) / check_zero(1. - theta2)
    del dY, vY, c_diagonal, c_row_sum, c_col_sum
    gc.collect()
    return ans, theta1, theta2
    # return ans


def Kappa_Statistic_binary(hi, hj, m):
    a, b, c, d = contingency_table(hi, hj)
    Theta_1 = (a + d) / float(m)
    Theta_2 = ((a + b) * (a + c) + (c + d) * (b + d)) / (float(m) ** 2)
    return (Theta_1 - Theta_2) / check_zero(1. - Theta_2)



# Double-Fault
# 
# 
def Double_Fault_Measure_multiclass(hi, hj, y, m):
    y = np.array(y, dtype=DTY_INT)
    hi = np.array(hi, dtype=DTY_INT)
    hj = np.array(hj, dtype=DTY_INT)
    return np.sum((hi != y) & (hj != y)) / float(m)

def Double_Fault_Measure_binary(hi, hj, y, m):
    ei = np.array(hi) != np.array(y)
    ej = np.array(hj) != np.array(y)
    e = np.sum(ei & ej)
    return e / float(m)



#-----------------------------------
# Non-pairwise Measures [multi-class classification]
#-----------------------------------



# Kohavi-Wolpert Variance
# the larger the kw value, the larger the diversity
#
def Kohavi_Wolpert_Variance_multiclass(yt, y, m, nb_cls):
    # nb_cls = len(yt);   m = len(y)  # = len(yt[0])
    #
    rho_x = number_individuals_correctly(yt, y)
    rho_x = np.array(rho_x, dtype=DTY_FLT)
    #
    return np.sum(rho_x * (nb_cls - rho_x)) / (float(m) * nb_cls ** 2)



# Interrater agreement
# =1, totally agree; \leqslant 0, even less than what is expected by chance
#
def Interrater_agreement_multiclass(yt, y, m, nb_cls):
    y = np.array(y, dtype=DTY_INT)
    yt = np.array(yt, dtype=DTY_INT)
    p_bar = np.sum(np.sum(yt == y, axis=1)) / (float(m) * nb_cls)
    rho_x = np.sum(yt == y, axis=0)
    numerator = np.sum(rho_x * (nb_cls - rho_x)) / float(nb_cls)  # 分子
    denominator = m * (nb_cls - 1.) * p_bar * (1. - p_bar)  # 分母
    return 1. - numerator / check_zero(denominator)



# Entropy [works for multi-class classification]
#

# def Entropy_cc_multiclass(yt, m, nb_cls):
def Entropy_cc_multiclass(yt, y, m, nb_cls):
    yt = np.array(yt, dtype=DTY_INT)
    # uY = np.unique(yt)
    uY = np.unique(np.concatenate([[y], yt]))
    ans = np.zeros(m, dtype=DTY_FLT) 
    for i in uY:
        P_y_xk = np.sum(yt == i, axis=0) / float(nb_cls)
        tem = list(map(check_zero, P_y_xk))
        tem = (-1. * P_y_xk) * np.log(tem)
        ans += tem
    return np.sum(ans) / float(m)

def Entropy_sk_multiclass(yt, y, m, nb_cls):
    #
    rho_x = number_individuals_correctly(yt, y)
    rho_x = np.array(rho_x, dtype=DTY_FLT)
    tmp = list(map(min, rho_x, nb_cls - rho_x))
    denominator = nb_cls - np.ceil(nb_cls / 2.)
    # 
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



#-----------------------------------
# Overall interface
#-----------------------------------


# def pairwise_measure_for_couple(name_diver, y, hi, hj, m):
def pairwise_measure_for_item(name_div, hi, hj, y, m):
    if name_div in ['Q_statistic', 'Correlation']:
        ya = np.array(hi) == np.array(y)
        yb = np.array(hj) == np.array(y)
        ya = np.array(ya, dtype=DTY_INT)  #.tolist()
        yb = np.array(yb, dtype=DTY_INT)  #.tolist()
    if name_div in ['Q_statistic', 'Correlation']:
        vY = np.unique(np.concatenate([y, hi, hj]))
        # if len(vY) == 2 and 0 in vY and 1 in vY:
        #     pass
        # elif len(vY) == 2 and -1 in vY and 1 in vY:
        #     pass
        # else:
        if len(vY) > 2:
            hi = ya.tolist();   hj = yb.tolist()
    #   #
    if name_div == 'Disagreement':
        ans = Disagreement_Measure_multiclass(hi, hj, m)
    elif name_div == 'Q_statistic':
        ans = Q_Statistic_binary(hi, hj)
        # ans = Q_Statistic_binary(ya, yb)
    elif name_div == 'Correlation':
        ans = Correlation_Coefficient_binary(hi, hj)
        # ans = Correlation_Coefficient_binary(ya, yb)
    elif name_div == 'K_statistic':
        ans = Kappa_Statistic_multiclass(hi, hj, y, m)
        ans = ans[0]
    elif name_div == 'Double_fault':
        ans = Double_Fault_Measure_multiclass(hi, hj, y, m)
    else:
        raise UserWarning("LookupError! Check the `name_diver` for pairwise_measure.")
    #
    return ans


def pairwise_measure_overall_value(name_div, yt, y, m, nb_cls):
    ans = 0. 
    for i in range(nb_cls - 1):
        for j in range(i + 1, nb_cls):
            tem = pairwise_measure_for_item(name_div, yt[i], yt[j], y, m)
            ans += tem
    return ans * 2. / (nb_cls * (nb_cls - 1.))



def nonpairwise_measure_overall_value(name_div, yt, y, m, nb_cls):
    if name_div == 'KWVariance':
        ans = Kohavi_Wolpert_Variance_multiclass(yt, y, m, nb_cls)
    elif name_div == 'Interrater':
        ans = Interrater_agreement_multiclass(yt, y, m, nb_cls)
    elif name_div == 'EntropyCC':
        ans = Entropy_cc_multiclass(yt, y, m, nb_cls)
    elif name_div == 'EntropySK':
        ans = Entropy_sk_multiclass(yt, y, m, nb_cls)
    elif name_div == 'Difficulty':
        ans = Difficulty_multiclass(yt, y, nb_cls)
    elif name_div == 'Generalized':
        ans = Generalized_Diversity_multiclass(yt, y, m, nb_cls)
    elif name_div == 'CoinFailure':
        ans = Coincident_Failure_multiclass(yt, y, m, nb_cls)
    else:
        raise UserWarning("LookupError! Check the `name_diver` for non-pairwise measure.")
    return ans


def nonpairwise_measure_for_item(name_div, hi, hj, y, m):
    yt = [hi, hj];  nb_cls = 2
    return nonpairwise_measure_overall_value(name_div, yt, y, m, nb_cls)



def calculate_overall_diversity(name_div, yt, y, m, nb_cls):
    if name_div in ['Disagreement', 'Q_statistic', 'Correlation', 'K_statistic', 'Double_fault']:
        return pairwise_measure_overall_value(name_div, yt, y, m, nb_cls)
    if name_div in ['KWVariance', 'Interrater', 'EntropyCC', 'EntropySK', 'Difficulty', 'Generalized', 'CoinFailure']:
        return nonpairwise_measure_overall_value(name_div, yt, y, m, nb_cls)
    raise UserWarning("LookupError! Check the `name_diver`.")
    return 


def calculate_item_in_diversity(name_div, hi, hj, y, m):
    if name_div in ['Disagreement', 'Q_statistic', 'Correlation', 'K_statistic', 'Double_fault']:
        return pairwise_measure_for_item(name_div, hi, hj, y, m)
    if name_div in ['KWVariance', 'Interrater', 'EntropyCC', 'EntropySK', 'Difficulty', 'Generalized', 'CoinFailure']:
        return nonpairwise_measure_for_item(name_div, hi, hj, y, m)
    raise UserWarning("LookupError! Check the `name_diver`.")
    return 



#-----------------------------------
#
#-----------------------------------



#===================================
#
#===================================


#-----------------------------------
#
#-----------------------------------


