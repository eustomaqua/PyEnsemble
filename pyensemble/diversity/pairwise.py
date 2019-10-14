# coding: utf8
# Aim to: diversity measures in ensembles (existing methods)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from copy import deepcopy
# import time
import numpy as np

from pyensemble.utils_const import DTY_FLT
from pyensemble.utils_const import DTY_INT
from pyensemble.utils_const import check_zero

from pyensemble.diversity.utils_diver import contingency_table
from pyensemble.diversity.utils_diver import multiclass_contingency_table



#-------------------------------------------
# Pairwise Measures [multi-class classification]
#   except `Q-statistic, Correlation-Coefficient` [binary classification]
#-------------------------------------------
#
# m = len(hi) = len(hj)



# Disagreement
# \in [0, 1];   the larger the value, the larger the diversity
#

def Disagreement_Measure_multiclass(hi, hj, m):
    hi = np.array(hi, dtype=DTY_INT)
    hj = np.array(hj, dtype=DTY_INT)
    return np.sum(hi != hj) / float(m)


def Disagreement_Measure_binary(hi, hj, m):
    a, b, c, d = contingency_table(hi, hj)
    return (b + c) / float(m)



# Q statistic
# [only works for binary classification]
# \in [-1, 1];  different / independent (=0) / similar predictions
#

def Q_Statistic_binary(hi, hj):
    a, b, c, d = contingency_table(hi, hj)
    tem = a * d + b * c
    return (a * d + b * c) / check_zero(tem)



# Correlation Coefficient
# [only works for binary classification]
# similar with Q-statistic;     |\rho_{ij}| \geqslant |Q_{ij}|
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
    # m = len(y)  # number of instances / samples
    vY = np.unique(np.concatenate([y, hi, hj]))  # L
    dY = len(vY)
    Cij = multiclass_contingency_table(hi, hj, y)
    Cij = np.array(Cij, dtype=DTY_FLT)
    #
    c_diagonal = [Cij[i, i] for i in range(dY)]
    theta1 = np.sum(c_diagonal) / float(m)
    #
    c_row_sum = [np.prod([Cij[i, i] + Cij[i, j] for j in range(dY) if j!=i]) for i in range(dY)]
    c_col_sum = [np.prod([Cij[i, j] + Cij[j, j] for i in range(dY) if i!=j]) for j in range(dY)]
    theta2 = np.sum(np.multiply(c_row_sum, c_col_sum)) / (float(m) ** 2)
    #
    ans = (theta1 - theta2) / check_zero(1. - theta2)
    return ans, theta1, theta2


def Kappa_Statistic_binary(hi, hj, m):
    a, b, c, d = contingency_table(hi, hj)
    Theta_1 = (a + d) / float(m)
    Theta_2 = ((a + b) * (a + c) + (c + d) * (b + d)) / (float(m) ** 2)
    return (Theta_1 - Theta_2) / check_zero(1. - Theta_2)



# Double-Fault
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


