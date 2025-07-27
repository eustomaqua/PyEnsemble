# coding: utf-8
#
# Target:
#   Existing diversity measures in ensemble learning
#


import numpy as np

from pyfair.facil.utils_const import (
    check_zero, judge_transform_need, DTY_INT)
from pyfair.facil.metric_cont import (
    contingency_tab_bi, contg_tab_mu_type3, contg_tab_mu_type1)
from pyfair.facil.utils_remark import PAIRWISE


# ==================================
#  General
# ==================================
#
# zhou2012ensemble     : binary (multi: self defined)
# kuncheva2003diversity: multiclass
#


# ----------------------------------
# Pairwise Measures
# ----------------------------------
# '''
# contingency_table_binary
#     |         | hi = +1 | hi = -1 |
#     | hj = +1 |    a    |    c    |
#     | hj = -1 |    b    |    d    |
#
# contingency_table_multi
#     |               | hb= c_0 | hb= c_1 | hb= c_{n_c-1} |
#     | ha= c_0       |  C_{00} |  C_{01} |  C_{0?}       |
#     | ha= c_1       |  C_{10} |  C_{11} |  C_{1?}       |
#     | ha= c_{n_c-1} |  C_{?0} |  C_{?1} |  C_{??}       |
#
# contingency_table_multiclass
#     |         | hb == y | hb != y |
#     | ha == y |    a    |    c    |
#     | ha != y |    b    |    d    |
# '''


def contingency_table_binary(hi, hj):  # list
    if len(hi) != len(hj):  # number of instances/samples
        raise AssertionError(
            "These two individual classifiers have different shapes.")
    if len(set(hi + hj)) > 2:
        raise AssertionError("contingency_table works for binary"
                             " classification only.")
    if -1 not in set(hi + hj):  # [0,1], not [-1,1]
        hi = [i * 2 - 1 for i in hi]
        hj = [i * 2 - 1 for i in hj]
    hi = np.array(hi, dtype=DTY_INT)  # 'int')
    hj = np.array(hj, dtype=DTY_INT)  # 'int')
    a, c, b, d = contingency_tab_bi(hi, hj, pos=1)
    return a, b, c, d


def contingency_table_multi(hi, hj, y):
    # dY = len(set(hi + hj + y))
    vY = sorted(set(hi + hj + y))  # list()
    ha, hb = np.array(hi), np.array(hj)
    # construct a contingency table
    return contg_tab_mu_type3(ha, hb, vY)


def contingency_table_multiclass(ha, hb, y):
    # construct a contingency table, Cij
    ha, hb = np.array(ha), np.array(hb)
    a, b, c, d = contg_tab_mu_type1(np.array(y), ha, hb)
    return int(a), int(b), int(c), int(d)


# ==================================
#  Pairwise Measures
#   [multi-class classification]
#   \citep{kuncheva2003diversity}
# ==================================


# ----------------------------------
# $Q$-Statistic
#   \in [-1, 1]
#   different / independent (=0) / similar predictions
#
# Q_ij = zero if hi and hj are independend;
# Q_ij is positive if hi and hj make similar predictions
# Q_ij is negative if hi and hj make different predictions
#

def Q_statistic_multiclass(ha, hb, y):
    a, b, c, d = contingency_table_multiclass(ha, hb, y)
    denominat = a * d + b * c  # 分母, denominator
    numerator = a * d - b * c  # 分子, numerator
    return numerator / check_zero(denominat)


def Q_Statistic_binary(hi, hj):
    a, b, c, d = contingency_table_binary(hi, hj)
    tem = a * d + b * c
    return (a * d - b * c) / check_zero(tem)


# self defined: more research needed
def Q_Statistic_multi(hi, hj, y):
    Cij = contingency_table_multi(hi, hj, y)
    # Cij --> np.ndarray
    # return:
    #   d  c
    #   b  a
    #
    # Cij = np.array(Cij)
    # # axd = np.prod(np.diag(Cij))  # np.diagonal
    mxn = np.shape(Cij)[0]  # mxn = Cij.shape[0]
    axd = [Cij[i][i] for i in range(mxn)]
    bxc = [Cij[i][mxn - 1 - i] for i in range(mxn)]
    axd = np.prod(axd)
    bxc = np.prod(bxc)
    return (axd - bxc) / check_zero(axd + bxc)


# ----------------------------------
# $\kappa$-Statistic
#   \in [-1, 1]?
#   =1, totally agree; =0, agree by chance;
#   <0, rare case, less than expected by chance
#
#   \kappa_p = \frac{ \Theta_1 - \Theta_2 }{ 1 - \Theta_2 }
#   \Theta_1 = \frac{a+d}{m}
#   \Theta_2 = \frac{(a+b)(a+c) + (c+d)(b+d)}{m^2}
#       \Theta_1 \in [0, 1]
#       \Theta_2 \in [0, 1]
#       \kappa_p \in [-1, 1] probably
#

def kappa_statistic_multiclass(ha, hb, y, m):
    a, b, c, d = contingency_table_multiclass(ha, hb, y)
    Theta_1 = (a + d) / float(m)
    numerator = (a + b) * (a + c) + (c + d) * (b + d)
    Theta_2 = numerator / float(m) ** 2
    denominat = 1. - Theta_2
    return (Theta_1 - Theta_2) / check_zero(denominat)


# (\Theta_1 - \Theta_2) \times m^2 = (a+d)(a+b+c+d) - (a+b)(a+c)-(c+d)(b+d)
#     method 1 = (a+b)[a+d-(a+c)] + (c+d)[a+d-(b+d)] = (a+b)(d-c)+(c+d)(a-b)
#              = ad-ac+bd-bc + ac-bc+ad-bd = 2(ad-bc)
#     method 2 = (a+c)[a+d-(a+b)] + (b+d)[a+d-(c+d)] = (a+c)(d-b)+(b+d)(a-c)
#              = ad-ab+cd-bc + ab-bc+ad-cd = 2(ad-bc)
# \Theta_1 - \Theta_2 = 2\frac{ad-bc}{m^2}
#
# (1 - \Theta_2) \times m^2 = m^2 - (a+b)(a+c)-(c+d)(b+d) = (a+b)(b+d)+(c+d)(a+c)
#     method 3 = (a+b+c+d)^2 - (a+b)(a+c) - (c+d)(b+d) = (a+b)(b+d)+(c+d)(a+c)
# 1 - \Theta_2 = \frac{ (a+b)(b+d) + (a+c)(c+d) }{m^2}
#
# \frac{\Theta_1 - \Theta_2}{1- \Theta_2} = 2\frac{ad-bc}{ (a+b)(b+d)+(a+c)(c+d) }
#     denominator = ad+ab+db+b^2 + ad+ac+dc+c^2 = 2ad+(a+d)(b+c)+b^2+c^2
#     numerator   = 2ad-2bc
# definitely, \kappa Statistic \in [-1, 1]?, even more narrow
#

def Kappa_Statistic_binary(hi, hj, m):
    a, b, c, d = contingency_table_binary(hi, hj)
    Theta_1 = float(a + d) / m
    Theta_2 = ((a + b) * (a + c) + (c + d) * (b + d)) / (float(m) ** 2)
    return (Theta_1 - Theta_2) / check_zero(1. - Theta_2)


# self defined: research needed
def Kappa_Statistic_multi(hi, hj, y, m):
    # m = len(y)  # number of instances / samples
    tem = np.concatenate([hi, hj, y])
    _, dY = judge_transform_need(tem)  # vY,
    del tem
    if dY == 1:
        dY = 2
    #   #
    Cij = np.array(contingency_table_multi(hi, hj, y))
    c_diagonal = [Cij[i, i] for i in range(dY)]
    theta1 = np.sum(c_diagonal) / float(m)
    c_row_sum = np.sum(Cij, axis=1)  # rows / float(m)
    c_col_sum = np.sum(Cij, axis=0)  # columns / float(m)
    theta2 = np.sum(c_row_sum * c_col_sum) / float(m) ** 2
    ans = (theta1 - theta2) / check_zero(1. - theta2)
    return ans, theta1, theta2


# ----------------------------------
# Disagreement
#   \in [0, 1]
#   the larger the value, the larger the diversity.
#

def disagreement_measure_multiclass(ha, hb, y, m):
    _, b, c, _ = contingency_table_multiclass(ha, hb, y)
    return (b + c) / float(m)


def Disagreement_Measure_binary(hi, hj, m):
    _, b, c, _ = contingency_table_binary(hi, hj)
    return float(b + c) / m


def Disagreement_Measure_multi(hi, hj, m):
    tem = np.sum(np.not_equal(hi, hj))  # np.sum(hi != hj)
    return float(tem) / m


# ----------------------------------
# Correlation Coefficient
#   \in [-1, 1]
#   |\rho_{ij}| \leqslant |Q_{ij}| with the same sign
#

def correlation_coefficient_multiclass(ha, hb, y):
    a, b, c, d = contingency_table_multiclass(ha, hb, y)
    numerator = a * d - b * c
    denominat = (a + b) * (a + c) * (c + d) * (b + d)
    denominat = np.sqrt(denominat)
    return numerator / check_zero(denominat)


# to see which one of Q_ij and \rho_{ij} is larger, compare
#       a * d + b * c =?= np.sqrt(np.prod([a+b, a+c, c+d, b+d]))
#       a^2d^2+2abcd+b^2c^2 =?= (a^2+ab+ac+bc)(bc+bd+cd+d^2)
#                             = (a^2+bc+ab+ac)(bc+d^2+bd+cd)
#                             = (a^2+bc)(bc+d^2) + ....
#   right= a^2bc+bcd^2+a^2d^2+b^2c^2 + (a^2+bc)(bd+cd)+(ab+ac)(bd+cd)+(ab+ac)(bc+d^2)
#   right-left= bc(a^2+d^2)-2abcd + ....
#             = bc(a-d)^2 +.... >= 0
#   0 <= left <= right
#   1/left >= 1/right
#       therefore, |Q_ij| \geqslant |\rho_ij|
#
#       0 =?= bc(a-d)^2 + ....
#   denominator of Q_ij is smaller, then abs(Q_ij) is larger
#   therefore, it should be |rho_{ij}| \leqslant |Q_{ij}|
#

def Correlation_Coefficient_binary(hi, hj):
    a, b, c, d = contingency_table_binary(hi, hj)
    denominator = (a + b) * (a + c) * (c + d) * (b + d)
    denominator = np.sqrt(denominator)
    return (a * d - b * c) / check_zero(denominator)


# self defined: more research needed
def Correlation_Coefficient_multi(hi, hj, y):
    Cij = np.array(contingency_table_multi(hi, hj, y))
    # list --> np.ndarray:  d  c
    #                       b  a
    mxn = Cij.shape[1]  # 主对角线,反对角线元素
    axd = np.prod([Cij[i, i] for i in range(mxn)])
    bxc = np.prod([Cij[i, mxn - 1 - i] for i in range(mxn)])
    C_row_sum = np.sum(Cij, axis=1)  # sum in the same row
    C_col_sum = np.sum(Cij, axis=0)  # sum in the same column
    denominator = np.multiply(C_col_sum, C_row_sum)  # element-wise
    # denominator = np_prod(denominator.tolist())
    denominator = np.prod(denominator)
    denominator = np.sqrt(denominator)
    return (axd - bxc) / check_zero(denominator)

# 这里发现了一个大 BUG！是 numpy 造成的
# numpy.prod([3886, 4440, 4964]) 结果是不对的
# 它输出为 -251284160，但实际上应为 85648061760
#
# 错因：是超出计算范围了，所以可能会得到 nan 的返回值结果


# ----------------------------------
# Double-Fault
#   \in [0, 1], should be
#

def double_fault_measure_multiclass(ha, hb, y, m):
    _, _, _, e = contingency_table_multiclass(ha, hb, y)
    # m = len(y)  # = a+b+c+d, number of instances
    # e = np.sum(np.logical_and(np.not_equal(ha,y), np.not_equal(hb,y)))
    return int(e) / float(m)


def Double_Fault_Measure_binary_multi(hi, hj, y, m):
    # np.ndarray
    ei = np.not_equal(hi, y)  # hi != y
    ej = np.not_equal(hj, y)  # hj != y
    e = np.sum(ei & ej)
    return float(e) / m


# ----------------------------------


# ----------------------------------
# Pairwise Measure


def pairwise_measure_item_multiclass(name_div, ha, hb, y, m):
    if name_div == "Disag":  # "Disagreement":
        ans = disagreement_measure_multiclass(ha, hb, y, m)
    elif name_div == "QStat":  # "Q_statistic":
        ans = Q_statistic_multiclass(ha, hb, y)
    elif name_div == "Corre":  # "Correlation":
        ans = correlation_coefficient_multiclass(ha, hb, y)
    elif name_div == "KStat":  # "K_statistic":
        ans = kappa_statistic_multiclass(ha, hb, y, m)
    elif name_div == "DoubF":  # "Double_fault":
        ans = double_fault_measure_multiclass(ha, hb, y, m)
    elif name_div not in PAIRWISE.keys():  # .values():
        raise ValueError("Pairwise-Measure doesn't work for"
                         " `name_div` =", name_div)
    return ans


def pairwise_measure_gather_multiclass(name_div, yt, y, m, nb_cls):
    ans = 0.
    for i in range(nb_cls - 1):
        for j in range(i + 1, nb_cls):
            tem = pairwise_measure_item_multiclass(
                name_div, yt[i], yt[j], y, m)
            ans += tem
    return ans * 2. / check_zero(nb_cls * (nb_cls - 1.))


def pairwise_measure_item_binary(name_div, hi, hj, y, m):
    if name_div == "Disagreement":
        ans = Disagreement_Measure_binary(hi, hj, m)
    elif name_div == "Q_statistic":
        ans = Q_Statistic_binary(hi, hj)
    elif name_div == "Correlation":
        ans = Correlation_Coefficient_binary(hi, hj)
    elif name_div == "K_statistic":
        ans = Kappa_Statistic_binary(hi, hj, m)
    elif name_div == "Double_fault":
        ans = Double_Fault_Measure_binary_multi(hi, hj, y, m)
    else:
        raise UserWarning("LookupError! Check the `name_diver`"
                          " for pairwise_measure.")
    return ans


def pairwise_measure_item_multi(name_div, hi, hj, y, m):
    if name_div == "Disagreement":
        ans = Disagreement_Measure_multi(hi, hj, m)
    elif name_div == "Double_fault":
        ans = Double_Fault_Measure_binary_multi(hi, hj, y, m)
    #   #   #
    # three self defined: more research needed
    elif name_div == "Q_statistic":
        ans = Q_Statistic_multi(hi, hj, y)
    elif name_div == "Correlation":
        ans = Correlation_Coefficient_multi(hi, hj, y)
    elif name_div == "K_statistic":
        ans, _, _ = Kappa_Statistic_multi(hi, hj, y, m)
    else:
        raise UserWarning("LookupError! Check the `name_diver`"
                          " for pairwise_measure.")
    return ans


def pairwise_measure_whole_binary(name_div, yt, y, m, nb_cls):
    ans = 0.
    for i in range(nb_cls - 1):
        hi = yt[i]
        for j in range(i + 1, nb_cls):
            hj = yt[j]
            ans += pairwise_measure_item_binary(
                name_div, hi, hj, y, m)
    ans = ans * 2. / check_zero(nb_cls * (nb_cls - 1.))
    return float(ans)


def pairwise_measure_whole_multi(name_div, yt, y, m, nb_cls):
    ans = 0.
    for i in range(nb_cls - 1):
        hi = yt[i]
        for j in range(i + 1, nb_cls):
            hj = yt[j]
            ans += pairwise_measure_item_multi(
                name_div, hi, hj, y, m)
    ans = ans * 2. / check_zero(nb_cls * (nb_cls - 1.))
    return float(ans)
