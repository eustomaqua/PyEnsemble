# coding: utf-8
#
# Target:
#   Existing diversity measures in ensemble learning
#


import numpy as np

from pyfair.facil.utils_const import (
    check_zero, judge_transform_need, DTY_FLT)


# ==================================
#  General
# ==================================
#
# zhou2012ensemble     : binary (multi: self defined)
# kuncheva2003diversity: multiclass
#


# ----------------------------------
# Non-Pairwise Measures
# ----------------------------------


def number_individuals_correctly(yt, y):
    # rho_x = np.sum(yt == y, axis=0)  # np.ndarray
    # return rho_x.copy()  # rho_x
    rho_x = np.sum(np.equal(yt, y), axis=0)
    return rho_x.tolist()


def number_individuals_fall_through(yt, y, nb_cls):
    # failed  # not failing down
    failing = np.sum(np.not_equal(yt, y), axis=0)  # yt!=y
    # nb_cls = len(yt)  # number of individual classifiers
    pi = []
    for i in range(nb_cls + 1):
        tem = np.mean(failing == i)  # np.sum()/m
        pi.append(float(tem))
    return pi


# ==================================
#  Non-Pairwise Measures
#   [multi-class classification]
# ==================================

# m, nb_cls = len(y), len(yt)  # number of instances / individuals
#


# ----------------------------------
# Kohavi-Wolpert Variance
#   the larger the kw value, the larger the diversity
#
# (1) KWVar = \frac{ 1 - \sum_{y\in\mathcal{Y}} \mathbf{P}(y|\mathbf{x})^2 }{2}
# (2) KWVar = \frac{1}{mn^2} \sum_{k=1}^m \rho(\mathbf{x}_k)(n - \rho(\mathbf{x}_k))
#           = \frac{1}{mn^2} \sum_{k=1}^m [-(\rho(\mathbf{x}) -n/2)^2 + n^2/4]
#       because \rho(\mathbf{x}_k) \in [0, n] i.e., [0, T]
#       then -(\rho(\mathbf{x}_k) -n/2)^2 + n^2/4 in [0, n^2/4]
#       therefore KWVar \in [0, 1/4]
#

def Kohavi_Wolpert_variance_multiclass(yt, y, m, nb_cls):
    rho_x = number_individuals_correctly(yt, y)
    numerator = np.multiply(rho_x, np.subtract(nb_cls, rho_x))
    denominat = nb_cls ** 2 * float(m)
    return np.sum(numerator) / denominat


# KWVar = \frac{1}{mT^2} * \sum_{k\in[1,m]} -[\rho(x_k) - T/2]^2 + T^2/4
#       because of \rho(x_k) \in [0, T]
#       then \sum_{k}-[]^2+T^2/4 \in [0, T^2/4] \times m
#       thus KWVar \in [0, 1/4]
#


# ----------------------------------
# Inter-rater agreement
#   =1, totally agree; \leqslant 0, even less than what is expected by chance
#
# (1) numerator = \frac{1}{n} \sum_{k=1}^m \rho(x_k)(n - \rho(x_k))
# (2) denominator = m(n-1) \bar{p}(1 - \bar{p})
#     where \bar{p} = \frac{1}{mn} \sum_{i=1}^n \sum_{k=1}^m \mathbb{I}(h_i(x_k) = y_k)
# (3) \kappa = 1 - \frac{numerator}{denominator}
#
#   \rho(x_k)(n-\rho(x_k)) = -(\rho(x_k) - n/2)^2 + n^2/4 \in [0, n^2/4]
#   \bar{p} = \frac{1}{mn} \sum_{k=1}^m \rho(x_k) \in [0,1] since \rho(x_k) \in [0,n]
#   \bar{p}(1-\bar{p}) = -(\bar{p}-1/2)^2+1/4 \in [0, 1/4]
#       denominator \in m(n-1) [0, 1/4] i.e., [0, m(n-1)/4]
#       numerator \in m/n [0, n^2/4] i.e., [0, mn/4]
#       \frac{numerator}{denominator} \in [0, +\infty)
#       \kappa \in (-\infty, 1]
#

def interrater_agreement_multiclass(yt, y, m, nb_cls):
    rho_x = number_individuals_correctly(yt, y)
    p_bar = np.sum(rho_x) / float(m * nb_cls)
    numerator = np.multiply(rho_x, np.subtract(nb_cls, rho_x))
    numerator = np.sum(numerator) / float(nb_cls)
    denominat = m * (nb_cls - 1.) * p_bar * (1. - p_bar)
    return 1. - numerator / check_zero(denominat)


# Interrater agreement
#
# numerator \in \frac{1}{T} \times m \times [0, T^2/4] = \frac{m}{T} \times [0, T^2/4]
#     \bar{p}            \in \frac{1}{mT} [0, Tm] = [0, 1]
#     \bar{p}{1-\bar{p}} = -[\bar{p} - 1/2]^2+1/4 \in [0, 1/4]
# denominator \in m(T-1) [0, 1/4]
# \frac{numerator}{denominator} = \frac{1}{m(T-1)} [0, T^2/4] / [0, 1/4]
#                                          [0, +inf), [T^2, +inf)
#                              ~= \frac{1}{m(T-1)} [0, +inf) = [0, +inf)
# 1-\frac{numerator}{denominator} \in (-inf, 1]
#


# ----------------------------------
# Entropy
#
# Ent_cc= \frac{1}{m}\sum_{k=1}^m \sum_{y\in\mathcal{Y}}
#                                     -\mathbf{P}(y|x_k) \log(\mathbf{P(y|x_k)})
#   where \mathbf{P}(y|\mathbf{x}_k) =
#                        \frac{1}{n}\sum_{i=1}^n \mathbb{I}(h_i(\mathbf{x}) =y)
#
# the calculation doesn't require to know the correctness of individual classifiers.
#

def Entropy_cc_multiclass(yt, y):
    vY = np.concatenate([[y], yt]).reshape(-1)
    vY, _ = judge_transform_need(vY)
    ans = np.zeros_like(y, dtype=DTY_FLT)  # 'float')
    for i in vY:
        P_y_xk = np.mean(np.equal(yt, i), axis=0)  # np.sum(..)/nb_cls
        tem = list(map(check_zero, P_y_xk))
        tem = -1. * P_y_xk * np.log(tem)
        ans += tem
    ans = np.mean(ans)  # np.sum(..)/m
    return float(ans)


# ----------------------------------
# Entropy
#
# Ent_sk = \frac{1}{m}\sum_{k=1}^m \frac{ \min(\rho(x_k), n-\rho(x_k)) }{n-ceil(n/2)}
#   \in [0, 1]
#   the larger the value, the larger the diversity; =0, totally agree
#

def Entropy_sk_multiclass(yt, y, nb_cls):
    rho_x = number_individuals_correctly(yt, y)
    sub_x = np.subtract(nb_cls, rho_x).tolist()
    tmp = list(map(min, rho_x, sub_x))
    denominator = nb_cls - np.ceil(nb_cls / 2.)
    ans = np.mean(tmp) / check_zero(denominator)
    return float(ans)


# ----------------------------------
# Entropy [works for multi-class classification]
#
#     -p*np.log(p) \in [0, a], 0.35 < a < 0.4, if p\in (0, 1]
#     -p*np.log(p)-(1-p)*np.log(1-p) \in [0, a], a<=0.7?, if p\in (0,1] and q=1-p
# Entropy_cc \in \frac{1}{m} m \times \sum_{multi values of p} -p * np.log(p)
#              = \sum_{multi values of p} -p*np.log(p), and \sum_{values of p}p = 1
#            \in [0, 1]
#
#     \rho(x_k) \in [0, T]      T-\rho(x_k) \in [0, T]
#     \min(\rho(x_k), T-\rho(x_k)) \in [0, T/2]
# Entropy_sk \in \frac{1}{m* floor(T/2)} m \times [0, T/2]
#              = \frac{1}{floor(T/2)} [0, T/2] = \frac{1}{floor(T/2)} [0, floor(T/2)]
#            \in [0, 1]
#


# ----------------------------------
# Difficulty
#   the smaller the theta value, the larger the diversity
#
# Uniform distribution x\sim [a,b]
#   Expection E(x) = (a+b)/2
#   Variance D(x) = (b-a)^2/12
#
# [0, 1] --> np.var(x) \in
# Note: this is not an uniform distribution.
#       the number of x taking values may vary
#

def difficulty_multiclass(yt, y):
    X = np.mean(np.equal(yt, y), axis=0)
    ans = np.var(X)
    return float(ans)


# ----------------------------------
# Generalized diversity
#   \in [0, 1]
#   the diversity is minimized when gd=0
#   the diversity is maximized when gd=1
#
# gd = 1 - \frac{p(2)}{p(1)}
# p(2) = \sum_{i=1}^n \frac{i}{n} p_i
# p(1) = \sum_{i=1}^n \frac{i}{n} \frac{i-1}{n-1} p_i
#   p_i \in [0,1], \forall\, i \in {0,1,2,...,n}
#   \frac{i}{n} p_i \in [0,1]
#   \frac{i}{n} \frac{i-1}{n-1} = \frac{ (i-1/2)^2-1/4 }{n(n-1)} \in [0,1]
#   therefore, p(1) \in [0,n], p(2) \in [0,n], p(2)/p(1) \in [0,+\infty)
#           gd = 1- p(2)/p(1) \in (-\infty, 1]
#

def generalized_diversity_multiclass(yt, y, nb_cls):
    pi = number_individuals_fall_through(yt, y, nb_cls)
    p_1, p_2 = 0., 0.
    for i in range(1, nb_cls + 1):
        p_1 += pi[i] * i / float(nb_cls)
        # p_2 += pi[i] * (i * (i - 1.) / nb_cls / (nb_cls - 1.))
        p_2 += pi[i] * (i * (i - 1.) / nb_cls / check_zero(nb_cls - 1.))
    return 1. - p_2 / check_zero(p_1)


#   \frac{i-1}{T-1} \in [0, 1], due to i \in [1, T]
#   \frac{i}{T}   \in [1/T, 1], due to i \in [1, T]
#   0 <= \frac{i}{T} \frac{i-1}{T-1} <= \frac{i}{T} <= 1
#   p_i \in [0, 1], i = {0, 1, ..., T}
# p(2)/p(1) \in [0, 1], then gd \in [0, 1]
#

def Generalized_Diversity_multi(yt, y, m, nb_cls):
    failing = np.sum(np.not_equal(yt, y), axis=0)  # yt!=y
    pi = [-1.]
    for i in range(1, nb_cls + 1):
        tem = np.sum(failing == i) / float(m)
        pi.append(float(tem))
    p_1, p_2 = 0., 0.
    for i in range(1, nb_cls + 1):
        p_1 += pi[i] * i / nb_cls
        p_2 += pi[i] * (i * (i - 1.) / nb_cls / (nb_cls - 1.))
    return 1. - p_2 / check_zero(p_1)


# ----------------------------------
# Coincident failure
#   when all individuals are the same, cfd=0
#   when each one is different from each other, cfd=1
#   \in [0, 1] ?
#
# cfd =| 0 , if p_0 = 1
#      | \frac{1}{1-p_0} \sum_{i=1}^n \frac{n-i}{n-1} p_i
#

def coincident_failure_multiclass(yt, y, nb_cls):
    pi = number_individuals_fall_through(yt, y, nb_cls)
    if pi[0] == 1.:
        return 0.
    if pi[0] < 1.:
        ans = 0.
        for i in range(1, nb_cls + 1):
            # ans += pi[i] * (nb_cls - i) / (nb_cls - 1.)
            ans += pi[i] * (nb_cls - i) / check_zero(nb_cls - 1.)
        return ans / check_zero(1. - pi[0])
    return 0.


# ----------------------------------


# ----------------------------------
# Non-Pairwise Measure


def nonpairwise_measure_gather_multiclass(
        name_div, yt, y, m, nb_cls):
    if name_div == "KWVar":  # "KWVariance":
        ans = Kohavi_Wolpert_variance_multiclass(yt, y, m, nb_cls)
    elif name_div == "Inter":  # "Interrater":
        ans = interrater_agreement_multiclass(yt, y, m, nb_cls)
    elif name_div == "EntCC":  # "EntropyCC":
        ans = Entropy_cc_multiclass(yt, y)
    elif name_div == "EntSK":  # "EntropySK":
        ans = Entropy_sk_multiclass(yt, y, nb_cls)
    elif name_div == "Diffi":  # "Difficulty":
        ans = difficulty_multiclass(yt, y)
    elif name_div == "GeneD":  # "Generalized":
        ans = generalized_diversity_multiclass(yt, y, nb_cls)
    elif name_div == "CFail":  # "CoinFailure":
        ans = coincident_failure_multiclass(yt, y, nb_cls)
    else:
        raise ValueError("Non-Pairwise-Measure doesn't work for"
                         " `name_div` =", name_div)
    return ans


def nonpairwise_measure_item_multiclass(name_div, ha, hb, y, m):
    yt = [ha, hb]  # yt = np.vstack([ha, hb])  # nb_cls = 2
    return nonpairwise_measure_gather_multiclass(
        name_div, yt, y, m, 2)
