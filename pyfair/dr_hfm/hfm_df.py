# coding: utf-8
# Usage: to measure the bias level within one classifier
#
# Author: Yj
# 1. Does Machine Bring in Extra Bias in Learning? Approximating Fairness
#    in Models Promptly [https://arxiv.org/abs/2405.09251 arXiv]
# 2. Approximating Discrimination Within Models When Faced With Several
#    Non-Binary Sensitive Attributes [https://arxiv.org/abs/2408.06099]
#


import numpy as np
from pyfair.facil.utils_timer import fantasy_timer
from pyfair.facil.utils_const import check_zero


# =============================================
# Estimated distance between sets


# ---------------------------------------------
# Algorithm 2
# Algorithm 1


# =============================================
# Harmonic fairness measure via manifolds (HFM)


# ---------------------------------------------
# 1. https://arxiv.org/abs/2405.09251


@fantasy_timer
def bias_degree_bin(Dy_dis, Df_dis):
    if (Dy_dis == 0) and (Df_dis == 0):
        tmp = 1.
    else:
        tmp = Df_dis / check_zero(Dy_dis)
    return tmp - 1.


# ---------------------------------------------
# 2. https://arxiv.org/abs/2408.06099


@fantasy_timer
def bias_degree_nonbin(Dy_dis, Df_dis):
    if (Dy_dis == 0) and (Df_dis == 0):
        tmp = 1.
    else:
        tmp = Df_dis / check_zero(Dy_dis)
    tmp = check_zero(tmp)
    return np.log10(tmp)


# ---------------------------------------------


@fantasy_timer
def bias_degree(Dy_dis, Df_dis):
    if (Dy_dis == 0) and (Df_dis == 0):
        tmp = 1.
    else:
        tmp = Df_dis / check_zero(Dy_dis)
    return tmp - 1., np.log10(check_zero(tmp))
