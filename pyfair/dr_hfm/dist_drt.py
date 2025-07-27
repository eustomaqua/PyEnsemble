# coding: utf-8
# Usage: to calculate the distance directly
#
# Author: Yj
# 1. Does Machine Bring in Extra Bias in Learning? Approximating Fairness
#    in Models Promptly [https://arxiv.org/abs/2405.09251 arXiv]
# 2. Approximating Discrimination Within Models When Faced With Several
#    Non-Binary Sensitive Attributes [https://arxiv.org/abs/2408.06099]
#


import numpy as np
import numba

from pyfair.facil.utils_timer import fantasy_timer


# ==========================================
# Distance between sets


# ------------------------------------------
# Euclidean metric
#
# Directly compute the distance (ext.)
# Si_c: complementary set /supplementary set


# Given the dataset $S=\{(\breve{x}_,\bm{a}_i,y_i) \mid i\in[n]\}$
# composed of instances including sensitive attributes, here we
# denote one instance by $\bm{x}= (\breve{x},\bm{a}) =[x_1,...,x_{n_x},
# a_1,...,a_{n_a}]^\mathsf{T}$ for clarity, where $n_a$ is the number
# of sensitive/protected attributes and $n_x$ is that of unprotected
# attributes in $\bm{x}$.


# ------------------------------------------
# """ Parameters
# n : number of instances in a dataset
# nd: number of non-sensitive features
# na: number of sensitive attributes
# nc: number of classes/labels
#
# X_nA_y: a matrix with size of (n, 1+nd)
#   Non-sensitive attributes of instances and their corresponding
#   labels
# """


@numba.jit(nopython=True)
def DistDirect_Euclidean(ele_i, ele_ic):
    return float(np.linalg.norm(ele_i - ele_ic))


@numba.jit(nopython=True)
def DistDirect_halfway_min(ele_i, Si_c):
    elements = [DistDirect_Euclidean(
        ele_i, ele_ic) for ele_ic in Si_c]
    return min(elements)  # float


@numba.jit(nopython=True)
def DistDirect_mediator(X_nA_y, idx_Si):
    Sj, Sj_c = X_nA_y[idx_Si], X_nA_y[~idx_Si]
    if len(Sj) == 0 or len(Sj_c) == 0:
        return 0., 0.  # default if Sj is an empty set
    elements = [DistDirect_halfway_min(
        ele_i, Sj_c) for ele_i in Sj]
    return max(elements), sum(elements)


# ------------------------------------------
# """ Parameters
# X_nA_y    : a matrix with size of (n, 1+nd)
#
# idx_Si    : an np.ndarray with the size of (n,)
#             whether the corresponding instance belongs to the privileged
#             group (if True) or not.
# idx_Sjs   : a list of np.ndarrays, of which each indicates whether
#             the instance belongs to different subgroups divided by
#             the values of this one sensitive attribute.
# idx_Ai_Sjs: a list of lists, where each element is a list of np.ndarrays
#             Basically, idx_Ai_Sjs = list of [idx_Sjs,], and idx_Si
#             is a special circumstance of idx_Sjs when $a_i$ is
#             bi-valued.
# """


# In face of one sensitive attribute with binary values, that is,
# na=1
#
# For a certain bi-valued sensitive attribute $a_i\in \mathcal{A}_i
# =\{ 0,1\}$, a dataset S can be  $a_i=1$
#
@fantasy_timer
@numba.jit(nopython=True)
def DirectDist_bin(X_nA_y, idx_Si):
    half_1, half_1avg = DistDirect_mediator(X_nA_y, idx_Si)
    half_2, half_2avg = DistDirect_mediator(X_nA_y, ~idx_Si)
    tmp = (half_1avg + half_2avg) / len(X_nA_y)
    return max(half_1, half_2), tmp


# In face of one sensitive attribute with multiple values, in other
# words, let $\bm{a}=[a_i]^\mathsf{T}$ be a single sensitive attribute,
# $n_a=1, a_i\in \mathcal{A}_i =\{1,2,...,n_{a_i}\}, n_{a_i}\geq 3$,
# and $n_{a_i}\in\mathbb{Z}_+$.
#
@fantasy_timer
def DirectDist_nonbin(X_nA_y, idx_Sjs):
    half_mid = [DistDirect_mediator(
        X_nA_y, idx_Si) for idx_Si in idx_Sjs]
    half_pl_max, half_pl_avg = zip(*half_mid)
    n = len(X_nA_y)
    return max(half_pl_max), sum(half_pl_avg) / n


# In face of several sensitive attributes with binary/multiple values,
# that is, na>=2, na\in\mathbb{Z}_+
#
# This is the general case, where we have several sensitive attributes
# $\bm{a}= [a_1,a_2,...,n_{n_a}]^\mathsf{T}$ and each $a_i\in
# \mathcal{A} =\{1,2,...,n_{a_i}\}$, where $n_{a_i}$ is the number of
# values for this sensitive attribute $a_i (1\leq i\leq n_a)$
#
@fantasy_timer
def DirectDist_multiver(X_nA_y, idx_Ai_Sjs):
    half_mid = [DirectDist_nonbin(
        X_nA_y, idx_Sjs) for idx_Sjs in idx_Ai_Sjs]
    half_mid, half_ut = zip(*half_mid)
    half_pl_max, half_pl_avg = zip(*half_mid)
    n_a = len(idx_Ai_Sjs)
    return max(half_pl_max), sum(half_pl_avg) / n_a, (
        half_pl_max, half_pl_avg, half_ut)


# ------------------------------------------
