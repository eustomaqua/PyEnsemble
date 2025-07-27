# coding: utf-8
# metric_dist.py
# Author: Yijun


import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.optimize as optim
# import cv2

from copy import deepcopy
from pyfair.facil.utils_const import check_zero, non_negative
# from hfm.utils.verifiers import check_zero, non_negative


# =======================================
# 推土机距离
# =======================================
#

# KL散度/相对熵/KL距离，不对称
# 取值范围 [0, +\infty)
def KL_divergence(p, q):
    return stats.entropy(p, q)


# JS散度基于KL散度，对称
# 同样是二者越相似，JS散度越小。
# 取值范围在0-1之间，完全相同时为0
def JS_divergence(p, q):
    m = np.add(p, q) / 2
    tmp_a = stats.entropy(p, m)
    tmp_b = stats.entropy(q, m)
    tmp = 0.5 * tmp_a + 0.5 * tmp_b
    return float(tmp)


# KL-divergence 的坏处在于它是无界的。事实上KL-divergence 属于
# 更广泛的 f-divergence 中的一种。
# 如果P和Q被定义成空间中的两个概率分布，则f散度被定义为

def _f_div(t):
    t = check_zero(t)
    return t * np.log(t)


def f_divergence(p, q):
    # i.e., scipy.stats.entropy(p, q)
    # return np.sum(q * _f_div(p / q))

    ans = 0.
    for i, j in zip(p, q):
        # '''
        # tmp = check_zero(j)
        # tmp = check_zero(i / tmp)
        # # tmp = check_zero(i / j)
        # '''
        tmp = i / check_zero(j)
        ans += j * _f_div(tmp)
    return float(ans)


def Hellinger_dist_v1(p, q):
    tmp = np.sqrt(p) - np.sqrt(q)
    tmp = np.linalg.norm(tmp)
    tmp = tmp / np.sqrt(2)
    return float(tmp)


def Hellinger_dist_v2(p, q):
    # return np.sqrt(1 - np.sum(np.sqrt(p * q)))

    ans = _BC_dis(p, q)
    tmp = non_negative(1. - ans)
    tmp = np.sqrt(tmp)
    return float(tmp)


# 巴氏距离（Bhattacharyya Distance）
# 测量两个离散或连续概率分布的相似性

def _BC_dis(p, q):
    ans = 0.
    for i, j in zip(p, q):
        ans += np.sqrt(i * j)
    return float(ans)


def Bhattacharyya_dist(p, q):
    ans = _BC_dis(p, q)
    tmp = non_negative(ans)
    tmp = -np.log(tmp)
    return float(tmp)


# MMD距离（Maximum mean discrepancy)
# 最大均值差异（Maximum mean discrepancy）
# 度量在再生希尔伯特空间中两个分布的距离，是一种核学习方法


# Wasserstein 距离，也叫Earth Mover's Distance，推土机距离，简称EMD，用来表示两个分布的相似程度。

def Wasserstein_distance(p, q, D):
    # 通过线性规划求Wasserstein距离
    # p.shape: (m,)   p.sum()=1   p\in[0,1]
    # q.shape: (n,)   q.sum()=1   q\in[0,1]
    # D.shape: (m, n)
    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    # D = D.reshape(-1)
    # result = optim.linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    # return result.fun
    d_tmp = np.reshape(D, -1)  # np.reshape(D, [?,-1])
    res = optim.linprog(d_tmp, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    return res.fun


def Wasserstein_dis(P, Q, D=None):
    return stats.wasserstein_distance(P, Q)


# =======================================
# SG
# =======================================


def _discrete_bar_counts(indexes, density=True):
    # input: np.ndarray
    # print("indexes:", indexes)
    idx_tmp = deepcopy(indexes)
    if len(np.shape(idx_tmp)) > 1:
        # idx_tmp = idx_tmp.reshape(-1)
        idx_tmp = np.reshape(idx_tmp, -1)
    mn, mx = min(idx_tmp), max(idx_tmp)
    opt_y = list(range(mn, mx + 1))  # freq_x
    freq_y = [0] * (mx + 1 - mn)
    for i in idx_tmp:
        freq_y[i - mn] += 1
    sm = len(idx_tmp)
    assert sum(freq_y) == sm
    if density:
        freq_y = [float(i) / sm for i in freq_y]
    return opt_y, freq_y  # return freq_x,freq_y


def _discrete_joint_cnts(X, Y, density=True, v=None):
    if v is None:
        mn = min(np.min(X), np.min(Y))
        mx = max(np.max(X), np.max(Y))
        v = list(range(mn, mx + 1))
    else:
        mn, mx = v[0], v[-1]
    px = [0] * (mx + 1 - mn)
    py = [0] * (mx + 1 - mn)
    for i in X:
        px[i - mn] += 1
    for i in Y:
        py[i - mn] += 1
    # assert len(X) == sum(px)
    # assert len(Y) == sum(py)
    if density:
        sx, sy = len(X), len(Y)
        px = [float(i) / sx for i in px]
        py = [float(i) / sy for i in py]
    return px, py, v


def JS_div(arr1, arr2, num_bins=368):
    max0 = max(np.max(arr1), np.max(arr2))
    min0 = min(np.min(arr1), np.min(arr2))
    # bins = np.linspace(min0 - 1e-4, max0 - 1e-4, num=num_bins)
    bins = np.linspace(
        min0 - 1e-4, max0 + 1e-4, num=num_bins + 1)
    PDF1 = pd.cut(arr1, bins).value_counts() / len(arr1)
    PDF2 = pd.cut(arr2, bins).value_counts() / len(arr2)
    return JS_divergence(PDF1.values, PDF2.values)
