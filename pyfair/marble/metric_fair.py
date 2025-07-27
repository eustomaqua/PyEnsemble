# coding: utf-8
#
# TARGET:
#   Oracle bounds regarding fairness for weighted vote
#   fairness in manifolds and its extension


import numpy as np
import numba

from pyfair.facil.utils_timer import fantasy_timer
from pyfair.facil.utils_const import (
    check_zero, judge_transform_need, DTY_INT)
from pyfair.facil.metric_cont import contingency_tab_bi
# from fairml.facils.metric_cont import (
#     contg_tab_mu_type3, contg_tab_mu_merge)


# =====================================
# Oracle bounds for fairness


# -------------------------------------
# Oracle bounds (previous version)
# hfm/metrics/fairness_gr(ou)p.py


# # Contingency table
# '''
# marginalised groups
# |      | h(xneg,gzero)=1 | h(xneg,gzero)=0 |
# | y= 1 |    TP_{gzero}   |    FN_{gzero}   |
# | y= 0 |    FP_{gzero}   |    TN_{gzero}   |
# privileged group
# |      | h(xneg,gones)=1 | h(xneg,gones)=0 |
# | y= 1 |    TP_{gones}   |    FN_{gones}   |
# | y= 0 |    FP_{gones}   |    TN_{gones}   |
#
# instance (xneg,xpos) --> (xneg,xqtb)
#         xpos might be `gzero` or `gones`
#
# C_{ij}
# |     | hx=0 | hx=1 | ... | hx=? |
# | y=0 | C_00 | C_01 | ... | C_0* |
# | y=1 | C_10 | C_11 |     | C_1* |
# | ... | ...  | ...  |     | ...  |
# | y=? | C_*0 | C_*1 | ... | C_*? |
# '''
# # y, hx: list of scalars (as elements)


def marginalised_contingency(y, hx, vY, dY):
    assert len(y) == len(hx), "Shapes do not match."
    Cij = np.zeros(shape=(dY, dY), dtype=DTY_INT)
    for i in range(dY):
        for j in range(dY):
            tmp = np.logical_and(
                np.equal(y, vY[i]), np.equal(hx, vY[j]))
            Cij[i, j] = np.sum(tmp).tolist()  # int()
            # Cij[i, j] = int(np.sum(tmp))
    return Cij  # np.ndarray


@numba.jit(nopython=True)
def marginalised_confusion(Cij, loc=1):
    Cm = np.zeros((2, 2), dtype=DTY_INT)
    # loca = vY.index(pos)  # [[TP,FN],[FP,TN]]

    Cm[0, 0] = Cij[loc, loc]
    Cm[0, 1] = np.sum(Cij[loc]) - Cij[loc, loc]
    Cm[1, 0] = np.sum(Cij[:, loc]) - Cij[loc, loc]

    # Cm[1, 1] = np.sum(Cij[:loca, :, loca])
    Cm[1, 1] = (np.sum(Cij) + Cij[loc, loc]
                - np.sum(Cij[loc]) - np.sum(Cij[:, loc]))
    return Cm  # np.ndarray


def marginalised_pd_mat(y, hx, pos=1, idx_priv=tuple()):
    if not isinstance(idx_priv, (list, np.ndarray)):
        idx_priv = np.array(idx_priv)  # list()
    # y : not pd.DataFrame, is pd.core.series.Series
    # hx: not pd.DataFrame, is np.ndarray
    # tmp = y.to_numpy().tolist() + hx.tolist()
    if isinstance(y, list) or isinstance(hx, list):
        y, hx = np.array(y), np.array(hx)

    # y : np.ndarray, =pd.DataFrame.to_numpy().reshape(-1)
    # hx: np.ndarray
    tmp = y.tolist() + hx.tolist()
    vY, _ = judge_transform_need(tmp)
    dY = len(vY)

    gones_y_ = y[idx_priv].tolist()
    gzero_y_ = y[np.logical_not(idx_priv)].tolist()
    gones_hx = hx[idx_priv].tolist()
    gzero_hx = hx[np.logical_not(idx_priv)].tolist()

    g1_Cij = marginalised_contingency(gones_y_, gones_hx, vY, dY)
    g0_Cij = marginalised_contingency(gzero_y_, gzero_hx, vY, dY)
    loca = vY.index(pos)

    gones_Cm = marginalised_confusion(g1_Cij, loca)
    gzero_Cm = marginalised_confusion(g0_Cij, loca)
    # gones_Cm:  for privileged group
    # gzero_Cm:  for marginalised groups
    return g1_Cij, g0_Cij, gones_Cm, gzero_Cm  # np.ndarray


# '''
# def marginalised_split_up(y, hx, priv=1, sen=list()):
#     gones_y_ = [i for i, j in zip(y, sen) if j == priv]
#     gzero_y_ = [i for i, j in zip(y, sen) if j != priv]
#     gones_hx = [i for i, j in zip(hx, sen) if j == priv]
#     gzero_hx = [i for i, j in zip(hx, sen) if j != priv]
#     return gones_y_, gzero_y_, gones_hx, gzero_hx
#     # return gones_y_,gones_hx,gzero_y_,gzero_hx  # sens
#
#
# def marginalised_matrixes(y, hx, pos=1, priv=1, sens=list()):
#     """ y, hx: list, shape=(N,), true label and prediction
#     pos : which label is viewed as positive, might be multi-class.
#     sens: which group these instances are from, including one priv-
#           ileged group and one/multiple marginalised group.
#           or list of boolean (as elements)
#     priv: which one indicates the privileged group.
#     """
#     vY, _ = judge_transform_need(y + hx)
#     dY = len(vY)
#
#     gones_y_, gzero_y_, gones_hx, gzero_hx \
#         = marginalised_split_up(y, hx, priv, sens)
#     g1_Cij = marginalised_contingency(gones_y_, gones_hx, vY, dY)
#     g0_Cij = marginalised_contingency(gzero_y_, gzero_hx, vY, dY)
#
#     loca = vY.index(pos)  # [[TP,FN],[FP,TN]]
#     gones_Cm = marginalised_confusion(g1_Cij, loca)
#     gzero_Cm = marginalised_confusion(g0_Cij, loca)
#     return g1_Cij, g0_Cij, gones_Cm, gzero_Cm  # np.ndarray
# '''


# 1) Demographic parity
# aka. (TP+FP)/N = P[h(x)=1]

def prev_unpriv_grp_one(gones_Cm, gzero_Cm):
    N1 = np.sum(gones_Cm)
    N0 = np.sum(gzero_Cm)
    N1 = check_zero(N1.tolist())
    N0 = check_zero(N0.tolist())
    g1 = (gones_Cm[0, 0] + gones_Cm[1, 0]) / N1
    g0 = (gzero_Cm[0, 0] + gzero_Cm[1, 0]) / N0
    return float(g1), float(g0)


# 2) Equality of opportunity
# aka. TP/(TP+FN) = recall
#                 = P[h(x)=1, y=1 | y=1]

def prev_unpriv_grp_two(gones_Cm, gzero_Cm):
    t1 = gones_Cm[0, 0] + gones_Cm[0, 1]
    t0 = gzero_Cm[0, 0] + gzero_Cm[0, 1]
    g1 = gones_Cm[0, 0] / check_zero(t1)
    g0 = gzero_Cm[0, 0] / check_zero(t0)
    return float(g1), float(g0)


# 3) Predictive (quality) parity
# aka. TP/(TP+FP) = precision
#                 = P[h(x)=1, y=1 | h(x)=1]

def prev_unpriv_grp_thr(gones_Cm, gzero_Cm):
    t1 = gones_Cm[0, 0] + gones_Cm[1, 0]
    t0 = gzero_Cm[0, 0] + gzero_Cm[1, 0]
    g1 = gones_Cm[0, 0] / check_zero(t1)
    g0 = gzero_Cm[0, 0] / check_zero(t0)
    return float(g1), float(g0)


# Assume different groups have the same potential
# aka. (TP+FN)/N = P[y=1]

def prev_unpriv_unaware(gones_Cm, gzero_Cm):
    # aka. prerequisite
    N1 = np.sum(gones_Cm)
    N0 = np.sum(gzero_Cm)
    N1 = check_zero(N1.tolist())
    N0 = check_zero(N0.tolist())
    g1 = (gones_Cm[0, 0] + gones_Cm[0, 1]) / N1
    g0 = (gzero_Cm[0, 0] + gzero_Cm[0, 1]) / N0
    return float(g1), float(g0)


# 在无意识前提下，分类
# aka. (TP+FP)/N = P[h(x)=1]
# def unpriv_prereq(gones_Cm, gzero_Cm):


# Self-defined using accuracy
# aka. (TP+TN)/N = P[h(x)=y]

def prev_unpriv_manual(gones_Cm, gzero_Cm):
    N1 = np.sum(gones_Cm)
    N0 = np.sum(gzero_Cm)
    N1 = check_zero(N1.tolist())
    N0 = check_zero(N0.tolist())
    g1 = (gones_Cm[0, 0] + gones_Cm[1, 1]) / N1
    g0 = (gzero_Cm[0, 0] + gzero_Cm[1, 1]) / N0
    return float(g1), float(g0)


# =====================================
# Fairness manifolds and its extension


# -------------------------------------
# Fairness research & oracle bounds
# hfm/metrics/fair_grp_ext.py


class _elem:
    @staticmethod
    def _indices(vA, idx, ex):
        tmp = list(range(len(vA)))
        tmp.remove(idx)
        n = sum(ex) - ex[idx]
        return tmp, n, sum(ex)  # =nt


def zero_division(dividend, divisor):
    # divided_by_zero
    if divisor == 0 and dividend == 0:
        return 0.
    elif divisor == 0:
        return 10.  # return 1.
    return dividend / divisor


def marginalised_np_mat(y, y_hat, pos_label=1,
                        priv_idx=tuple()):
    if not isinstance(priv_idx, (list, np.ndarray)):
        priv_idx = np.array(priv_idx)  # default:list()
    if isinstance(y, list) or isinstance(y_hat, list):
        y, y_hat = np.array(y), np.array(y_hat)

    g1_y = y[priv_idx]  # idx_priv]
    g0_y = y[~priv_idx]
    g1_hx = y_hat[priv_idx]
    g0_hx = y_hat[~priv_idx]

    g1_Cm = contingency_tab_bi(g1_y, g1_hx, pos_label)
    g0_Cm = contingency_tab_bi(g0_y, g0_hx, pos_label)
    # g1_Cm: for the privileged group
    # g0_Cm: for marginalised group(s)
    return g1_Cm, g0_Cm


def marginalised_np_gen(y, y_hat, A, priv_val=1,
                        pos_label=1):
    # # if 0 in A and len(set(A)) == 2:
    # if (0 in A) and (len(set(A)) == 2):
    #     vA = list(set(A))[:: -1]
    #     idx = vA.index(priv_val)
    #     g_y = [y[A == i] for i in vA]
    #     g_hx = [y_hat[A == i] for i in vA]
    #     g_Cm = [contingency_tab_bi(
    #         i, j, pos_label) for i, j in zip(g_y, g_hx)]
    #     ex = [sum(A == i) for i in vA]
    #     return g_Cm, vA, idx, ex
    #
    # vA = sorted(set(A))
    # idx = vA.index(priv_val)
    # g_y = [y[A == i] for i in vA]
    # g_hx = [y_hat[A == i] for i in vA]
    # gs_Cm = [contingency_tab_bi(
    #     i, j, pos_label) for i, j in zip(g_y, g_hx)]
    # ex = [sum(A == i) for i in vA]
    # return gs_Cm, vA, idx, ex

    if (0 in A) and (len(set(A)) == 2):
        vA = list(set(A))[:: -1]
    else:
        vA = sorted(set(A))
    idx = vA.index(priv_val)
    g_y = [y[A == i] for i in vA]
    g_hx = [y_hat[A == i] for i in vA]
    gs_Cm = [contingency_tab_bi(
        i, j, pos_label) for i, j in zip(g_y, g_hx)]
    ex = [sum(A == i) for i in vA]
    return gs_Cm, vA, idx, ex


# # Group fairness (measures)
# ''' Cm
# |        | hx= pos | hx= neg |
# | y= pos |    TP   |    FN   |
# | y= neg |    FP   |    TN   |
# '''
# # return tp, fp, fn, tn


# 1) demographic parity
# 人口统计均等
# aka. (TP+FP)/N = P[h(x)=1]

def unpriv_group_one(g1_Cm, g0_Cm):
    n1 = check_zero(sum(g1_Cm))
    n0 = check_zero(sum(g0_Cm))
    g1 = (g1_Cm[0] + g1_Cm[1]) / n1
    g0 = (g0_Cm[0] + g0_Cm[1]) / n0
    return float(g1), float(g0)


# 2) equality of opportunity
# 胜率均等
# aka. TP/(TP+FN) = recall
#                 = P[h(x)=1, y=1 | y=1]

def unpriv_group_two(g1_Cm, g0_Cm):
    t1 = g1_Cm[0] + g1_Cm[2]
    t0 = g0_Cm[0] + g0_Cm[2]
    g1 = g1_Cm[0] / check_zero(t1)
    g0 = g0_Cm[0] / check_zero(t0)
    return float(g1), float(g0)


# 3) predictive quality parity
# 预测概率均等
# 3) predictive parity
# aka. TP/(TP+FP) = precision
#                 = P[h(x)=1, y=1 | h(x)=1]

def unpriv_group_thr(g1_Cm, g0_Cm):
    t1 = g1_Cm[0] + g1_Cm[1]
    t0 = g0_Cm[0] + g0_Cm[1]
    g1 = g1_Cm[0] / check_zero(t1)
    g0 = g0_Cm[0] / check_zero(t0)
    return float(g1), float(g0)


def calc_fair_group(g1, g0):
    # aka. def group_fair()
    return abs(g1 - g0)


# 假设不同群体成员具有同样的工作潜能
# aka. (TP+FN)/N = P[y=1]

def unpriv_unaware(g1_Cm, g0_Cm):
    # aka. prerequisite
    n1 = check_zero(sum(g1_Cm))
    n0 = check_zero(sum(g0_Cm))
    g1 = (g1_Cm[0] + g1_Cm[2]) / n1
    g0 = (g0_Cm[0] + g0_Cm[2]) / n0
    return float(g1), float(g0)


# 自定义 = accuracy 准确度
# aka. (TP+TN)/N = P[h(x)=y]

def unpriv_manual(g1_Cm, g0_Cm):
    n1 = check_zero(sum(g1_Cm))
    n0 = check_zero(sum(g0_Cm))
    g1 = (g1_Cm[0] + g1_Cm[3]) / n1
    g0 = (g0_Cm[0] + g0_Cm[3]) / n0
    return float(g1), float(g0)


# -------------------------------------
# Fairness research & oracle bounds (cont.)
# Fairness manifold + extension

# Note that y|hx is np.ndarray


@fantasy_timer
def DPext_alterSP(y, hx, idx_Sjs, pos_label=1):
    # item = [np.mean(hx[idx] == pos_label) for idx in idx_Sjs]
    # item = np.nan_to_num(item).tolist()  # Don't use list()
    item = [hx[idx] == pos_label for idx in idx_Sjs]
    # item = [0. if i.shape[0] == 0 else np.mean(i) for i in item]
    item = [0. if not i.shape[0] else float(np.mean(i)) for i in item]

    n_aj = len(idx_Sjs)
    intermediate = []
    for i in range(n_aj - 1):
        for j in range(i + 1, n_aj):
            intermediate.append(abs(item[i] - item[j]))
    # pdb.set_trace()
    return max(intermediate), float(
        np.mean(intermediate)), intermediate


@fantasy_timer
def StatsParity_sing(hx, idx_Sjs, pos=1):
    total = np.mean(hx == pos)

    # item = [np.mean(hx[idx] == pos) for idx in idx_Sjs]
    # item = np.nan_to_num(item).tolist()  # for robustness
    item = [hx[idx] == pos for idx in idx_Sjs]
    item = [np.mean(i) if i.shape[0] else 0. for i in item]

    elements = [float(np.abs(i - total)) for i in item]
    n_ai = len(idx_Sjs)
    return max(elements), sum(elements) / n_ai


@fantasy_timer
def StatsParity_mult(hx, idx_Ai_Sjs, pos=1):
    half_mid = [StatsParity_sing(
        hx, idx_Sjs) for idx_Sjs in idx_Ai_Sjs]
    half_mid, half_ut = zip(*half_mid)
    half_pl_max, half_pl_avg = zip(*half_mid)
    n_a = len(idx_Ai_Sjs)
    return max(half_pl_max), sum(half_pl_avg) / float(n_a), (
        half_pl_max, half_pl_avg, half_ut)


@fantasy_timer
def extGrp1_DP_sing(y, hx, idx_Sjs, pos=1):
    # total = np.mean(hx == pos)
    # alternative = [np.mean(hx[idx] == pos) for idx in idx_Sjs]
    # if np.isnan(alternative).any():
    #     alternative = np.nan_to_num(alternative).tolist()
    # if np.isnan(total):
    #     total = float(np.nan_to_num(total))

    total = hx == pos
    total = np.mean(total) if total.shape[0] else 0.
    alternative = [hx[idx] == pos for idx in idx_Sjs]
    alternative = [np.mean(
        i) if i.shape[0] else 0. for i in alternative]

    elements = [float(np.abs(i - total)) for i in alternative]
    # if np.isnan(elements).any():
    #     pdb.set_trace()
    n_aj = len(idx_Sjs)
    return max(elements), sum(elements) / n_aj, alternative


@fantasy_timer
def extGrp2_EO_sing(y, hx, idx_Sjs, pos=1):
    idx = y == pos  # renew_y = y[idx]
    renew_hx = hx[idx]
    renew_Sj = [Sj[idx] for Sj in idx_Sjs]
    # total = np.mean(renew_hx == pos)
    # alternative = [np.mean(renew_hx[Sj] == pos) for Sj in renew_Sj]
    #
    # if np.isnan(alternative).any():
    #     alternative = np.nan_to_num(alternative).tolist()
    # if np.isnan(total):
    #     total = float(np.nan_to_num(total))

    total = renew_hx == pos
    total = np.mean(total) if total.shape[0] else 0.
    alternative = [renew_hx[Sj] == pos for Sj in renew_Sj]
    alternative = [np.mean(
        i) if i.shape[0] else 0. for i in alternative]

    elements = [float(np.abs(i - total)) for i in alternative]
    # if np.isnan(elements).any():
    #     pdb.set_trace()
    n_aj = len(idx_Sjs)
    return max(elements), sum(elements) / n_aj, alternative


@fantasy_timer
def extGrp3_PQP_sing(y, hx, idx_Sjs, pos=1):
    idx = hx == pos
    renew_y = y[idx]
    renew_Sj = [Sj[idx] for Sj in idx_Sjs]
    # total = np.mean(renew_y == pos)
    # alternative = [np.mean(renew_y[Sj] == pos) for Sj in renew_Sj]
    #
    # if np.isnan(alternative).any():
    #     alternative = np.nan_to_num(alternative).tolist()
    # if np.isnan(total):  # or total!=total
    #     total = float(np.nan_to_num(total))

    total = renew_y == pos
    total = np.mean(total) if total.shape[0] else 0.
    alternative = [renew_y[Sj] == pos for Sj in renew_Sj]
    alternative = [np.mean(
        i) if i.shape[0] else 0. for i in alternative]

    elements = [float(np.abs(i - total)) for i in alternative]
    n_aj = len(idx_Sjs)
    return max(elements), sum(elements) / n_aj, alternative


@fantasy_timer
def alterGrps_sing(alternative, idx_Sjs):
    ele = list(map(float, alternative))
    n_ai = len(idx_Sjs)
    renewal = [abs(ele[i] - ele[j]) for i in range(
        n_ai) for j in range(i + 1, n_ai)]
    n_aj_prime = len(renewal)
    return max(renewal), sum(renewal) / n_aj_prime


# def alterGroups_pl(half_alter, idx_Ai_Sjs, n_a):
#     half_alter = [alterGrps_sing(alt, idx_Sjs)[
#         0] for alt, idx_Sjs in zip(half_alter, idx_Ai_Sjs)]
#     half_pl_max, half_pl_avg = zip(*half_alter)
#     return max(half_pl_max), sum(half_pl_avg) / n_a
#
#
# @fantasy_timer
# def extGrp1_DP_pl(y, hx, idx_Ai_Sjs, pos=1, alter=False):
#     half_mid = [extGrp1_DP_sing(
#         y, hx, idx_Sjs, pos)[0] for idx_Sjs in idx_Ai_Sjs]
#     half_pl_max, half_pl_avg, half_alter = zip(*half_mid)
#     n_a = len(idx_Ai_Sjs)
#     if not alter:
#         return max(half_pl_max), sum(half_pl_avg) / n_a
#     return alterGroups_pl(half_alter, idx_Ai_Sjs, n_a)
#
#
# @fantasy_timer
# def extGrp2_EO_pl(y, hx, idx_Ai_Sjs, pos=1, alter=False):
#     half_mid = [extGrp2_EO_sing(
#         y, hx, idx_Sjs, pos)[0] for idx_Sjs in idx_Ai_Sjs]
#     half_pl_max, half_pl_avg, half_alter = zip(*half_mid)
#     n_a = len(idx_Ai_Sjs)
#     if not alter:
#         return max(half_pl_max), sum(half_pl_avg) / n_a
#     return alterGroups_pl(half_alter, idx_Ai_Sjs, n_a)
#
#
# @fantasy_timer
# def extGrp3_PQP_pl(y, hx, idx_Ai_Sjs, pos=1, alter=False):
#     half_mid = [extGrp3_PQP_sing(
#         y, hx, idx_Sjs, pos)[0] for idx_Sjs in idx_Ai_Sjs]
#     half_pl_max, half_pl_avg, half_alter = zip(*half_mid)
#     n_a = len(idx_Ai_Sjs)
#     if not alter:
#         return max(half_pl_max), sum(half_pl_avg) / n_a
#     return alterGroups_pl(half_alter, idx_Ai_Sjs, n_a)


# -------------------------------------
#


# =====================================
