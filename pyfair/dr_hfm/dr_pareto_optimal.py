# coding: utf-8
# Aim to provide:
#
# TARGET:
#   Oracle bounds regarding fairness for weighted vote
#   PAC bounds
#


from copy import deepcopy
import gc
from itertools import permutations, combinations
import numpy as np
from pathos import multiprocessing as pp

from pyfair.dr_hfm.discriminative_risk import (
    hat_L_fair, hat_L_loss, Erho_sup_L_fair,
    E_rho_L_loss_f, tandem_objt, cal_L_obj_v1)
from pyfair.facil.ensem_voting import weighted_voting
from pyfair.granite.ensem_pruning import _PEP_flipping_uniformly

# Algorithm 4
from pyfair.marble.metric_fair import (
    marginalised_pd_mat, prev_unpriv_unaware, prev_unpriv_manual,
    prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr)
from pyfair.marble.draw_hypos import _Friedman_sequential


gc.enable()

unpriv_group_one = prev_unpriv_grp_one
unpriv_group_two = prev_unpriv_grp_two
unpriv_group_thr = prev_unpriv_grp_thr
unpriv_unaware = prev_unpriv_unaware
unpriv_manual = prev_unpriv_manual
del prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr
del prev_unpriv_unaware, prev_unpriv_manual


# =====================================
# Section 3.3 Ensemble Pruning
# =====================================

# Pareto Optimal Ensemble Pruning via Improving Accuracy and
# Fairness Concurrently (POAF)


# -------------------------------------
# Algorithm 1.
# -------------------------------------


# Domination
# (1) G_A \succeq G_B    aka. GA[0]<=GB[0] and GA[1]<=GB[1]
# (2) G_A \succ G_B      aka. GA[0]< GB[0] and GA[1]<=GB[1]
#                          or GA[0]<=GB[0] and GA[1]< GB[1]
# (3) G_B \succeq G_A    aka. GA[0]>=GB[0] and GA[1]>=GB[1]
# (4) G_B \succ G_A      aka. GA[0]> GB[0] and GA[1]>=GB[1]
#                          or GA[0]>=GB[0] and GA[1]> GB[1]
# special case: G_A==G_B
# G_A \succeq G_B, but not G_A \succ G_B
# G_B \succeq G_A, but not G_B \succ G_A

# GA \succeq GB, GB \succeq GC, then GA \succeq GC
# GA \succ   GB, GB \succ   GC, then GA \succ   GC


def _weakly_dominate(G_1, G_2):
    if G_1[0] <= G_2[0] and G_1[1] <= G_2[1]:
        return True
    return False


def _dominate(G_1, G_2):
    if _weakly_dominate(G_1, G_2):
        if G_1[0] < G_2[0] or G_1[1] < G_2[1]:
            return True
    return False


def _bi_objectives(G, lam=.5):
    # G = (L_fair, L_acc)
    return lam * G[0] + (1. - lam) * G[1]


def _bi_goals_whole(y, zt, zq, wgt, bf_h):
    # aka. G(MVrho) = (L_fair(MV), L_acc(MV))
    #     normal way to calculate for individual members
    # equiv. `hat_L_objt`

    yt = zt[bf_h].tolist()
    yq = zq[bf_h].tolist()
    coef = np.array(wgt)[bf_h].tolist()

    fens_t = weighted_voting(yt, coef)  # y,
    fens_q = weighted_voting(yq, coef)  # y,
    sub_no1 = hat_L_fair(fens_t, fens_q)
    sub_no2 = hat_L_loss(fens_t, y)
    return (sub_no1, sub_no2)


def _bi_goals_split(y, zt, zq, wgt, bf_h):
    # aka. L(MVrho) = lam* +(1-lam)*
    #     specified definition for ensemble classifiers
    # equiv. `cal_L_obj_v1`/`cal_L_obj_v2`

    yt = zt[bf_h].tolist()
    yq = zq[bf_h].tolist()
    coef = np.array(wgt)[bf_h].tolist()
    sub_no1 = Erho_sup_L_fair(yt, yq, coef)
    sub_no2 = E_rho_L_loss_f(yt, y, coef)
    return sub_no1, sub_no2


def _find_argmin_h(y, zt, zq, wgt, cal_V, lam):
    # sub_obj = [_bi_objectives(
    #     y, zt, zq, wgt, bf_v, lam) for bf_v in cal_V]
    # sub_obj = [lam * i + (1. - lam) * j for i, j in sub_obj]
    if not cal_V:
        return -1, []

    sub_obj = [cal_L_obj_v1(zt[bf_v].tolist(),
                            zq[bf_v].tolist(),
                            y,
                            np.array(wgt)[bf_v].tolist(),
                            lam) for bf_v in cal_V]
    idx_v = sub_obj.index(min(sub_obj))
    return idx_v, sub_obj


# Pareto optimal
#   seems like: _find_exist_succ == _setminus_succeq
#               _find_exist_succeq == _setminus_succ


def _setminus_succeq(y, zt, zq, wgt, cal_R, bf_h):
    sub_obj = [_bi_goals_whole(
        y, zt, zq, wgt, bf_r) for bf_r in cal_R]
    sub_H = _bi_goals_whole(y, zt, zq, wgt, bf_h)
    idx_not_ = [_weakly_dominate(sub_H, i) for i in sub_obj]
    idx_in_R = np.where(np.logical_not(idx_not_))[0]
    return idx_in_R.tolist()


def _find_exist_succ(y, zt, zq, wgt, cal_H, bf_hp):
    sub_obj = [_bi_goals_whole(
        y, zt, zq, wgt, bf_h) for bf_h in cal_H]
    sub_Hp = _bi_goals_whole(y, zt, zq, wgt, bf_hp)
    idx_needs = [_dominate(i, sub_Hp) for i in sub_obj]
    idx_needs = np.where(idx_needs)[0]
    return idx_needs.tolist()


def _setminus_succ(y, zt, zq, wgt, cal_R, bf_h):
    sub_obj = [_bi_goals_whole(
        y, zt, zq, wgt, bf_r) for bf_r in cal_R]
    sub_H = _bi_goals_whole(y, zt, zq, wgt, bf_h)
    idx_not_ = [_dominate(sub_H, i) for i in sub_obj]
    idx_in_R = np.where(np.logical_not(idx_not_))[0]
    return idx_in_R.tolist()


def _find_exist_succeq(y, zt, zq, wgt, cal_H, bf_hp):
    sub_obj = [_bi_goals_whole(
        y, zt, zq, wgt, bf_h) for bf_h in cal_H]
    sub_Hp = _bi_goals_whole(y, zt, zq, wgt, bf_hp)
    idx_needs = [_weakly_dominate(i, sub_Hp) for i in sub_obj]
    idx_needs = np.where(idx_needs)[0]
    return idx_needs.tolist()


def _randomly_choose(nb_cls, nb_pru):
    # H = np.zeros(nb_cls, dtype='int').tolist()
    H = np.zeros(nb_cls, dtype='bool').tolist()
    while sum(H) < nb_pru:
        tmp = np.random.choice(range(nb_cls))
        H[tmp] = True
        # H[np.random.choice(range(nb_cls))] = 1
    return H


def _hamming_distance(bf_h, bf_hp):
    # return np.sum(np.not_equal(bf_h, bf_hp)).tolist()

    # bf_h : list, elements of boolean
    # bf_hp: list, elements of boolean
    return sum(bf_hp) - sum(bf_h)


def _neighbour_sets(bf_h, nb_cls):
    # bf_h: list, element {0,1}
    # cal_N_neg, cal_N_pos = [], []
    idx_in_H = np.where(bf_h)[0].tolist()
    idx_not_ = np.where(np.logical_not(bf_h))[0].tolist()

    N_neg, N_pos = [], []
    for i in bf_h:
        if i:
            N_neg.append(deepcopy(bf_h))
        else:
            N_pos.append(deepcopy(bf_h))

    # mathcal{N}_-(H)
    # N_neg = [deepcopy(bf_h) for i in bf_h if i]
    if len(idx_in_H) <= 1:
        N_neg = []
    for hs, k in zip(N_neg, idx_in_H):
        hs[k] = False  # 0

    # mathcal{N}_+(H)
    # N_pos = [deepcopy(bf_h) for i in bf_h if not i]
    if len(idx_in_H) >= nb_cls:
        N_pos = []
    for hs, k in zip(N_pos, idx_not_):
        hs[k] = True  # 1

    return N_neg + N_pos  # list,(nb_cls,nb_cls)


def neighbour_neg(bf_h, nb_cls, dist=1):
    idx = np.where(bf_h)[0].tolist()
    if len(idx) <= 1:
        return [], list()  # tuple()
    elif dist >= len(idx):
        return [], list()  # tuple()

    # N_neg = [deepcopy(bf_h) for i in bf_h if i]
    # loc = list(permutations(idx, dist))
    loc = list(combinations(idx, dist))

    N_neg = [deepcopy(bf_h) for _ in loc]
    for k, tmp in enumerate(loc):
        for v in tmp:
            N_neg[k][v] = False
    return N_neg, loc


def neighbour_pos(bf_h, nb_cls, dist=1):
    idx = np.where(bf_h)[0].tolist()
    if len(idx) >= nb_cls:
        return [], list()
    elif nb_cls - len(idx) <= dist:
        return [], list()

    # N_pos = [deepcopy(bf_h) for i in bf_h if not i]
    idx = np.where(np.logical_not(bf_h))[0].tolist()
    # loc = list(permutations(idx, dist))
    loc = list(combinations(idx, dist))

    N_pos = [deepcopy(bf_h) for _ in loc]
    for k, tmp in enumerate(loc):
        for v in tmp:
            N_pos[k][v] = True
    return N_pos, loc


def find_nbor_neg(bf_h, nb_cls, dist=2):
    T_neg, _ = neighbour_neg(bf_h, nb_cls, dist)

    cal_T_neg = []
    for bf_r in T_neg:
        tmp, _ = neighbour_pos(bf_r, nb_cls, dist)
        # idx_in_T = [i == bf_h for i in tmp]
        idx_in_T = [i not in cal_T_neg for i in tmp]
        tmp = [tmp[k] for k, i in enumerate(idx_in_T) if i]
        cal_T_neg.extend(tmp)

    return cal_T_neg


def find_nbor_pos(bf_h, nb_cls, dist=2):
    T_pos, _ = neighbour_pos(bf_h, nb_cls, dist)

    cal_T_pos = []
    for bf_r in T_pos:
        tmp, _ = neighbour_neg(bf_r, nb_cls, dist)
        idx_in_T = [i not in cal_T_pos for i in tmp]
        tmp = [tmp[k] for k, i in enumerate(idx_in_T) if i]
        cal_T_pos.extend(tmp)

    return cal_T_pos


# Ensemble pruning


def _pareto_sub_neighbor_v1(bf_h, nb_cls):
    cal_R = _neighbour_sets(bf_h, nb_cls)

    cal_T = []
    for bf_r in cal_R:
        tmp = _neighbour_sets(bf_r, nb_cls)
        idx_in_T = [i not in cal_T for i in tmp]
        tmp = [tmp[k] for k, i in enumerate(idx_in_T) if i]
        cal_T.extend(tmp)
    # cal_T: list, at most (nb_cls**2) elements

    idx_in_T = [i not in cal_R for i in cal_T]
    cal_T = [cal_T[k] for k, i in enumerate(idx_in_T) if i]
    cal_R.extend(cal_T)
    return cal_R


def _pareto_sub_neighbor_v2(bf_h, nb_cls, dist=1):
    cal_R = [deepcopy(bf_h)]
    # cal_T_neg, cal_T_pos = [], []
    cal_T_neg = find_nbor_neg(bf_h, nb_cls, dist)
    cal_T_pos = find_nbor_pos(bf_h, nb_cls, dist)

    idx_in_T = [i not in cal_R for i in cal_T_neg]
    cal_T_neg = [cal_T_neg[k] for k, i in enumerate(idx_in_T) if i]
    cal_R.extend(cal_T_neg)
    idx_in_T = [i not in cal_R for i in cal_T_pos]
    cal_T_pos = [cal_T_pos[k] for k, i in enumerate(idx_in_T) if i]
    cal_R.extend(cal_T_pos)
    return cal_R


def _pareto_sub_neighbor_v5(bf_h, nb_cls):
    cal_R = [deepcopy(bf_h)]
    T_neg = find_nbor_neg(bf_h, nb_cls, 1)
    T_pos = find_nbor_pos(bf_h, nb_cls, 1)

    cal_T_neg, cal_T_pos = [], []
    for bf_r in T_neg:
        tmp = find_nbor_pos(bf_r, nb_cls, 1)
        idx_in_T = [i not in cal_T_neg for i in tmp]
        tmp = [tmp[k] for k, i in enumerate(idx_in_T) if i]
        cal_T_neg.extend(tmp)

        tmp = find_nbor_neg(bf_r, nb_cls, 1)
        idx_in_T = [i not in cal_T_neg for i in tmp]
        tmp = [tmp[k] for k, i in enumerate(idx_in_T) if i]
        cal_T_neg.extend(tmp)

    for bf_r in T_pos:
        tmp = find_nbor_neg(bf_r, nb_cls, 1)
        idx_in_T = [i not in cal_T_pos for i in tmp]
        tmp = [tmp[k] for k, i in enumerate(idx_in_T) if i]
        cal_T_pos.extend(tmp)

        tmp = find_nbor_pos(bf_r, nb_cls, 1)
        idx_in_T = [i not in cal_T_pos for i in tmp]
        tmp = [tmp[k] for k, i in enumerate(idx_in_T) if i]
        cal_T_pos.extend(tmp)

    idx_in_T = [i not in cal_R for i in cal_T_neg]
    cal_T_neg = [cal_T_neg[k] for k, i in enumerate(idx_in_T) if i]
    cal_R.extend(cal_T_neg)
    idx_in_T = [i not in cal_R for i in cal_T_pos]
    cal_T_pos = [cal_T_pos[k] for k, i in enumerate(idx_in_T) if i]
    cal_R.extend(cal_T_pos)
    return cal_R


def find_index_in_V(cal_R):
    li = list(map(lambda x: np.where(x)[0].tolist(), cal_R))
    # return list(map(tuple, li))
    return sorted(map(tuple, li))


def _pareto_sub_neighbor_v3a(bf_h, nb_cls, nb_pru):
    idx = list(range(nb_cls))
    perm = list(permutations(idx, nb_pru))
    cal_V = [tuple(sorted(i)) for i in perm]
    # cal_V = list(set(cal_V))
    cal_V = list(sorted(set(cal_V)))
    cal_R = [np.zeros(
        nb_cls, dtype='bool').tolist() for _ in range(len(cal_V))]
    for k, tmp in enumerate(cal_V):
        for v in tmp:
            cal_R[k][v] = True
    idx_in_R = [i == bf_h for i in cal_R]
    return [cal_R[k] for k, i in enumerate(idx_in_R) if not i]


def _pareto_sub_neighbor_v3b(bf_h, nb_cls, nb_pru):
    idx = list(range(nb_cls))
    comb = list(combinations(idx, nb_pru))
    cal_R = [np.zeros(
        nb_cls, dtype='bool').tolist() for _ in range(len(comb))]
    for k, tmp in enumerate(comb):
        for v in tmp:
            cal_R[k][v] = True
    idx_in_R = [i == bf_h for i in cal_R]
    return [cal_R[k] for k, i in enumerate(idx_in_R) if not i]


def _pareto_sub_moveout_v4(y, zt, zq, wgt, cal_R):
    cal_H = []
    while cal_R:
        # i_hp = np.random.choice(range(len(cal_R)))
        i_hp = np.random.randint(len(cal_R))
        bf_hp = cal_R[i_hp]
        del cal_R[i_hp]

        idx_in_Q = _find_exist_succ(y, zt, zq, wgt, cal_R, bf_hp)
        if not idx_in_Q:
            cal_H.append(deepcopy(bf_hp))
        idx_in_R = _setminus_succ(y, zt, zq, wgt, cal_R, bf_hp)
        cal_R = [cal_R[i] for i in idx_in_R]
    return cal_H


def Pareto_Optimal_EPAF_Pruning(y, yt, yq, wgt, nb_pru, lam, dist=1):
    # default: dist=2
    nb_cls = len(wgt)
    zt, zq = np.array(yt), np.array(yq)
    bf_h = _randomly_choose(nb_cls, nb_pru)
    # cal_H = [deepcopy(bf_h)]
    cal_R = _pareto_sub_neighbor_v2(bf_h, nb_cls, dist)
    cal_H = _pareto_sub_moveout_v4(y, zt, zq, wgt, cal_R)
    if not cal_H:
        return np.where(bf_h).tolist()
    idx_v, _ = _find_argmin_h(y, zt, zq, wgt, cal_H, lam)  # ,sub_obj
    H = cal_H[idx_v]
    return np.where(H)[0].tolist()


# =====================================
# Appendix B.
# TWO EXTRA EASILY-IMPLEMENTED PRUNING METHODS
# =====================================

# Ensemble Pruning via Improving Accuracy and Fairness
# Concurrently aka. EPAF (with Centralised/Distributed ver.)


# -------------------------------------
# Algorithm 2.
#     Centralised Version (EPAF-C)
# -------------------------------------

# 1. H= an arbitrary individual member f_i\in F
# 2. for i=2 to k do
# 3.    f*= argmin_{f_i\in F\H} sum_{f_j\in H} hat_L(fi,fj,S)
# 4.    Move f* from F to H
# 5. end for


def _tandem_obj_sum(fa, fa_q, Fb, Fb_q, wgt, y, lam):
    ans = 0.
    for fb, fb_q, c in zip(Fb, Fb_q, wgt):
        ans += c * tandem_objt(fa, fa_q, fb, fb_q, y, lam)
    return ans


def _arg_min_p(y, yt, yq, wgt, H, lam):
    # sum_j_in_H [pos/qtb]
    Fb = yt[H].tolist()
    Fb_q = yq[H].tolist()
    coef = [i for i, j in zip(wgt, H) if j]

    idx_i_not_H = np.where(np.logical_not(H))[0]
    if len(idx_i_not_H) == 0:
        return -1

    ans = [_tandem_obj_sum(
        yt[i], yq[i], Fb, Fb_q,
        coef, y, lam) for i in idx_i_not_H]
    idx_i = ans.index(np.min(ans))
    idx = idx_i_not_H[idx_i]

    return idx


def Centralised_EPAF_Pruning(y, yt, yq, wgt, nb_pru, lam):
    nb_cls = len(wgt)

    H = np.zeros(nb_cls, dtype='bool')
    p = np.random.randint(0, nb_cls)
    H[p] = True

    yt = np.array(yt)
    yq = np.array(yq)
    for _ in range(1, nb_pru):  # for i in
        idx = _arg_min_p(y, yt, yq, wgt, H, lam)

        if idx > -1:
            H[idx] = True
    # return H.tolist()
    return np.where(H)[0].tolist()


# -------------------------------------
# Algorithm 3.
#     Distributed Version (EPAF-D)
# -------------------------------------

# 1. Partition F randomly into n_m groups as equally as possible
# 2. for i=1 to n_m do
# 3.    H_i = EPAF-C(F_i, k)
# 4. end for
# 5. H' = EPAF-C(\bigcup_{1<=i<=n_m} H_i, k)
# 6. H = argmin_{\tao\in } hat_L(T, S)


def _randomly_partition(n, m):
    tmp = np.arange(n)
    np.random.shuffle(tmp)
    idx = np.zeros(n, dtype='int')

    if n % m == 0:
        gap = n // m
        for k in range(m):
            j = tmp[(k * gap): ((k + 1) * gap)]
            idx[j] = k
    else:

        floors = int(np.floor(n / m))
        ceilings = int(np.ceil(n / m))
        modulus = n - m * floors  # mod: n % m
        mumble = m * ceilings - n

        for k in range(modulus):
            j = tmp[(k * ceilings): ((k + 1) * ceilings)]
            idx[j] = k

        gap = ceilings * modulus
        for k in range(mumble):
            j = tmp[(k * floors + gap): ((k + 1) * floors + gap)]
            idx[j] = k + modulus

    return idx.tolist()


def _find_idx_in_sub(i, grp, yt, yq, wgt, y, k, lam):
    sub_idx_in_N = np.where(grp == i)[0]

    zt = yt[grp == i].tolist()
    zq = yq[grp == i].tolist()
    coef = [i for i, j in zip(wgt, grp == i) if j]
    sub_idx = Centralised_EPAF_Pruning(y, zt, zq, coef, k, lam)

    # NO NEED. # sub_idx = np.where(sub_idx)[0]
    ans = sub_idx_in_N[sub_idx]
    return ans


def _argmin_Tao(yt, yq, wgt, y, Hs, lam):
    # nb_cls = sum(Hs)
    # wgt = [1. / nb_cls for _ in range(nb_cls)]

    zt = yt[Hs].tolist()
    zq = yq[Hs].tolist()
    coef = np.array(wgt)[Hs].tolist()
    ans = cal_L_obj_v1(zt, zq, y, coef, lam)
    return ans


def Distributed_EPAF_Pruning(y, yt, yq, wgt, nb_pru, lam, n_m):
    nb_cls = len(yt)
    grp = _randomly_partition(n=nb_cls, m=n_m)
    grp = np.array(grp)  # partitioned groups
    H = np.zeros(nb_cls, dtype='int') - 1

    yt = np.array(yt)
    yq = np.array(yq)

    with pp.ProcessingPool(nodes=n_m) as pool:
        sub_idx = pool.map(_find_idx_in_sub,
                           list(range(n_m)),
                           [grp] * n_m,
                           [yt] * n_m,
                           [yq] * n_m,
                           [wgt] * n_m,
                           [y] * n_m,
                           [nb_pru] * n_m,
                           [lam] * n_m)
        sub_idx = list(sub_idx)
    for i in range(n_m):
        H[sub_idx[i]] = i
    del grp

    sub_all_in_N = np.where(H != -1)[0]
    zt = yt[H != -1].tolist()
    zq = yq[H != -1].tolist()
    coef = [i for i, j in zip(wgt, H != -1) if j]
    sub_all = Centralised_EPAF_Pruning(y, zt, zq, coef, nb_pru, lam)
    sub_all = sub_all_in_N[sub_all]

    del sub_all_in_N

    obj_tmpH = _argmin_Tao(yt, yq, wgt, y, sub_all, lam)
    obj_Hs = [_argmin_Tao(
        yt, yq, wgt, y, i, lam) for i in sub_idx]

    if np.sum(np.array(obj_Hs) < obj_tmpH) >= 1:
        tmp_argmin_l = obj_Hs.index(np.min(obj_Hs))
        sub_all = sub_idx[tmp_argmin_l]
        del tmp_argmin_l

    del obj_tmpH, obj_Hs, yt, yq, H
    gc.collect()
    return sub_all.tolist()


# =====================================
# Section 3.3 Ensemble Pruning (alternative)
# =====================================


# -------------------------------------
# Algorithm 1. alternative
# -------------------------------------


# Sub-algo 1. preliminary

_POAF_weakly_ds = _weakly_dominate
_POAF_dominate = _dominate
_POAF_calc_eval = _bi_objectives


def _POAF_check_randpick(s, nb_pru=None):
    # aka. def _POAF_random_pick()
    if sum(s) >= 1:
        return s
    nb_cls = len(s)
    sp = [0 for _ in s]  # sp = [0] * nb_cls
    sp[np.random.choice(range(nb_cls))] = 1
    if nb_pru is None:
        return sp
    i = np.random.randint(nb_cls, size=nb_pru - 1)
    for j in i:
        sp[j] = 1
    return sp


def _POAF_check_flipping(s, nb_pru=None):
    sp = _PEP_flipping_uniformly(s)
    if sum(sp) > 0:
        return sp
    nb_cls = len(s)
    sp[np.random.choice(range(nb_cls))] = 1
    if nb_pru is None:
        return sp
    i = np.random.randint(nb_cls, size=nb_pru - 1)
    for j in i:
        sp[j] = 1
    return sp


def _POAF_bi_objects(y, zt, zq, wgt, s):
    # aka. def _bi_goals_whole()
    # but not quite the same exactly
    bf_h = np.array(s, dtype='bool')  # DTY_BOL)

    yt = zt[bf_h].tolist()
    yq = zq[bf_h].tolist()
    coef = np.array(wgt)[bf_h].tolist()

    fens_t = weighted_voting(yt, coef)  # y,
    fens_q = weighted_voting(yq, coef)  # y,
    sub_no1 = hat_L_loss(fens_t, y)
    sub_no2 = hat_L_fair(fens_t, fens_q)
    return (sub_no1, sub_no2)


def _POAF_obj_eval(y, zt, zq, wgt, s, lam):
    # aka. def _bi_goals_split()
    # but not quite the same exactly
    bf_h = np.array(s, dtype='bool')  # DTY_BOL)
    if np.sum(bf_h) < 1:
        return 1, (1, 1)
    # namely, def _POAF_eval()

    yt = zt[bf_h].tolist()
    yq = zq[bf_h].tolist()
    coef = np.array(wgt)[bf_h].tolist()

    sub_no1 = E_rho_L_loss_f(yt, y, coef)
    sub_no2 = Erho_sup_L_fair(yt, yq, coef)
    G_mv = (sub_no1, sub_no2)

    ans = _POAF_calc_eval(G_mv, lam)
    return ans, G_mv


# Sub-algo 2. (VDS Subroutine)
#
# Given a pseudo-Boolean function f and a solution \mathbf{s}, it
# contains:
#
#  1.   Q= \emptyset, L= \emptyset
#  2.   Let N(.) denote the set of neighbor solutions of a binary
#       vector with Hamming distance 1.
#  3.   While V_s={ y\in N(s)| (y_i\neq s_i ==> i\nin L)} \neq \emptyset
#  4.       Choose y\in V_s with the minimal f value
#  5.       Q= Q\bigcup {y}
#  6.       L= L\bigcup {i| y_i\neq s_i}
#  7.       s = y
#  8.   Output Q.
#


def POAF_VDS(y, zt, zq, wgt, s, lam):
    nb_cls = len(wgt)
    QL = np.zeros(nb_cls, dtype='bool')
    sp = deepcopy(s)
    Q, L = [], []
    while np.sum(QL) < nb_cls:
        Ns = [deepcopy(sp) for i in range(nb_cls)]
        for i in range(nb_cls):
            Ns[i][i] = 1 - Ns[i][i]
        Ns = np.array(Ns)

        idx_Vs = np.where(np.logical_not(QL))[0]
        Vs = Ns[idx_Vs].tolist()

        obj_f = [_POAF_obj_eval(y, zt, zq, wgt, i, lam)[0] for i in Vs]
        idx_f = obj_f.index(np.min(obj_f))
        yp = Vs[idx_f]

        Q.append(deepcopy(yp))
        L.append(int(idx_Vs[idx_f]))
        QL[idx_Vs[idx_f]] = True

        sp = deepcopy(yp)
        del Ns, idx_Vs, Vs, obj_f, idx_f, yp
    del QL
    return deepcopy(Q), deepcopy(L)  # np.ndarray


# Sub-algo 1. (PEP)
#
# Given a set of trained classifiers H={h_i}_{i=1}^n, an objective f:
# 2^H ->\mathbb{R}, and an evaluation criterion eval, it contains:
#
#  1.   Let $\mathbf{g}(s)=(f(H_s),|s|) be the bi-objective.
#  2.   Let s=randomly selected from {0,1}^n and P={s}.
#  3.   Repeat
#  4.       Select s\in P uniformly at random.
#  5.       Generate s' by flipping each bit of s with prob. 1/n.
#  6.       if \nexists z\in P such that z\succ_g (s')
#  7.           P=(P- {z\in P| s'\succeq_g z}) \bigcup {s'}
#  8.           Q= VDS(f,s')
#  9.           for q\in Q
# 10.               if \nexists z\in P such that z\succ_g q
# 11.                   P=(P- {z\in P| q\succeq_g z}) \bigcup {q}
# 12.   Output argmin_{s\in P} eval(s)
#


def POAF_if_nexists_succ(y, zt, zq, wgt, P, sp):
    # aka. sp/q
    g_sp = _POAF_bi_objects(y, zt, zq, wgt, sp)
    for z in P:
        g_z = _POAF_bi_objects(y, zt, zq, wgt, z)
        if _POAF_dominate(g_z, g_sp):
            return True, deepcopy(z)
    return False, []


def POAF_refresh_succeq(y, zt, zq, wgt, P, sp):
    # aka. sp/q
    g_sp = _POAF_bi_objects(y, zt, zq, wgt, sp)
    all_z_in_P = [_POAF_bi_objects(y, zt, zq, wgt, i) for i in P]
    all_z_in_P = [_POAF_weakly_ds(g_sp, i) for i in all_z_in_P]
    idx = np.where(np.logical_not(all_z_in_P))[0]
    P = [P[i] for i in idx]
    P.append(deepcopy(sp))
    return P, idx


def POAF_check_item_empty(P):
    # aka. POAF_check_empty()
    # need_to_del = [np.sum(t) == 0 for t in P]
    need_to_del = [sum(t) == 0 for t in P]
    return [v for k, v in zip(need_to_del, P) if not k]


def POAF_PEP(y, yt, yq, wgt, lam, nb_pru):
    nb_cls = len(wgt)
    rho = float(nb_pru) / nb_cls
    zt, zq = np.array(yt), np.array(yq)

    s = np.random.randint(2, size=nb_cls).tolist()
    s = _POAF_check_randpick(s, nb_pru)  # _POAF_check_randpick(s)
    P = [deepcopy(s)]

    nb_cnt = int(np.ceil(rho * nb_cls))
    # nb_cnt = max([nb_cnt, nb_pru, nb_pru + 1])
    while nb_cnt > 0:
        idx = np.random.randint(len(P))
        s0 = P[idx]
        sp = _POAF_check_flipping(
            s0, nb_pru)  # _PEP_flipping_uniformly(s0)

        signal_1, _ = POAF_if_nexists_succ(y, zt, zq, wgt, P, sp)

        if not signal_1:
            P, _ = POAF_refresh_succeq(y, zt, zq, wgt, P, sp)
            P = POAF_check_item_empty(P)

            Q, _ = POAF_VDS(y, zt, zq, wgt, sp, lam)
            # IF sum(sp)==1, then it will get bug
            Q = POAF_check_item_empty(Q)
            for q in Q:
                signal_3, _ = POAF_if_nexists_succ(
                    y, zt, zq, wgt, P, q)

                if not signal_3:
                    P, _ = POAF_refresh_succeq(y, zt, zq, wgt, P, q)
                    P = POAF_check_item_empty(P)

        #     del signal_3, g_q
        #   del q, Q
        # del signal_1, g_sp, sp, s0, idx
        nb_cnt -= 1
    del nb_cnt, s

    P = POAF_check_item_empty(P)
    obj_eval = [_POAF_obj_eval(y, zt, zq, wgt, t, lam)[0] for t in P]
    idx_eval = obj_eval.index(min(obj_eval))
    s = P[idx_eval]

    del P, obj_eval, idx_eval
    seq = np.where(np.array(s) == 1)[0]
    return seq.tolist()


# -------------------------------------
# -------------------------------------


# =====================================
# Algorithm 4.
# Ranking based (different fairness measures)
# =====================================


def Ranking_based_criterion(y, hx, f_qtb, lam, criteria="DR",
                            # pos=1, idx_priv=list()):
                            pos=1, idx_priv=tuple()):
    if criteria == "DR":
        l_fair = hat_L_fair(hx, f_qtb)
        l_acc_ = hat_L_loss(hx, y)
        return lam * l_acc_ + (1. - lam) * l_fair

    _, _, gones_Cm, gzero_Cm = marginalised_pd_mat(y, hx, pos,
                                                   idx_priv)
    if criteria == "unaware":
        g1, g0 = unpriv_unaware(gones_Cm, gzero_Cm)
    elif criteria == "DP":
        g1, g0 = unpriv_group_one(gones_Cm, gzero_Cm)
    elif criteria == "EO":
        g1, g0 = unpriv_group_two(gones_Cm, gzero_Cm)
    elif criteria == "PQP":
        g1, g0 = unpriv_group_thr(gones_Cm, gzero_Cm)
    elif criteria == "manual":
        g1, g0 = unpriv_manual(gones_Cm, gzero_Cm)
    else:
        raise ValueError("Wrong criteria `{}`".format(criteria))
    return abs(g1 - g0)


def Ranking_based_fairness_Pruning(y, yt, yq, nb_pru, lam,
                                   criteria="DR", pos=1,
                                   # idx_priv=list()):
                                   idx_priv=tuple()):
    nb_cls = len(yt)  # len(wgt)
    rank_val = list(map(Ranking_based_criterion,
                        [y] * nb_cls, yt, yq,
                        # yt.tolist(), yq.tolist(),
                        [lam] * nb_cls, [criteria] * nb_cls,
                        [pos] * nb_cls, [idx_priv] * nb_cls))

    # U = np.array([rank_val])
    # rank_idx = _Friedman_sequential(U, mode='ascend', dim=2)
    # idx_bar = _Friedman_successive(U, rank_idx)

    rank_idx = _Friedman_sequential(
        [rank_val], mode='ascend', dim=2)[0]
    H = rank_idx <= nb_pru
    return H.tolist(), rank_idx.tolist()


# -------------------------------------
# Algorithm 4.
# -------------------------------------
