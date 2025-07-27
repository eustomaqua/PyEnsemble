# coding: utf-8
# hfm.experiment/utils/fair_rev_indiv.py
#
# Aim to provide:
#   Discrimination/ fairness metrics
#


import numpy as np
from pyfair.facil.utils_timer import fantasy_timer
from pyfair.facil.utils_const import check_zero
from pyfair.marble.metric_fair import _elem

from pyfair.dr_hfm.dist_drt import (
    DirectDist_bin, DirectDist_nonbin, DirectDist_multiver)
from pyfair.dr_hfm.dist_est_nonbin import (
    ApproxDist_nonbin_mpver, ExtendDist_multiver_mp)
from pyfair.dr_hfm.dist_est_bin import ApproxDist_bin

# from hfm.dist_drt import DistDirect_mediator
# from hfm.hfm_df import bias_degree_bin as fair_degree_v3
# from hfm.hfm_df import bias_degree_nonbin as fair_degree_v4
# from hfm.hfm_df import bias_degree as fair_degree
# from hfm.dist_est_nonbin import AcceleDist_nonbin as DistAccele

DistApprox = ApproxDist_nonbin_mpver
del ApproxDist_nonbin_mpver


# =====================================
# Individual fairness


# -------------------------------------
# Definition 2.10 (General entropy indices)


class GEI_Theil(_elem):
    @staticmethod
    def _benefits(y, y_hat):
        # bi = y_hat - y + 1
        bi = np.subtract(y_hat, y) + 1
        n = len(y)
        mu = np.sum(bi) / float(n)
        return bi, mu, n

    @classmethod
    @fantasy_timer
    def get_GEI(cls, y, y_hat, alpha=.5):
        bi, mu, _ = cls._benefits(y, y_hat)
        numerator = (bi / mu) ** alpha - 1
        denominator = len(y) * alpha * (alpha - 1)
        # return np.sum(numerator) / denominator
        ans = np.sum(numerator) / denominator
        return float(ans)

    @classmethod
    @fantasy_timer
    def get_Theil(cls, y, y_hat):
        # a special case for alpha=1
        bi, mu, n = cls._benefits(y, y_hat)
        tmp_1 = bi / mu
        tmp_2 = [check_zero(i) for i in tmp_1]
        tmp_2 = np.log(tmp_2)
        tmp = tmp_1 * tmp_2
        del tmp_1, tmp_2
        # return np.sum(tmp) / float(n)
        ans = np.sum(tmp) / float(n)
        return float(ans)


# -------------------------------------
# Definition 2.14 (γ-subgroup fairness)
#   that is, False positive (FP) subgroup fairness


# -------------------------------------
# Definition 2.16 (Bounded group loss)


# -------------------------------------
#


# -------------------------------------
#


# =====================================
# Individual fairness (cont.)


# -------------------------------------
# Discriminative risk (DR)


class prop_L_fair(_elem):
    @staticmethod
    @fantasy_timer
    def hat_L(y_hat, y_qtb):
        tmp = np.not_equal(y_hat, y_qtb)
        return float(tmp.mean())

    @staticmethod
    @fantasy_timer
    def tandem(fi, fi_q, fk, fk_q):
        hi = np.not_equal(fi, fi_q)
        hk = np.not_equal(fk, fk_q)
        tmp = np.logical_and(hi, hk)
        return float(tmp.mean())

    # Theorem 3.1: First-order oracle bound
    @classmethod
    @fantasy_timer
    def E_rho_L(cls, y_hat, y_qtb, wgt):
        E_rho = [cls.hat_L(y, yq)[0] for y, yq in zip(y_hat, y_qtb)]
        tmp = np.sum(np.multiply(wgt, E_rho))
        return float(tmp)

    # Theorem 3.3: Second-order oracle bound
    @classmethod
    def Erho_sup_L(cls, y_hat, y_qtb, wgt, nb_cls=None):
        if not nb_cls:
            nb_cls = len(wgt)  # number of weights

        L_f_fp = []
        for p in range(nb_cls):
            tmp = [cls.tandem(
                y_hat[p], y_qtb[p],
                y_hat[i], y_qtb[i])[0] for i in range(nb_cls)]
            L_f_fp.append(tmp)
        # L_f_fp: list, shape= (nb_cls, nb_cls)

        E_rho2 = np.sum(np.multiply(L_f_fp, wgt), axis=1)
        E_rho2 = np.sum(np.multiply(E_rho2, wgt), axis=0)
        # E_rho: list, shape= (nb_cls,)
        return float(E_rho2)

    # Lemma 3.2
    @classmethod
    def ED_Erho_I(cls, y_hat, y_qtb, wgt):
        wt = np.array([wgt]).T
        I_f = np.not_equal(y_hat, y_qtb)  # sz= (nb_cls, n)
        Erho = np.sum(wt * I_f, axis=0)   # size= (n,)
        ED = np.mean(Erho * Erho)
        return float(ED)

    # Theorem 3.4: C-tandem oracle bound


class prop_L_loss(_elem):
    @staticmethod
    @fantasy_timer
    def hat_L(y_hat, y):
        tmp = np.not_equal(y_hat, y)
        return float(tmp.mean())

    @staticmethod
    @fantasy_timer
    def tandem(fi, fk, y):
        hi = np.not_equal(fi, y)
        hk = np.not_equal(fk, y)
        tmp = np.logical_and(hi, hk)
        return float(tmp.mean())

    # Theorem 3.1: First-order oracle bound
    @classmethod
    @fantasy_timer
    def E_rho_L(cls, y_hat, y, wgt):
        E_rho = [cls.hat_L(yp, y)[0] for yp in y_hat]
        tmp = np.sum(np.multiply(wgt, E_rho))
        return float(tmp)

    # Theorem 3.3: Second-order oracle bound
    @classmethod
    def Erho_sup_L(cls, y_hat, y, wgt, nb_cls=None):
        if not nb_cls:
            nb_cls = len(wgt)  # length of weights

        L_f_fp = []
        for p in range(nb_cls):
            tmp = [cls.tandem(
                y_hat[p], y_hat[i], y)[0] for i in range(nb_cls)]
            L_f_fp.append(tmp)
        # L_f_fp: list, shape= (nb_cls, nb_cls)

        E_rho2 = np.sum(np.multiply(L_f_fp, wgt), axis=1)
        E_rho2 = np.sum(np.multiply(E_rho2, wgt), axis=0)
        return float(E_rho2)

    # Lemma 3.2
    @classmethod
    def ED_Erho_I(cls, y_hat, y, wgt):
        wt = np.array([wgt]).T
        I_f = np.not_equal(y_hat, y)  # sz= (nb_cls, n)
        Erho = np.sum(wt * I_f, axis=0)   # size= (n,)
        ED = np.mean(Erho * Erho)
        return float(ED)

    # Theorem 3.4: C-tandem oracle bound


# -------------------------------------
# Harmonious fairness via manifolds


class DistDirect(_elem):
    @staticmethod
    def bin(X_nA_y, idx_Si):
        # return DistDirect_bin(X_nA_y, idx_Si)
        return DirectDist_bin(X_nA_y, idx_Si)

    @staticmethod
    def nonbin(X_nA_y, idx_Sjs):
        # return DistDirect_nonbin(X_nA_y, idx_Sjs)
        return DirectDist_nonbin(X_nA_y, idx_Sjs)

    @staticmethod
    def multivar(X_nA_y, idx_As_Sj):
        # return DistDirect_multivar(X_nA_y, idx_As_Sj)
        return DirectDist_multiver(X_nA_y, idx_As_Sj)


# class HFM_Approx_bin(_elem):
#     @staticmethod
#     def bin(X_nA_y, A_j, idx_Sj, m1, m2):
#         # '''
#         # Si_c = ~idx_Sj  # idx_Sj: i.e. indices of non_sa
#         # return ApproxDist_bin(X_nA_y, A_j, Si_c, idx_Sj, m1, m2)
#         # '''
#         return ApproxDist_bin(X_nA_y, A_j, idx_Sj, m1, m2)
#
#
# class HFM_DistApprox(_elem):
#     @staticmethod
#     def bin(X_nA_y, A_j, m1, m2, n_e=3, pool=None):
#         # B_j = A_j.copy()
#         # B_j[B_j != 1] = 0  # privileged_val = 1
#         # return DistApprox(X_nA_y, B_j, m1, m2, n_e, pool)
#         return DistApprox(X_nA_y, A_j, m1, m2, n_e, pool)
#
#     @staticmethod
#     def nonbin(X_nA_y, A_j, m1, m2, n_e=3, pool=None):
#         return DistApprox(X_nA_y, A_j, m1, m2, n_e, pool)
#
#     @staticmethod
#     def multivar(X_nA_y, A, m1, m2, n_e=3, pool=None):
#         # return DistExtend(X_nA_y, A, m1, m2, n_e, pool)
#         return ExtendDist_multiver_mp(
#             X_nA_y, A, m1, m2, n_e, pool)


# -------------------------------------
#

# -------------------------------------
#
