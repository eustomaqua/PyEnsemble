# coding: utf-8
# hfm.experiment/utils/fair_rev_group.py
#
# Aim to provide:
#   Discrimination/ fairness metrics
#


import numpy as np
from pyfair.facil.utils_timer import fantasy_timer
from pyfair.facil.metric_cont import contingency_tab_bi
from pyfair.marble.metric_fair import (  # hfm.metrics.fair_grp_ext
    marginalised_np_mat, marginalised_np_gen, _elem, zero_division)


# =====================================
# Fairness research
# =====================================


# =====================================
# Group fairness


# -------------------------------------
# Independence
#
# Definition 1.
#   Random variables (A,R) satisfy independence if A⊥R.
# e.g.
#   Pr{\hat{Y}=1 | A=a} = Pr{\hat{Y}=1 | A=b}
#


# Definition 2.1 (Demographic parity /statistic parity)

class UD_grp1_DP(_elem):
    @staticmethod
    def _core_alt(g_Cm):
        # (tp+fp)/n where n=tp+fp+fn+tn
        # return tmp / sum(g_Cm)
        tmp = g_Cm[0] + g_Cm[1]
        return zero_division(tmp, sum(g_Cm))

    @classmethod
    def _core(cls, g1_Cm, g0_Cm):
        g1 = cls._core_alt(g1_Cm)
        g0 = cls._core_alt(g0_Cm)
        return abs(g0 - g1), float(g1), float(g0)

    @classmethod
    @fantasy_timer
    def bival(cls, y, y_hat, priv_idx, pos_label=1):
        g1_Cm, g0_Cm = marginalised_np_mat(
            y, y_hat, pos_label, priv_idx)
        return cls._core(g1_Cm, g0_Cm)

    @classmethod
    @fantasy_timer
    def mu_sp(cls, y, y_hat, A, priv_val=1, pos_label=1):
        g1_Cm, g0_Cm = marginalised_np_mat(
            y, y_hat, pos_label, A == priv_val)
        return cls._core(g1_Cm, g0_Cm)[0]

    @classmethod
    @fantasy_timer
    def mu_cx(cls, y, y_hat, A, priv_val=1, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, priv_val, pos_label)
        g = [cls._core_alt(k) for k in g_Cm]
        ans = alt = 0.
        tmp, nt, n = cls._indices(vA, idx, ex)
        for i in tmp:
            ans += abs(g[i] - g[idx]) * ex[i] / n
            alt += abs(g[i] - g[idx]) * ex[i] / nt
        return ans, alt

    @classmethod
    @fantasy_timer
    def yev_sp(cls, y, y_hat, A, vals_in_A, pos_label=1):
        ans_middling, ans, alt = [], 0., 0.
        for v in vals_in_A:
            res = cls.mu_sp(y, y_hat, A, v, pos_label)[0]
            ans_middling.append(res)
            alt += res * np.mean(A == v)
        ans = float(np.mean(ans_middling))
        return ans, max(ans_middling), float(alt), ans_middling

    @classmethod
    @fantasy_timer
    def yev_cx(cls, y, y_hat, A, vals_in_A, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, vals_in_A[0], pos_label)  # priv_val
        g = [cls._core_alt(k) for k in g_Cm]
        ans = 0.  # ans = alt = 0.
        ans_mediator, n_a = [], len(vA)
        for i in range(n_a - 1):
            for j in range(i + 1, n_a):
                ans_mediator.append(abs(g[j] - g[i]))
        ans = float(np.mean(ans_mediator))
        return ans, max(ans_mediator), ans_mediator


# Definition 2.2

class UD_grp1_DisI(UD_grp1_DP):
    @classmethod
    def _core(cls, g1_Cm, g0_Cm):
        g1 = cls._core_alt(g1_Cm)
        g0 = cls._core_alt(g0_Cm)
        return zero_division(g0, g1), float(g1), float(g0)

    @classmethod
    @fantasy_timer
    def mu_cx(cls, y, y_hat, A, priv_val=1, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, priv_val, pos_label)
        g = [cls._core_alt(k) for k in g_Cm]
        ans = alt = 0.
        tmp, nt, n = cls._indices(vA, idx, ex)
        for i in tmp:
            tk = zero_division(g[i], g[idx])
            ans += tk * ex[i] / n
            alt += tk * ex[i] / nt
        return ans, alt

    @classmethod
    @fantasy_timer
    def yev_cx(cls, y, y_hat, A, vals_in_A, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, vals_in_A[0], pos_label)
        g = [cls._core_alt(k) for k in g_Cm]
        t1 = cls.yev_cx_v1(cls, g, vA)[: -1]
        t2 = cls.yev_cx_v2(cls, g, vA)[: -1]
        return t2[0], t1[0], t2[1], t1[1]

    def yev_cx_v1(self, g, vA):
        ans_mediator, n_a = [], len(vA)
        for i in range(n_a - 1):
            for j in range(i + 1, n_a):
                ans_mediator.append(zero_division(g[j], g[i]))
        ans = float(np.mean(ans_mediator))
        return ans, max(ans_mediator), ans_mediator

    def yev_cx_v2(self, g, vA):
        ans_mediator, n_a = [], len(vA)
        for i in range(n_a):
            for j in range(n_a):
                if j == i:
                    continue
                ans_mediator.append(zero_division(g[j], g[i]))
        ans = float(np.mean(ans_mediator))
        return ans, max(ans_mediator), ans_mediator


# Definition 2.3 (Disparate treatment)

class UD_grp1_DisT(UD_grp1_DP):
    @classmethod
    def _core(cls, g1_Cm, g0_Cm):
        g1 = cls._core_alt(g1_Cm)
        g0 = cls._core_alt(g0_Cm)
        z_Cm = [i + j for i, j in zip(g1_Cm, g0_Cm)]
        z = cls._core_alt(z_Cm)
        n1, n0, n = sum(g1_Cm), sum(g0_Cm), sum(z_Cm)
        ans = 0.
        ans += abs(g0 - z) * n0 / n
        ans += abs(g1 - z) * n1 / n
        return ans, z, g1, g0

    @classmethod
    @fantasy_timer
    def mu_cx(cls, y, y_hat, A, priv_val=1, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, priv_val, pos_label)
        g = [cls._core_alt(k) for k in g_Cm]
        z_Cm = contingency_tab_bi(y, y_hat, pos_label)
        z = cls._core_alt(z_Cm)
        ans, n = 0., sum(ex)
        for i in range(len(vA)):
            ans += abs(g[i] - z) * ex[i] / n
        return ans

    @classmethod
    @fantasy_timer
    def yev_sp(cls, y, y_hat, A, vals_in_A, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, vals_in_A[0], pos_label)
        g = [cls._core_alt(k) for k in g_Cm]
        z_Cm = contingency_tab_bi(y, y_hat, pos_label)
        z = cls._core_alt(z_Cm)
        ans_middling = []
        for i in range(len(vA)):
            ans_middling.append(abs(g[i] - z))
        return float(np.mean(ans_middling))

    @classmethod
    @fantasy_timer
    def yev_cx(cls, y, y_hat, A, vals_in_Ai, pos_label=1):
        return NotImplementedError


# Definition 2.7 (Conditional statistical parity)

# class group1_CSP(group1_DP):
#     @classmethod
#     def _core(cls, g1_Cm, g0_Cm):
#         pass


# -------------------------------------
# Separation
#
# Definition 2.
#   Random variables (R,A,Y) satisfy separation if R⊥A|Y.
# e.g.
#   Pr{\hat{Y}=1 | Y=1,A=a} = Pr{\hat{Y}=1 | Y=1,A=b}
#   Pr{\hat{Y}=1 | Y=0,A=a} = Pr{\hat{Y}=1 | Y=0,A=b}
#


# Definition 2.5 (Equality of opportunity)

class UD_grp2_EO(_elem):
    @staticmethod
    def _core_alt(g_Cm):
        # tp/(tp+fn)
        # return g_Cm[0] / check_zero(tmp)
        tmp = g_Cm[0] + g_Cm[2]
        return zero_division(g_Cm[0], tmp)

    @classmethod
    def _core(cls, g1_Cm, g0_Cm):
        g1 = cls._core_alt(g1_Cm)
        g0 = cls._core_alt(g0_Cm)
        return abs(g0 - g1), float(g1), float(g0)

    @classmethod
    @fantasy_timer
    def bival(cls, y, y_hat, priv_idx, pos_label=1):
        g1_Cm, g0_Cm = marginalised_np_mat(
            y, y_hat, pos_label, priv_idx)
        return cls._core(g1_Cm, g0_Cm)

    @classmethod
    @fantasy_timer
    def mu_sp(cls, y, y_hat, A, priv_val=1, pos_label=1):
        g1_Cm, g0_Cm = marginalised_np_mat(
            y, y_hat, pos_label, A == priv_val)
        return cls._core(g1_Cm, g0_Cm)[0]

    @classmethod
    @fantasy_timer
    def mu_cx(cls, y, y_hat, A, priv_val=1, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, priv_val, pos_label)
        g = [cls._core_alt(k) for k in g_Cm]
        ans = alt = 0.
        tmp, nt, n = cls._indices(vA, idx, ex)
        for i in tmp:
            ans += abs(g[i] - g[idx]) * ex[i] / n
            alt += abs(g[i] - g[idx]) * ex[i] / nt
        return ans, alt

    @classmethod
    @fantasy_timer
    def yev_sp(cls, y, y_hat, A, vals_in_A, pos_label=1):
        ans_middling, ans, alt = [], 0., 0.
        for v in vals_in_A:
            res = cls.mu_sp(y, y_hat, A, v, pos_label)[0]
            ans_middling.append(res)
            alt += res * np.mean(A == v)
        ans = float(np.mean(ans_middling))
        return ans, max(ans_middling), float(alt), ans_middling

    @classmethod
    @fantasy_timer
    def yev_cx(cls, y, y_hat, A, vals_in_Ai, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, vals_in_Ai[0], pos_label)
        g = [cls._core_alt(k) for k in g_Cm]
        ans = 0.  # ans = alt = 0.
        ans_mediator, n_a = [], len(vA)
        for i in range(n_a - 1):
            for j in range(i + 1, n_a):
                ans_mediator.append(abs(g[j] - g[i]))
        ans = float(np.mean(ans_mediator))
        return ans, max(ans_mediator), ans_mediator


# Definition 2.4 (Equalised odds)

class UD_grp2_EOdd(UD_grp2_EO):
    @staticmethod
    def _corrected_neg(g_Cm):
        # fp/(fp+tn)
        tmp = g_Cm[1] + g_Cm[3]
        return zero_division(g_Cm[1], tmp)

    @staticmethod
    def _neg_label(y, y_hat, pos_label):
        # only for binary classification
        neg_label = set(y) | set(y_hat)
        if pos_label in neg_label:
            neg_label.remove(pos_label)
        return list(neg_label)[0]

    @classmethod
    @fantasy_timer
    def bival(cls, y, y_hat, priv_idx, pos_label=1):
        g1_Cm, g0_Cm = marginalised_np_mat(
            y, y_hat, pos_label, priv_idx)
        g_delta, g1, g0 = cls._core(g1_Cm, g0_Cm)

        z1 = cls._corrected_neg(g1_Cm)
        z0 = cls._corrected_neg(g0_Cm)
        z_delta = abs(z0 - z1)
        z1, z0 = float(z1), float(z0)

        n_pos, n = sum(y == pos_label), len(y)
        n_neg = n - n_pos  # sum(y != pos_label)
        ans = g_delta * n_pos / n + z_delta * n_neg / n
        return ans, (g_delta, z_delta), (g1, g0, z1, z0)

    @classmethod
    @fantasy_timer
    def mu_sp(cls, y, y_hat, A, priv_val=1, pos_label=1):
        priv_idx = A == priv_val
        tmp, _ = cls.bival(y, y_hat, priv_idx, pos_label)
        return tmp[0]

    @classmethod
    @fantasy_timer
    def mu_cx(cls, y, y_hat, A, priv_val=1, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, priv_val, pos_label)
        g = [cls._core_alt(k) for k in g_Cm]
        g_ans = g_alt = 0.
        tmp, nt, n = cls._indices(vA, idx, ex)
        for i in tmp:
            g_ans += abs(g[i] - g[idx]) * ex[i] / n
            g_alt += abs(g[i] - g[idx]) * ex[i] / nt

        z = [cls._corrected_neg(k) for k in g_Cm]
        z_ans = z_alt = 0.
        for i in tmp:
            z_ans += abs(z[i] - z[idx]) * ex[i] / n
            z_alt += abs(z[i] - z[idx]) * ex[i] / nt

        n_pos, n = sum(y == pos_label), len(y)
        n_neg = sum(y != pos_label)  # n - n_pos
        ans = g_ans * n_pos / n + z_ans * n_neg / n
        alt = g_alt * n_pos / n + z_alt * n_neg / n
        return ans, alt

    @classmethod
    @fantasy_timer
    def yev_sp(cls, y, y_hat, A, vals_in_A, pos_label=1):
        ans_middling, ans, alt = [], 0., 0.
        for v in vals_in_A:
            res = cls.mu_sp(y, y_hat, A, v, pos_label)[0]
            ans_middling.append(res)
            alt += res * np.mean(A == v)
        ans = float(np.mean(ans_middling))
        return ans, max(ans_middling), float(alt), ans_middling

    @classmethod
    @fantasy_timer
    def yev_cx(cls, y, y_hat, A, vals_in_Ai, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, vals_in_Ai[0], pos_label)
        g = [cls._core_alt(k) for k in g_Cm]
        n_ai = len(vA)  # not n_a
        g_ans = g_alt = g_alt_alt = 0.
        _, nt, n = cls._indices(vA, idx, ex)
        for i in range(n_ai):
            for j in range(n_ai):
                if j == i:
                    continue
                g_ans += abs(g[j] - g[i])
                g_alt += abs(g[j] - g[i]) * ex[j] / n
                g_alt_alt += abs(g[j] - g[i]) * ex[j] / nt
        g_ans /= n_ai * (n_ai - 1.)

        z = [cls._corrected_neg(k) for k in g_Cm]
        z_ans = z_alt = z_alt_alt = 0.
        for i in range(n_ai):
            for j in range(n_ai):
                if j == i:
                    continue
                z_ans += abs(z[j] - z[i])
                z_alt += abs(z[j] - z[i]) * ex[j] / n
                z_alt_alt += abs(z[j] - z[i]) * ex[j] / nt
        z_ans /= n_ai * (n_ai - 1.)

        n_pos, n = sum(y == pos_label), len(y)
        n_neg = n - n_pos
        ans = g_ans * n_pos / n + z_ans * n_neg / n
        alt = g_alt * n_pos / n + z_alt * n_neg / n
        alt_alt = g_alt_alt * n_pos / n + z_alt_alt * n_neg / n
        del g_alt, g_alt_alt, z_alt, z_alt_alt
        del n, n_ai, n_pos, n_neg
        return ans, alt, alt_alt


# Definition 2.8 (Predictive equality)

class UD_grp2_PEq(UD_grp2_EO):
    @classmethod
    def _core(cls, g1_Cm, g0_Cm):
        g1 = cls._corrected_neg(g1_Cm)
        g0 = cls._corrected_neg(g0_Cm)
        z_Cm = [i + j for i, j in zip(g1_Cm, g0_Cm)]
        z = cls._corrected_neg(z_Cm)

        n1, n0, n = sum(g1_Cm), sum(g0_Cm), sum(z_Cm)
        ans = 0.
        ans += abs(g0 - z) * n0 / n
        ans += abs(g1 - z) * n1 / n
        return ans, z, g1, g0

    @classmethod
    @fantasy_timer
    def mu_cx(cls, y, y_hat, A, priv_val=1, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, priv_val, pos_label)
        g = [cls._corrected_neg(k) for k in g_Cm]
        z_Cm = contingency_tab_bi(y, y_hat, pos_label)
        z = cls._corrected_neg(z_Cm)

        ans = 0.     # ans = alt = 0.
        n = sum(ex)  # n, nt = sum(ex), sum(z_Cm)
        for i in range(len(vA)):
            ans += abs(g[i] - z) * ex[i] / n
        return ans

    @staticmethod
    def _corrected_neg(g_Cm):
        tmp = g_Cm[1] + g_Cm[3]
        return zero_division(g_Cm[1], tmp)

    @classmethod
    @fantasy_timer
    def yev_sp(cls, y, y_hat, A, vals_in_A, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, vals_in_A[0], pos_label)
        g = [cls._corrected_neg(k) for k in g_Cm]
        z_Cm = contingency_tab_bi(y, y_hat, pos_label)
        z = cls._corrected_neg(z_Cm)
        ans_middling = []  # ans, n = 0., sum(ex)
        for i in range(len(vA)):
            ans_middling.append(abs(g[i] - z))
        return float(np.mean(ans_middling))

    @classmethod
    @fantasy_timer
    def yev_cx(cls, y, y_hat, A, vals_in_Ai, pos_label=1):
        return NotImplementedError


# -------------------------------------
# Sufficiency
#
# Definition 3.
#   We say the random variables (R,A,Y) satisfy sufficiency
#   if Y⊥A|R.
# e.g.
#   Pr{Y=1 | R=r,A=a} = Pr{Y=1 | R=r,A=b}
#


# Definition 2.6 (Predictive parity)

class UD_grp3_PQP(_elem):
    @staticmethod
    def _core_alt(g_Cm):
        # tp/(tp+fp)
        # return g_Cm[0] / check_zero(tmp)
        tmp = g_Cm[0] + g_Cm[1]
        return zero_division(g_Cm[0], tmp)

    @classmethod
    def _core(cls, g1_Cm, g0_Cm):
        g1 = cls._core_alt(g1_Cm)
        g0 = cls._core_alt(g0_Cm)
        return abs(g0 - g1), float(g1), float(g0)

    @classmethod
    @fantasy_timer
    def bival(cls, y, y_hat, priv_idx, pos_label=1):
        g1_Cm, g0_Cm = marginalised_np_mat(
            y, y_hat, pos_label, priv_idx)
        return cls._core(g1_Cm, g0_Cm)

    @classmethod
    @fantasy_timer
    def mu_sp(cls, y, y_hat, A, priv_val=1, pos_label=1):
        g1_Cm, g0_Cm = marginalised_np_mat(
            y, y_hat, pos_label, A == priv_val)
        return cls._core(g1_Cm, g0_Cm)[0]

    @classmethod
    @fantasy_timer
    def mu_cx(cls, y, y_hat, A, priv_val=1, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, priv_val, pos_label)
        g = [cls._core_alt(k) for k in g_Cm]
        ans = alt = 0.
        tmp, nt, n = cls._indices(vA, idx, ex)
        for i in tmp:
            ans += abs(g[i] - g[idx]) * ex[i] / n
            alt += abs(g[i] - g[idx]) * ex[i] / nt
        return ans, alt

    @classmethod
    @fantasy_timer
    def yev_sp(cls, y, y_hat, A, vals_in_A, pos_label=1):
        ans_middling, ans, alt = [], 0., 0.
        for v in vals_in_A:
            res = cls.mu_sp(y, y_hat, A, v, pos_label)[0]
            ans_middling.append(res)
            alt += res * np.mean(A == v)
        ans = float(np.mean(ans_middling))
        return ans, max(ans_middling), float(alt), ans_middling

    @classmethod
    @fantasy_timer
    def yev_cx(cls, y, y_hat, A, vals_in_Ai, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, vals_in_Ai[0], pos_label)
        g = [cls._core_alt(k) for k in g_Cm]
        ans = 0.  # ans = alt = 0.
        ans_mediator, n_a = [], len(vA)
        for i in range(n_a):
            for j in range(i + 1, n_a):
                ans_mediator.append(abs(g[j] - g[i]))
        ans = float(np.mean(ans_mediator))
        return ans, max(ans_mediator), ans_mediator


# -------------------------------------
#


# =====================================
# Individual fairness
# -------------------------------------
# Definition 2.10 (General entropy indices)


# Definition 2.14 (γ-subgroup fairness)
#   that is, False positive (FP) subgroup fairness

class UD_gammaSubgroup(_elem):
    @staticmethod
    def _corrected_neg(g_Cm):
        # fp/(fp+tn)
        # return g_Cm[1] / check_zero(tmp)
        tmp = g_Cm[1] + g_Cm[3]
        return zero_division(g_Cm[1], tmp)

    @classmethod
    def _core(cls, g1_Cm, g0_Cm):
        # g1 = cls._corrected_neg(g1_Cm)
        g0 = cls._corrected_neg(g0_Cm)
        z_Cm = [i + j for i, j in zip(g1_Cm, g0_Cm)]
        z = cls._corrected_neg(z_Cm)
        # n1, n0 = sum(g1_Cm), sum(g0_Cm)
        # n = float(n1 + n0)
        n = sum(z_Cm)  # sum(g1_Cm)+sum(g0_Cm)
        beta_f = abs(g0 - z)
        alph_f = (g0_Cm[1] + g0_Cm[3]) / n
        return alph_f * beta_f, alph_f, beta_f

    @classmethod
    @fantasy_timer
    def bival(cls, y, y_hat, priv_idx, pos_label=1):
        g1_Cm, g0_Cm = marginalised_np_mat(
            y, y_hat, pos_label, priv_idx)
        return cls._core(g1_Cm, g0_Cm)

    @classmethod
    @fantasy_timer
    def mu_sp(cls, y, y_hat, A, priv_val=1, pos_label=1):
        g1_Cm, g0_Cm = marginalised_np_mat(
            y, y_hat, pos_label, A == priv_val)
        return cls._core(g1_Cm, g0_Cm)[0]

    @classmethod
    @fantasy_timer
    def mu_cx(cls, y, y_hat, A, priv_val=1, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, priv_val, pos_label)
        g = [cls._corrected_neg(k) for k in g_Cm]
        z_Cm = contingency_tab_bi(y, y_hat, pos_label)
        z = cls._corrected_neg(z_Cm)
        n = float(sum(ex))

        ans = alt = 0.
        tmp, nt, _ = cls._indices(vA, idx, ex)
        for i in tmp:
            alph_f = (g_Cm[i][1] + g_Cm[i][3]) / n
            beta_f = abs(g[i] - z)
            ans += alph_f * beta_f * ex[i] / n
            alt += alph_f * beta_f * ex[i] / nt
        return ans, alt

    @classmethod
    @fantasy_timer
    def yev_sp(cls, y, y_hat, A, vals_in_A, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, vals_in_A[0], pos_label)
        g = [cls._corrected_neg(k) for k in g_Cm]
        z_Cm = contingency_tab_bi(y, y_hat, pos_label)
        z = cls._corrected_neg(z_Cm)
        n, n_a = float(sum(ex)), len(vA)
        ans = alt = 0.
        for i in range(n_a):
            alph_f = (g_Cm[i][1] + g_Cm[i][3]) / n
            beta_f = abs(g[i] - z)
            ans += alph_f * beta_f
            alt += alph_f * beta_f * ex[i] / n
        # ans /= n_a  ## len(vals_in_A)
        return ans / float(n_a), alt

    @classmethod
    @fantasy_timer
    def yev_cx(cls, y, y_hat, A, vals_in_Ai, pos_label=1):
        return NotImplementedError


# Definition 2.16 (Bounded group loss)

class UD_BoundedGrpLos(_elem):
    @classmethod
    def _core(cls, g1_Cm, g0_Cm):
        g1 = cls._los4accuracy(g1_Cm)
        g0 = cls._los4accuracy(g0_Cm)
        n1, n0 = sum(g1_Cm), sum(g0_Cm)
        n = float(n1 + n0)
        ans = 0.
        ans += g1 * n1 / n
        ans += g0 * n0 / n
        return ans, g1, g0

    @classmethod
    @fantasy_timer
    def bival(cls, y, y_hat, priv_idx, pos_label=1):
        g1_Cm, g0_Cm = marginalised_np_mat(
            y, y_hat, pos_label, priv_idx)
        return cls._core(g1_Cm, g0_Cm)

    @classmethod
    @fantasy_timer
    def mu_sp(cls, y, y_hat, A, priv_val=1, pos_label=1):
        g1_Cm, g0_Cm = marginalised_np_mat(
            y, y_hat, pos_label, A == priv_val)
        return cls._core(g1_Cm, g0_Cm)[0]

    @staticmethod
    def _los4accuracy(g_Cm):
        tmp = g_Cm[0] + g_Cm[3]
        tmp = zero_division(tmp, sum(g_Cm))
        return 1. - tmp

    @classmethod
    @fantasy_timer
    def mu_cx(cls, y, y_hat, A, priv_val=1, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, priv_val, pos_label)
        g = [cls._los4accuracy(k) for k in g_Cm]
        ans, n = 0., float(sum(ex))  # =len(y | y_hat)
        for i in range(len(vA)):
            ans += g[i] * ex[i] / n
        return ans

    @classmethod
    @fantasy_timer
    def yev_sp(cls, y, y_hat, A, vals_in_A, pos_label=1):
        g_Cm, vA, idx, ex = marginalised_np_gen(
            y, y_hat, A, vals_in_A[0], pos_label)
        g = [cls._los4accuracy(k) for k in g_Cm]
        return float(np.mean(g))

    @classmethod
    @fantasy_timer
    def yev_cx(cls, y, y_hat, A, vals_in_Ai, pos_label=1):
        return NotImplementedError
