# coding: utf-8

import sklearn.metrics as metrics
import numpy as np
# import pdb
from pyfair.facil.utils_const import (
    check_equal, judge_transform_need,
    synthetic_dat, synthetic_clf, synthetic_set)


from pyfair.marble.metric_perf import (
    # from hfm.metrics.excl_perf_bin import (
    # contingency_tab_bi, contingency_tab_mu,
    calc_accuracy, calc_precision, calc_recall,
    calc_f1_score, calc_f_beta)
from pyfair.facil.metric_cont import (
    contingency_tab_bi, contg_tab_mu_type3)
contingency_tab_mu = contg_tab_mu_type3
del contg_tab_mu_type3


# ==========================
# Metrics
# hfm/metrics/performance.py
# test_performance.py
# --------------------------
# hfm/metrics/test_perf_bin.py


n = 201  # n, nc = 110, 3
y = np.random.randint(2, size=n)
yp_hat = np.random.randint(2, size=n)
# z = np.random.randint(nc, size=n)
# z_hat = np.random.randint(nc, size=n)
vY = [1, 0]


def test_contingency_bi():
    tmp = metrics.confusion_matrix(y, yp_hat)
    ans = metrics.confusion_matrix(y, yp_hat, labels=vY)

    assert tmp[0, 0] == ans[1, 1]
    assert tmp[1, 1] == ans[0, 0]
    assert tmp[0, 1] == ans[1, 0]
    assert tmp[1, 0] == ans[0, 1]

    res = contingency_tab_bi(y, yp_hat, pos=vY[0])
    # pdb.set_trace()
    tp, fp, fn, tn = res
    res = (tp, fn, fp, tn)
    assert np.equal(ans.ravel(), res).all()

    res = contingency_tab_mu(y, yp_hat, vY)
    assert np.equal(ans, res).all()
    return


def test_perf_bin():
    # pos = vY[0]
    cm_bin = contingency_tab_bi(y, yp_hat, pos=vY[0])
    cm_non = contingency_tab_mu(y, yp_hat, vY)
    cm = metrics.confusion_matrix(y, yp_hat, labels=vY)
    tp, fp, fn, tn = cm_bin
    cm_alt = (tp, fn, fp, tn)
    assert np.equal(cm_alt, cm.ravel()).all()
    # assert np.equal(cm_bin, cm.ravel()).all()
    assert np.equal(cm_non, cm).all()

    ans_1 = metrics.accuracy_score(y, yp_hat)       # float
    ans_2 = metrics.precision_score(y, yp_hat)      # np.float64
    ans_3 = metrics.recall_score(y, yp_hat)         # np.float64
    ans_4 = metrics.f1_score(y, yp_hat)             # np.float64
    ans_5 = metrics.fbeta_score(y, yp_hat, beta=2)  # np.float64

    res_1 = calc_accuracy(*cm_bin)
    p = calc_precision(*cm_bin)
    r = calc_recall(*cm_bin)
    res_4 = calc_f1_score(*cm_bin)
    res_5 = calc_f_beta(p, r, beta=2)

    assert ans_1 == res_1
    # if ans_2 != p:
    #     pdb.set_trace()
    assert ans_2 == p
    assert ans_3 == r
    assert check_equal(ans_4, res_4)  # assert ans_4==res_4
    assert check_equal(ans_5, res_5)  # assert ans_5==res_5
    assert check_equal(ans_4, res_4, 10**8)
    assert check_equal(ans_5, res_5, 10**8)
    return


# =====================================
# discriminative risk

nb_spl, nb_feat, nb_lbl = 121, 7, 3
X_trn, y_trn = synthetic_dat(nb_lbl, nb_spl, nb_feat)
y_hat, hx_qtb = synthetic_clf(y_trn, 2, err=.2, prng=None)
z_hat, hz_qtb = synthetic_clf(y_trn, 2, err=.3, prng=None)


def test_my_DR():
    # def excl_test_my_DR():
    # from fairml.discriminative_risk import (
    from pyfair.dr_hfm.discriminative_risk import (
        hat_L_fair, hat_L_loss, tandem_fair, tandem_loss,
        hat_L_objt, tandem_objt,
        cal_L_obj_v1, cal_L_obj_v2,  # L_fair_MV_rho, L_loss_MV_rho,
        # E_rho_L_fair_f, E_rho_L_loss_f, Erho_sup_L_fair,
        # Erho_sup_L_loss, ED_Erho_I_fair, ED_Erho_I_loss,
        perturb_numpy_ver, perturb_pandas_ver)  # disturb_slightly)

    ans = hat_L_fair(y_hat, hx_qtb)
    res = hat_L_loss(y_hat, y_trn)
    err = float(1. - np.mean(np.equal(y_trn, y_hat)))
    assert check_equal(err, res)

    ans = tandem_fair(y_hat, hx_qtb, z_hat, hz_qtb)
    res = tandem_loss(y_hat, z_hat, y_trn)
    lam = .5
    ans = hat_L_objt(y_hat, hx_qtb, y_trn, lam)
    assert 0. <= ans <= 1.
    ans = tandem_objt(y_hat, hx_qtb, z_hat, hz_qtb, y_trn, lam)
    assert 0. <= ans <= 1.

    nb_cls = 5
    # tmp = synthetic_clf(y_trn, nb_cls * 2, err=.15)
    # yt_hat, hx_hat = tmp[:nb_cls], tmp[-nb_cls:]
    # tmp = synthetic_clf(y_trn, nb_cls * 2, err=.2)
    # yt_hat_qtb, hx_hat_qtb = tmp[:nb_cls], tmp[nb_cls:]
    yt_hat = synthetic_clf(y_trn, nb_cls, err=.15)
    yt_hat_qtb = synthetic_clf(y_trn, nb_cls, err=.2)
    coef = np.random.rand(nb_cls)
    coef /= np.sum(coef)
    coef = coef.tolist()
    res = cal_L_obj_v2(yt_hat, yt_hat_qtb, y_trn, coef)
    ans = cal_L_obj_v1(yt_hat, yt_hat_qtb, y_trn, coef)
    assert check_equal(res, ans)  # res == ans
    import pandas as pd

    X = np.random.randint(5, size=(5, 4))  # .tolist()
    # X_qtb = disturb_slightly(X, sen=[2], ratio=.5)
    # X, X_qtb = np.array(X), np.array(X_qtb)
    X_qtb = perturb_numpy_ver(X, [2, 1], [1, 1], ratio=.97)
    assert np.all(np.equal(X[:, [0, 3]], X_qtb[:, [0, 3]]))
    assert not np.equal(X, X_qtb).all()

    X = pd.DataFrame(X, columns=['A', 'B', 'C', 'D'])
    X_qtb = perturb_pandas_ver(X, ['B', 'C'], [1, 0], ratio=.97)
    tmp = (X[['A', 'D']] == X_qtb[['A', 'D']]).all()
    assert tmp.to_numpy().all()
    assert not (X == X_qtb).to_numpy().all()

    # pdb.set_trace()
    return


# =====================================
# metric_perf.py


# binary classification
_, z_trn = synthetic_dat(2, nb_spl, nb_feat)
z_hat, _ = synthetic_clf(z_trn, 2, err=.2, prng=None)

y_trn, y_hat = np.array(y_trn), np.array(y_hat)
z_trn, z_hat = np.array(z_trn), np.array(z_hat)


def test_contingency():
    from pyfair.facil.metric_cont import (
        contingency_tab_bi, contg_tab_mu_type3)

    ans = metrics.cluster.contingency_matrix(y_trn, y_hat)
    res = contg_tab_mu_type3(y_trn, y_hat, list(range(nb_lbl)))
    assert np.all(np.equal(ans, res))
    ans = metrics.cluster.contingency_matrix(z_trn, z_hat)
    res = contg_tab_mu_type3(z_trn, z_hat, [0, 1])
    assert np.all(np.equal(ans, res))

    res = contingency_tab_bi(z_trn, z_hat, pos=1)
    assert res[0] == ans[1, 1]
    assert res[-1] == ans[0, 0]
    assert ans[0, 1] == res[1]  # fp
    assert ans[1, 0] == res[2]  # fn
    return


def test_performance():
    from pyfair.facil.metric_cont import contingency_tab_bi
    from pyfair.marble.metric_perf import (
        calc_accuracy, calc_precision, calc_recall,
        calc_f1_score, calc_f_beta, calc_error_rate)

    mid = contingency_tab_bi(z_trn, z_hat, pos=1)
    res = calc_accuracy(*mid)
    assert res == metrics.accuracy_score(z_trn, z_hat)
    assert check_equal(res, 1. - calc_error_rate(*mid))

    p = calc_precision(*mid)
    assert p == metrics.precision_score(z_trn, z_hat)
    r = calc_recall(*mid)
    assert r == metrics.recall_score(z_trn, z_hat)
    res = calc_f1_score(*mid)
    assert check_equal(res, metrics.f1_score(z_trn, z_hat))

    res = calc_f_beta(p, r, beta=1)
    assert check_equal(
        res, metrics.fbeta_score(z_trn, z_hat, beta=1))
    res = calc_f_beta(p, r, beta=2)
    assert check_equal(
        res, metrics.fbeta_score(z_trn, z_hat, beta=2))
    # assert res == metrics.fbeta_score(z_trn, z_hat, beta=2)
    # pdb.set_trace()
    return


# =====================================
# metric_fair.py


nb_spl, nb_lbl, nb_clf = 371, 3, 2  # nb_clf=7
y_bin, _, _ = synthetic_set(2, nb_spl, nb_clf)
y_non, _, _ = synthetic_set(nb_lbl, nb_spl, nb_clf)
ht_bin = synthetic_clf(y_bin, nb_clf, err=.4)
ht_non = synthetic_clf(y_non, nb_clf, err=.4)

idx_priv = np.random.randint(2, size=nb_spl, dtype='bool')
idx_Sjs = [idx_priv == 1, idx_priv == 0]
A_j = np.random.randint(3, size=nb_spl, dtype='int')
Sjs_bin = [idx_priv == 1, idx_priv != 1]
Sjs_non = [idx_priv == 1, idx_priv == 0, idx_priv == 2]


def test_group_fair():
    from pyfair.marble.metric_fair import (
        # unpriv_grp_one, unpriv_grp_two, unpriv_grp_thr,
        unpriv_group_one, unpriv_group_two, unpriv_group_thr,
        marginalised_np_mat, unpriv_unaware, unpriv_manual,
        calc_fair_group, StatsParity_sing,  # StatsParity_mult,
        zero_division, alterGrps_sing,  # alterGroups_pl,
        extGrp1_DP_sing, extGrp2_EO_sing, extGrp3_PQP_sing,
        # extGrp1_DP_pl, extGrp2_EO_pl, extGrp3_PQP_pl,
        DPext_alterSP,  # extDP_SPalter,
        marginalised_pd_mat, prev_unpriv_manual, prev_unpriv_unaware,
        prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr)

    def sub_extra(prev, z, ht, pos, Sjs_bin, Sjs_non):
        res_bi, _ = StatsParity_sing(ht, Sjs_bin, pos)
        res_mu, _ = StatsParity_sing(ht, Sjs_non, pos)
        tmp_bi, _ = DPext_alterSP(z, ht, Sjs_bin, pos)
        tmp_mu, _ = DPext_alterSP(z, ht, Sjs_non, pos)
        assert check_equal(prev[0], [  # tmp_mu[-1][0],
            tmp_bi[0], tmp_bi[1], tmp_bi[-1][0], tmp_mu[-1][0], ])

        ans_1b, _ = extGrp1_DP_sing(z, ht, Sjs_bin, pos)
        ans_1m, _ = extGrp1_DP_sing(z, ht, Sjs_non, pos)
        ans_2b, _ = extGrp2_EO_sing(z, ht, Sjs_bin, pos)
        ans_2m, _ = extGrp2_EO_sing(z, ht, Sjs_non, pos)
        ans_3b, _ = extGrp3_PQP_sing(z, ht, Sjs_bin, pos)
        ans_3m, _ = extGrp3_PQP_sing(z, ht, Sjs_non, pos)
        ans_1b_alt, _ = alterGrps_sing(ans_1b[-1], Sjs_bin)
        ans_1m_alt, _ = alterGrps_sing(ans_1m[-1], Sjs_non)
        ans_2b_alt, _ = alterGrps_sing(ans_2b[-1], Sjs_bin)
        ans_2m_alt, _ = alterGrps_sing(ans_2m[-1], Sjs_non)
        ans_3b_alt, _ = alterGrps_sing(ans_3b[-1], Sjs_bin)
        ans_3m_alt, _ = alterGrps_sing(ans_3m[-1], Sjs_non)

        assert ans_1b_alt[0] == ans_1b_alt[1] == prev[0]
        assert ans_2b_alt[0] == ans_2b_alt[1] == prev[1]
        assert ans_3b_alt[0] == ans_3b_alt[1] == prev[2]
        assert ans_1m_alt[0] > ans_1m_alt[1] > prev[0]
        assert ans_2m_alt[0] > ans_2m_alt[1] > prev[1]
        assert ans_3m_alt[0] > ans_3m_alt[1] > prev[2]
        assert ans_1m[0] > prev[0] > ans_1b[0]
        fp2 = ans_2m[0] > prev[1] > ans_2b[0]
        fp3 = ans_3m[0] > prev[2] > ans_3b[0]
        # if not fp2 or not fp3:  # not (fp2 and fp3):
        #     pdb.set_trace()
        assert fp2  # assert ans_2m[0] > ans_2b[0]
        assert fp3  # assert ans_3m[0] > ans_3b[0]

    def subroutine(y, hx, pos, A_j, Sjs_bin, Sjs_non):
        vY, _ = judge_transform_need(y)  # ,dY
        vY = vY[:: -1]
        z, ht = np.array(y), np.array(hx)  # priv=Sjs_bin[0]
        g1M, g0M = marginalised_np_mat(z, ht, pos, Sjs_non[0])
        _, _, c1, c0 = marginalised_pd_mat(z, ht, pos, Sjs_non[0])

        just_one = unpriv_group_one(g1M, g0M)
        just_two = unpriv_group_two(g1M, g0M)
        just_thr = unpriv_group_thr(g1M, g0M)
        just_zero = unpriv_unaware(g1M, g0M)
        just_four = unpriv_manual(g1M, g0M)
        assert check_equal(just_one, prev_unpriv_grp_one(c1, c0))
        assert check_equal(just_two, prev_unpriv_grp_two(c1, c0))
        assert check_equal(just_thr, prev_unpriv_grp_thr(c1, c0))
        assert check_equal(just_zero, prev_unpriv_unaware(c1, c0))
        assert check_equal(just_four, prev_unpriv_manual(c1, c0))

        assert zero_division(0., 0.) == 0.
        assert zero_division(1., 0.) == 10
        assert zero_division(1.5, 0.2) == 7.5
        ans = calc_fair_group(*just_one)
        res = StatsParity_sing(ht, Sjs_bin, pos)[0]
        tmp_1 = extGrp1_DP_sing(z, ht, Sjs_bin, pos)[0]
        assert check_equal(res, tmp_1[:-1])  # tmp[: -1])
        res = StatsParity_sing(ht, Sjs_non, pos)[0]
        tmp_2 = extGrp1_DP_sing(z, ht, Sjs_non, pos)[0]
        assert check_equal(res, tmp_2[:-1])  # tmp[: -1])

        # res = extDP_SPalter(z, ht, idx_Sjs, pos)[0]
        tmp_1 = DPext_alterSP(z, ht, Sjs_bin, pos)[0]
        tmp_2 = DPext_alterSP(z, ht, Sjs_non, pos)[0]
        assert check_equal(ans, [
            tmp_2[-1][0], tmp_1[-1][0], tmp_1[0], tmp_1[1], ])

        # pdb.set_trace()
        prev = [calc_fair_group(*just_one),
                calc_fair_group(*just_two),
                calc_fair_group(*just_thr),
                calc_fair_group(*just_zero),
                calc_fair_group(*just_four)]
        sub_extra(prev, z, ht, pos, Sjs_bin, Sjs_non)
    # subroutine(y_bin, ht_bin[0], 1, idx_priv)
    # subroutine(y_non, ht_non[0], 1, idx_priv)
    subroutine(y_bin, ht_bin[0], 1, A_j, Sjs_bin, Sjs_non)
    return


def test_group_prev():
    # from fairml.metrics.group_fair import (
    # from fairml.facils.fairness_group import (
    from pyfair.marble.metric_fair import (
        marginalised_contingency, marginalised_confusion,
        prev_unpriv_grp_one, prev_unpriv_grp_two,
        prev_unpriv_grp_thr, marginalised_pd_mat,
        prev_unpriv_manual, prev_unpriv_unaware)
    unpriv_group_one = prev_unpriv_grp_one
    unpriv_group_two = prev_unpriv_grp_two
    unpriv_group_thr = prev_unpriv_grp_thr
    unpriv_unaware = prev_unpriv_unaware
    unpriv_manual = prev_unpriv_manual

    def subroutine(y, pos, priv):
        vY, dY = judge_transform_need(y)  # + hx)
        vY = vY[:: -1]
        hx = np.random.randint(dY, size=nb_spl).tolist()
        Cij = marginalised_contingency(y, hx, vY, dY)
        Cm = marginalised_confusion(Cij, vY.index(pos))
        assert np.sum(Cm) == np.sum(Cij) == len(y)

        g1M, g0M, g1, g0 = marginalised_pd_mat(y, hx, pos, priv)
        assert np.sum(g1M) == np.sum(g1)
        assert np.sum(g0M) == np.sum(g0)
        assert np.sum(g1M) + np.sum(g0M) == len(y)

        just_one = unpriv_group_one(g1, g0)
        just_two = unpriv_group_two(g1, g0)
        just_thr = unpriv_group_thr(g1, g0)
        just_zero = unpriv_unaware(g1, g0)
        just_four = unpriv_manual(g1, g0)
        assert all([
            0 <= i <= 1 for i in just_one + just_two + just_thr])
        assert all([0 <= i <= 1 for i in just_zero + just_four])
        # pdb.set_trace()

    # idx_priv = np.random.randint(2, size=nb_spl, dtype='bool')
    subroutine(y_bin, 1, idx_priv)  # ht_bin[0],
    subroutine(y_non, 1, idx_priv)  # ht_non[0],
    return


# def test_my_DR():
#     return
