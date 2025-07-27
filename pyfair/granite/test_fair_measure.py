# coding: utf-8


from pyfair.facil.utils_const import (
    synthetic_clf, synthetic_set, judge_transform_need, check_equal)
from pyfair.marble.metric_fair import (  # hfm.metrics.fairness_grp
    marginalised_pd_mat, prev_unpriv_manual, prev_unpriv_unaware,
    prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr)
from pyfair.marble.metric_fair import (  # hfm.metrics.fair_grp_ext
    marginalised_np_mat, unpriv_manual, unpriv_unaware,
    unpriv_group_one, unpriv_group_two, unpriv_group_thr,
    zero_division, calc_fair_group, StatsParity_sing,
    extGrp1_DP_sing, extGrp2_EO_sing, extGrp3_PQP_sing, alterGrps_sing)
# from experiment.utils.fair_rev_group import (
#     UD_grp1_DP, UD_grp2_EO, UD_grp3_PQP)
# from hfm.utils.simulator import synthetic_clf, synthetic_set
# from hfm.utils.verifiers import judge_transform_need, check_equal


import numpy as np
import pdb
# import lightgbm

from pyfair.facil.utils_const import check_equal
# from pyfair.marble.metric_fair import (
#     marginalised_np_mat, marginalised_np_gen)  # addtl,addl
from pyfair.granite.fair_meas_indiv import (
    GEI_Theil,  # prop_L_fair, prop_L_loss,
    DistDirect)  # HFM_Approx_bin, HFM_DistApprox)
from pyfair.granite.fair_meas_group import (
    UD_grp1_DP, UD_grp1_DisI, UD_grp1_DisT,
    UD_grp2_EO, UD_grp2_EOdd, UD_grp2_PEq,
    UD_grp3_PQP, UD_gammaSubgroup, UD_BoundedGrpLos)

# from pyfair.datasets import PropublicaViolentRecidivism
# from pyfair.preprocessing_hfm import (
#     renewed_prep_and_adversarial, renewed_transform_X_A_and_y,
#     check_marginalised_indices)
# from pyfair.preprocessing_dr import transform_unpriv_tag


n = 110
y = np.random.randint(2, size=n)
y_hat = np.random.randint(2, size=n)

nc, na = 3, 2
A = np.random.randint(nc, size=(n, na)) + 1
idx_Ai_Sjs = [[A[:, i] == j + 1 for j in range(
    nc)] for i in range(na)]
idx_Sjs = [A[:, 0] == j + 1 for j in range(nc)]

A_bin = A.copy()
A_bin[A_bin != 1] = 0
idx_a = [[A_bin[:, i] == 1, A_bin[:, i] != 1] for i in range(na)]
idx_ai = idx_a[0]
pos, priv_val = 1, 1,
A_i, priv_idx = A[:, 1], A[:, 1] == 1
vals_in_Ai = list(set(A_i))
A1_bin, val_A1 = A_bin[:, 1], [1, 0]

hfm_idx_nsa_bin = [idx_Ai_Sjs[1][0], ~idx_Ai_Sjs[1][0]]


"""
ds = PropublicaViolentRecidivism()
df = ds.load_raw_dataset()
(origin_dat, processed_dat, process_mult, perturbed_dat, perturb_mult
 ) = renewed_prep_and_adversarial(ds, df, .97, None)
processed_Xy = process_mult['numerical-multisen']
perturbed_Xy = perturb_mult['numerical-multisen']
X, A, y, _ = renewed_transform_X_A_and_y(ds, processed_Xy, False)
_, Aq, _, _ = renewed_transform_X_A_and_y(ds, perturbed_Xy, False)
# tmp = processed_dat['original'][ds.label_name]
sen_att = ds.get_sensitive_attrs_with_joint()[: 2]
priv_val = ds.get_privileged_group_with_joint('')[: 2]
marginalised_grp = origin_dat['marginalised_groups']
margin_indices = check_marginalised_indices(
    processed_dat['original'], sen_att, priv_val,
    marginalised_grp)
# new_attr = '-'.join(sen_att) if len(sen_att) > 1 else None
# belongs_priv, ptb_with_joint = transform_unpriv_tag(
#     ds, processed_dat['original'], 'both')

X_and_A = np.concatenate([X, A], axis=1)
X_and_Aq = np.concatenate([X, Aq], axis=1)
clf = lightgbm.LGBMClassifier(n_estimators=7)
clf.fit(X_and_A, y)
clf.fit(X_and_A, y)
clf.fit(X_and_A, y)
y_hat = clf.predict(X_and_A)
pos = ds.get_positive_class_val('')
y, priv_val = y.values, 1
A_i = A['race'].values           # A[:, 1]
priv_idx = margin_indices[1][0]  # A_i == priv_val[1]
vals_in_Ai = list(set(A_i))
# pdb.set_trace()
del ds, df, origin_dat, processed_Xy, perturbed_Xy  # , tmp
del processed_dat, process_mult, perturbed_dat, perturb_mult
# del new_attr, belongs_priv, ptb_with_joint
acc = (y == y_hat).mean()
"""


def test_metric_grp1():
    m1 = UD_grp1_DP.bival(y, y_hat, priv_idx, pos)
    m2 = UD_grp1_DP.mu_sp(y, y_hat, A_i, priv_val, pos)
    m3 = UD_grp1_DP.mu_cx(y, y_hat, A_i, priv_val, pos)
    # assert m1[0][0] == m2[0] < m3[0][0]  # fp =
    tm1, _ = extGrp1_DP_sing(y, y_hat, hfm_idx_nsa_bin, pos)
    tm2, _ = extGrp1_DP_sing(y, y_hat, idx_Ai_Sjs[1], pos)
    tm3, _ = alterGrps_sing(tm2[-1], idx_Ai_Sjs[1])
    assert tm1[0] <= tm2[0] < tm3[0]

    m4 = UD_grp1_DP.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    m5 = UD_grp1_DP.yev_cx(y, y_hat, A_i, vals_in_Ai, pos)
    m6 = UD_grp1_DP.yev_cx(y, y_hat, A1_bin, val_A1, pos)
    m7 = UD_grp1_DP.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    m8 = UD_grp1_DP.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(m1[0][0], [m2[0], m4[0][3][0]])  # m3[0][0],
    assert check_equal(m6[0][2][0], [m7[0], m8[0][1]])  # m8[0][0]])

    qa_1 = UD_grp1_DisI.bival(y, y_hat, priv_idx, pos)
    qa_2 = UD_grp1_DisI.mu_sp(y, y_hat, A_i, priv_val, pos)
    qa_3 = UD_grp1_DisI.mu_cx(y, y_hat, A_i, priv_val, pos)

    qa_4 = UD_grp1_DisI.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    qa_5 = UD_grp1_DisI.yev_cx(y, y_hat, A_i, vals_in_Ai, pos)
    qa_6 = UD_grp1_DisI.yev_cx(y, y_hat, A1_bin, val_A1, pos)
    qa_7 = UD_grp1_DisI.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    qa_8 = UD_grp1_DisI.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(qa_1[0][0], [qa_2[0], qa_4[0][3][0]])
    # assert check_equal(qa_6[0][2][0], [qa_7[0], qa_8[0][1]])
    assert check_equal(qa_7[0], qa_8[0][1])

    qb_1 = UD_grp1_DisT.bival(y, y_hat, priv_idx, pos)
    qb_2 = UD_grp1_DisT.mu_sp(y, y_hat, A_i, priv_val, pos)
    qb_3 = UD_grp1_DisT.mu_cx(y, y_hat, A_i, priv_val, pos)

    qb_5_a = UD_grp1_DisT.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    qb_5_b = UD_grp1_DisT.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    qb_4 = UD_grp1_DisT.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    qb_6 = UD_grp1_DisT.yev_sp(y, y_hat, A1_bin, val_A1, pos)
    qb_7 = UD_grp1_DisT.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    qb_8 = UD_grp1_DisT.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(qb_1[0][0], [qb_2[0], qb_7[0], qb_8[0],
                                    qb_5_a[0], qb_5_b[0]])

    # pdb.set_trace()
    return


def test_metric_grp2():
    m1 = UD_grp2_EO.bival(y, y_hat, priv_idx, pos)
    m2 = UD_grp2_EO.mu_sp(y, y_hat, A_i, priv_val, pos)
    m3 = UD_grp2_EO.mu_cx(y, y_hat, A_i, priv_val, pos)
    # assert m1[0][0] == m2[0] < m3[0][0]  # fp =
    tm1, _ = extGrp2_EO_sing(y, y_hat, hfm_idx_nsa_bin, pos)
    tm2, _ = extGrp2_EO_sing(y, y_hat, idx_Ai_Sjs[1], pos)
    tm3, _ = alterGrps_sing(tm2[-1], idx_Ai_Sjs[1])
    assert tm1[0] <= tm2[0] < tm3[0]

    m4 = UD_grp2_EO.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    m5 = UD_grp2_EO.yev_cx(y, y_hat, A_i, vals_in_Ai, pos)
    m6 = UD_grp2_EO.yev_cx(y, y_hat, A1_bin, val_A1, pos)
    m7 = UD_grp2_EO.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    m8 = UD_grp2_EO.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(m1[0][0], [m2[0], m4[0][3][0]])
    assert check_equal(m6[0][2][0], [m7[0], m8[0][1]])

    qa_1 = UD_grp2_EOdd.bival(y, y_hat, priv_idx, pos)
    qa_2 = UD_grp2_EOdd.mu_sp(y, y_hat, A_i, priv_val, pos)
    qa_3 = UD_grp2_EOdd.mu_cx(y, y_hat, A_i, priv_val, pos)

    qa_4 = UD_grp2_EOdd.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    qa_5 = UD_grp2_EOdd.yev_cx(y, y_hat, A_i, vals_in_Ai, pos)
    qa_6 = UD_grp2_EOdd.yev_cx(y, y_hat, A1_bin, val_A1, pos)
    qa_7 = UD_grp2_EOdd.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    qa_8 = UD_grp2_EOdd.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    # assert check_equal(qa_1[0][0], [qa_2[0][0], qa_4[0][3][0]])
    assert check_equal(qa_1[0][0], [qa_2[0], qa_4[0][3][0]])
    # assert check_equal(qa_6[0][2][0], [qa_7[0], qa_8[0][0]])
    # TODO
    assert check_equal(qa_2[0], [qa_7[0], qa_8[0][1], qa_6[0][1]])

    qb_1 = UD_grp2_PEq.bival(y, y_hat, priv_idx, pos)
    qb_2 = UD_grp2_PEq.mu_sp(y, y_hat, A_i, priv_val, pos)
    qb_3 = UD_grp2_PEq.mu_cx(y, y_hat, A_i, priv_val, pos)

    qb_4 = UD_grp2_PEq.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    qb_6 = UD_grp2_PEq.yev_sp(y, y_hat, A1_bin, val_A1, pos)
    qb_7 = UD_grp2_PEq.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    qb_8 = UD_grp2_PEq.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(qb_1[0][0], qb_2[0])  # [qb_2[0], qb_4[0][3][0]])
    assert check_equal(qb_2[0], [qb_7[0], qb_8[0]])

    # pdb.set_trace()
    return


def test_metric_grp3():
    m1 = UD_grp3_PQP.bival(y, y_hat, priv_idx, pos)
    m2 = UD_grp3_PQP.mu_sp(y, y_hat, A_i, priv_val, pos)
    m3 = UD_grp3_PQP.mu_cx(y, y_hat, A_i, priv_val, pos)
    # assert m1[0][0] == m2[0] < m3[0][0]  # fp =
    # tm1, _ = extGrp3_PQP_sing(y, y_hat, [
    #     idx_Ai_Sjs[1][0], ~idx_Ai_Sjs[1][0]], pos)
    tm1, _ = extGrp3_PQP_sing(y, y_hat, hfm_idx_nsa_bin, pos)
    tm2, _ = extGrp3_PQP_sing(y, y_hat, idx_Ai_Sjs[1], pos)
    tm3, _ = alterGrps_sing(tm2[-1], idx_Ai_Sjs[1])
    # assert tm1[0][0] < tm2[0][0] < tm3[0]
    assert tm1[0] <= tm2[0] < tm3[0]

    m4 = UD_grp3_PQP.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    # assert check_equal(m1[0], [m2[0], m3[0][0], m4[0][3][0]])
    m5 = UD_grp3_PQP.yev_cx(y, y_hat, A_i, vals_in_Ai, pos)
    m6 = UD_grp3_PQP.yev_cx(y, y_hat, A1_bin, val_A1, pos)
    m7 = UD_grp3_PQP.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    m8 = UD_grp3_PQP.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(m1[0][0], [m2[0], m4[0][3][0]])
    assert check_equal(m6[0][2][0], [m7[0], m8[0][1]])

    # pdb.set_trace()
    return


def test_metric_indv():
    m1 = UD_gammaSubgroup.bival(y, y_hat, priv_idx, pos)
    m2 = UD_gammaSubgroup.mu_sp(y, y_hat, A_i, priv_val, pos)
    m3 = UD_gammaSubgroup.mu_cx(y, y_hat, A_i, priv_val, pos)
    m4 = UD_gammaSubgroup.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    m7 = UD_gammaSubgroup.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    m8 = UD_gammaSubgroup.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(m1[0][0], [m2[0], m7[0], m8[0][1]])

    m1 = UD_BoundedGrpLos.bival(y, y_hat, priv_idx, pos)
    m2 = UD_BoundedGrpLos.mu_sp(y, y_hat, A_i, priv_val, pos)
    m3 = UD_BoundedGrpLos.mu_cx(y, y_hat, A_i, priv_val, pos)
    m4 = UD_BoundedGrpLos.yev_sp(y, y_hat, A_i, vals_in_Ai, pos)
    m7 = UD_BoundedGrpLos.mu_sp(y, y_hat, A1_bin, priv_val, pos)
    m8 = UD_BoundedGrpLos.mu_cx(y, y_hat, A1_bin, priv_val, pos)
    assert check_equal(m1[0][0], [m2[0], m3[0], m7[0], m8[0]])
    assert check_equal(m2[0], m3[0])

    t1, _ = GEI_Theil.get_GEI(y, y_hat, alpha=.5)
    t2, _ = GEI_Theil.get_Theil(y, y_hat)
    assert 0 <= t1 <= 1 and 0. <= t2 <= 1.
    X_nA_y = np.concatenate([
        np.random.randint(2, size=n).reshape(-1, 1).astype('float'),
        np.random.rand(n, 3), ], axis=1)
    t1, _ = DistDirect.bin(X_nA_y, idx_Ai_Sjs[1][0])  # hfm_idx_nsa_bin[0])
    # t2 = DistDirect.bin(X_nA_y, idx_Ai_Sjs[1][0])  #,idx_Ai_Sjs[1]) #A_i)
    t3, _ = DistDirect.nonbin(X_nA_y, hfm_idx_nsa_bin)
    t4, _ = DistDirect.nonbin(X_nA_y, idx_Ai_Sjs[1])
    t5, _ = DistDirect.multivar(X_nA_y, [hfm_idx_nsa_bin])
    t6, _ = DistDirect.multivar(X_nA_y, [idx_Ai_Sjs[1]])
    assert check_equal(t1, t3) and check_equal(t1, t5[:-1])
    assert check_equal(t4, t6[:-1])

    # m1, m2 = 3, 5
    # tm1, _ = HFM_Approx_bin.bin(X_nA_y, A_i, idx_Ai_Sjs[1][0], m1, m2)
    # tm2, _ = HFM_DistApprox.nonbin(  # .bin(
    #     X_nA_y, idx_Ai_Sjs[1][0].astype('int'), m1, m2)
    # tm3, _ = HFM_DistApprox.nonbin(X_nA_y, A_i, m1, m2)
    # tm4, _ = HFM_DistApprox.multivar(X_nA_y, A_i.reshape(-1, 1), m1, m2)
    # tm5, _ = HFM_DistApprox.multivar(
    #     X_nA_y, idx_Ai_Sjs[1][0].astype('int').reshape(-1, 1), m1, m2)
    # assert tm1 >= t1[0]
    # assert tm2[0] >= t3[0] and tm2[1] >= t3[1]
    # assert tm3[0] >= t4[0] and tm3[1] >= t4[1]
    # assert tm4[0] >= t6[0] and tm4[1] >= t6[1]
    # assert tm5[0] >= t5[0] and tm5[1] >= t5[1]
    # # pdb.set_trace()
    return


# ==========================
# --------------------------
# hfm/metrics/test_fairness.py


nb_spl, nb_lbl, nb_clf = 371, 3, 2  # nb_clf=7
y_bin, _, _ = synthetic_set(2, nb_spl, nb_clf)
y_non, _, _ = synthetic_set(nb_lbl, nb_spl, nb_clf)
ht_bin = synthetic_clf(y_bin, nb_clf, err=.4)
ht_non = synthetic_clf(y_non, nb_clf, err=.4)

idx_priv = np.random.randint(2, size=nb_spl, dtype='bool')
idx_Sjs = [idx_priv == 1, idx_priv == 0]
multi_pv = np.random.randint(3, size=nb_spl, dtype='int')
Sjs_bin = [multi_pv == 1, multi_pv != 1]
Sjs_non = [multi_pv == 1, multi_pv == 0, multi_pv == 2]


def test_group_fair():
    def subroutine(y, hx, pos, priv):
        vY, dY = judge_transform_need(y)
        vY = vY[:: -1]
        z, ht = np.array(y), np.array(hx)
        g1M, g0M = marginalised_np_mat(z, ht, pos, priv)
        _, _, c1, c0 = marginalised_pd_mat(z, ht, pos, priv)

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
        res = StatsParity_sing(ht, idx_Sjs, pos)[0]
        tmp = extGrp1_DP_sing(z, ht, idx_Sjs, pos)[0][: -1]
        assert check_equal(res, tmp)
        assert 0. <= ans <= 1.

        # pdb.set_trace()
    subroutine(y_bin, ht_bin[0], 1, idx_priv)
    subroutine(y_non, ht_non[0], 1, idx_priv)
    return


def test_fair_update():
    vY, dY = judge_transform_need(y_bin)
    vY = vY[:: -1]
    z, ht = np.array(y_bin), np.array(ht_bin[0])
    g1, g0 = marginalised_np_mat(z, ht, 1, Sjs_bin[0])

    just_one = unpriv_group_one(g1, g0)
    just_two = unpriv_group_two(g1, g0)
    just_thr = unpriv_group_thr(g1, g0)
    ans_1 = calc_fair_group(*just_one)
    ans_2 = calc_fair_group(*just_two)
    ans_3 = calc_fair_group(*just_thr)
    val_inA = [1, 0, 2]

    def sub_update(ans, pos, priv, func, extcls, grpstr):
        tmp_1 = func(z, ht, Sjs_bin, pos)[0]
        tmp_2 = func(z, ht, Sjs_non, pos)[0]
        res_0 = extcls.bival(z, ht, Sjs_bin[0], pos)[0]
        res_1 = extcls.mu_sp(z, ht, priv, 1, pos)[0]
        res_2 = extcls.mu_cx(z, ht, priv, 1, pos)[0]
        res_3 = extcls.yev_sp(z, ht, priv, val_inA, pos)[0]
        res_4 = extcls.yev_cx(z, ht, priv, val_inA, pos)[0]

        assert len(tmp_1[-1]) == 2 and len(tmp_2[-1]) == 3
        assert ans == res_0[0] == res_1
        # pdb.set_trace()
        # if grpstr in ('DP'):
        #     assert check_equal(res_1, res_2[1])

    sub_update(ans_1, 1, multi_pv, extGrp1_DP_sing, UD_grp1_DP, 'DP')
    sub_update(ans_2, 1, multi_pv, extGrp2_EO_sing, UD_grp2_EO, 'EO')
    sub_update(ans_3, 1, multi_pv, extGrp3_PQP_sing, UD_grp3_PQP, 'PP')
    return
