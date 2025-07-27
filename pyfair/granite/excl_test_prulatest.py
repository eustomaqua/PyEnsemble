# coding: utf-8
# ensem_prulately.py

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


# import pdb
import numpy as np
from pyfair.facil.utils_const import synthetic_set, synthetic_dat

nb_inst, nb_labl = 100, 5
nb_cls, nb_pru = 11, 3
bi_y, bi_yt, bi_c = synthetic_set(2, nb_inst, nb_cls)
mu_y, mu_yt, mu_c = synthetic_set(nb_labl, nb_inst, nb_cls)
tr_yt = np.array(bi_yt) * 2 - 1
tr_yt = tr_yt.tolist()
tr_y = [i * 2 - 1 for i in bi_y]

ki, kj = 0, 1  # idx
tr_ha, tr_hb = tr_yt[ki], tr_yt[kj]
bi_ha, bi_hb = bi_yt[ki], bi_yt[kj]
mu_ha, mu_hb = mu_yt[ki], mu_yt[kj]

# or: Sp, Sq
Sm = np.random.randint(2, size=nb_cls, dtype='bool').tolist()
Sm[np.random.choice(range(nb_cls))] = True
Sn = list(range(nb_cls))
np.random.shuffle(Sn)
Sn = Sn[: nb_pru]
# CANNOT USE, MIGHT REPEAT#
# Sn = np.random.randint(nb_cls, size=nb_pru).tolist()

nb_feat = 4
X_trn, _ = synthetic_dat(nb_labl, nb_inst, nb_feat)
X_val, _ = synthetic_dat(nb_labl, nb_inst, nb_feat)
X_tst, _ = synthetic_dat(nb_labl, nb_inst, nb_feat)
indices = [np.random.randint(
    nb_inst, size=nb_inst).tolist() for _ in range(nb_cls)]
OB_i = [list(set(range(nb_inst)) - set(i)) for i in indices]

mu_ycast = np.random.randint(nb_labl, size=(nb_cls, nb_inst)).tolist()
bi_ycast = np.random.randint(2, size=(nb_cls, nb_inst)).tolist()
tr_ycast = (np.array(bi_ycast) * 2 - 1).tolist()


# ----------------------------------
# xia2018maximum


def test_MRMR():
    from pyfair.granite.ensem_prulatest import (
        _relevancy_score, _complementary_score,
        _MRMR_MI_binary, _MRMR_MI_multiclass,
        _MRMR_MI, _MRMR_subroute, procedure_MRMR)
    tr_rel, tr_c, tr_ci = _relevancy_score(tr_y, tr_yt)
    bi_rel, bi_c, bi_ci = _relevancy_score(tr_y, tr_yt)
    assert id(tr_rel) != id(bi_rel)
    assert id(tr_ci) != id(bi_ci)

    bi_rel, bi_c, bi_ci = _relevancy_score(bi_y, bi_yt)
    mu_rel, mu_c, mu_ci = _relevancy_score(mu_y, mu_yt)
    assert np.all(np.equal(tr_rel, bi_rel))
    assert np.all(np.equal(tr_ci, bi_ci))
    assert 0 <= tr_c == bi_c <= 1
    assert len(mu_rel) == nb_cls

    assert 0 <= mu_c <= 1
    assert all(0 <= i <= 1 for i in tr_ci)
    assert all(0 <= i <= 1 for i in bi_ci)
    assert all(0 <= i <= 1 for i in mu_ci)

    tr_com, tr_s, tr_ci = _complementary_score(
        tr_y, tr_yt[:-1], tr_yt[-1])
    bi_com, bi_s, bi_ci = _complementary_score(
        bi_y, bi_yt[:-1], bi_yt[-1])
    mu_com, mu_s, mu_ci = _complementary_score(
        mu_y, mu_yt[:-1], mu_yt[-1])
    assert tr_com == bi_com
    assert 0 <= tr_s == bi_s <= 1
    assert 0 <= tr_ci == bi_ci <= 1
    assert 0 <= mu_s <= 1 and 0 <= mu_ci <= 1
    assert isinstance(mu_com, float)

    tr_ans = _MRMR_MI_binary(tr_ha, tr_hb)
    bi_ans = _MRMR_MI_binary(bi_ha, bi_hb)
    mu_ans = _MRMR_MI_binary(mu_ha, mu_hb)
    tr_res = _MRMR_MI_multiclass(tr_ha, tr_hb, tr_y)
    bi_res = _MRMR_MI_multiclass(bi_ha, bi_hb, bi_y)
    mu_res = _MRMR_MI_multiclass(mu_ha, mu_hb, mu_y)
    assert 0 <= tr_ans == bi_ans <= 1
    assert 0 <= tr_res == bi_res <= 1
    assert 0 <= mu_ans <= 1 and 0 <= mu_res <= 1

    tr_ans = _MRMR_MI(tr_ha, tr_hb)
    bi_ans = _MRMR_MI(bi_ha, bi_hb)
    mu_ans = _MRMR_MI(mu_ha, mu_hb)
    assert 0 <= tr_ans == bi_ans <= 1
    assert 0 <= mu_ans <= 1

    fj_in_S = np.random.randint(
        nb_cls, size=(nb_pru + 1)).tolist()
    fj_in_S, k = fj_in_S[: -1], fj_in_S[-1]
    tr_ans = _MRMR_subroute(tr_y, tr_yt, fj_in_S, k)
    bi_ans = _MRMR_subroute(bi_y, bi_yt, fj_in_S, k)
    mu_ans = _MRMR_subroute(mu_y, mu_yt, fj_in_S, k)
    assert tr_ans == bi_ans
    assert all(isinstance(
        i, float) for i in [tr_ans, bi_ans, mu_ans])

    tr_P, tr_seq = procedure_MRMR(tr_y, tr_yt, nb_cls, nb_pru)
    bi_P, bi_seq = procedure_MRMR(tr_y, tr_yt, nb_cls, nb_pru)
    assert id(tr_P) != id(bi_P)
    assert id(tr_seq) != id(bi_seq)
    bi_P, bi_seq = procedure_MRMR(bi_y, bi_yt, nb_cls, nb_pru)
    mu_P, mu_seq = procedure_MRMR(mu_y, mu_yt, nb_cls, nb_pru)
    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_seq, bi_seq))
    assert sum(tr_P) == len(tr_seq) == sum(
        bi_P) == len(bi_seq) == nb_pru
    assert sum(mu_P) == len(mu_seq) == nb_pru


def test_MRMC():
    from pyfair.granite.ensem_prulatest import (
        _relevancy_score,
        _complementary_score,
        _relevance_complementary_score)  # ,
    #    procedure_MRMC_ordered_EP,
    #    procedure_MRMC_EP_with_original_ensemble)
    from pyfair.granite.ensem_prulatest import \
        procedure_MRMC_ordered_EP as MRMC_order
    from pyfair.granite.ensem_prulatest import \
        procedure_MRMC_EP_with_original_ensemble as MRMC_prun

    tr_Rel_Ci, _, _ = _relevancy_score(tr_y, tr_yt)
    bi_Rel_Ci, _, _ = _relevancy_score(bi_y, bi_yt)
    mu_Rel_Ci, _, _ = _relevancy_score(mu_y, mu_yt)
    tr_Com_Ci = [_complementary_score(tr_y, tr_yt, i)[0] for i in tr_yt]
    bi_Com_Ci = [_complementary_score(bi_y, bi_yt, i)[0] for i in bi_yt]
    mu_Com_Ci = [_complementary_score(mu_y, mu_yt, i)[0] for i in mu_yt]

    tr_ans = [_relevance_complementary_score(
        i, j) for i, j in zip(tr_Rel_Ci, tr_Com_Ci)]
    bi_ans = [_relevance_complementary_score(
        i, j) for i, j in zip(bi_Rel_Ci, bi_Com_Ci)]
    mu_ans = [_relevance_complementary_score(
        i, j) for i, j in zip(mu_Rel_Ci, mu_Com_Ci)]
    assert np.all(np.equal(tr_ans, bi_ans))
    assert all(isinstance(i, float) for i in mu_ans)

    tr_P, tr_seq = MRMC_order(tr_y, tr_yt, nb_cls, nb_pru)
    bi_P, bi_seq = MRMC_order(tr_y, tr_yt, nb_cls, nb_pru)
    assert id(tr_P) != id(bi_P) and id(tr_seq) != id(bi_seq)
    bi_P, bi_seq = MRMC_order(bi_y, bi_yt, nb_cls, nb_pru)
    mu_P, mu_seq = MRMC_order(mu_y, mu_yt, nb_cls, nb_pru)
    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_seq, bi_seq))
    assert sum(tr_P) == len(tr_seq) == sum(bi_P) == len(bi_seq) == nb_pru
    assert sum(mu_P) == len(mu_seq) == nb_pru

    S = np.random.randint(2, size=nb_cls, dtype='bool').tolist()
    tr_P, tr_seq = MRMC_order(tr_y, tr_yt, nb_cls, nb_pru, S)
    bi_P, bi_seq = MRMC_order(tr_y, tr_yt, nb_cls, nb_pru, S)
    assert id(tr_P) != id(bi_P) and id(tr_seq) != id(bi_seq)
    bi_P, bi_seq = MRMC_order(bi_y, bi_yt, nb_cls, nb_pru, S)
    mu_P, mu_seq = MRMC_order(mu_y, mu_yt, nb_cls, nb_pru, S)
    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_seq, bi_seq))
    assert sum(tr_P) == len(tr_seq) == sum(bi_P) == len(bi_seq) == nb_pru
    assert sum(mu_P) == len(mu_seq) == nb_pru

    tr_P, tr_seq = MRMC_prun(tr_y, tr_yt, nb_cls, nb_pru)
    bi_P, bi_seq = MRMC_prun(tr_y, tr_yt, nb_cls, nb_pru)
    assert id(tr_P) != id(bi_P) and id(tr_seq) != id(bi_seq)
    bi_P, bi_seq = MRMC_prun(bi_y, bi_yt, nb_cls, nb_pru)
    mu_P, mu_seq = MRMC_prun(mu_y, mu_yt, nb_cls, nb_pru)
    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_seq, bi_seq))
    assert sum(tr_P) == len(tr_seq) == sum(bi_P) == len(bi_seq) == nb_pru
    assert sum(mu_P) == len(mu_seq) == nb_pru


# ----------------------------------
# li2018mrmr


# 3.4.1 Measure the capability of base classifiers

def test_MRMREP_capability():
    from pyfair.granite.ensem_prulatest import (
        _judge_double_check_pruned_index,
        _MRMREP_mutual_information_of_whole_subset,
        _MRMREP_F_statistic,
        _normalization_min_max,
        _normalization_z_score,
        _MRMREP_capability_of_classifiers)
    Pm = _judge_double_check_pruned_index(Sm)
    assert sum(Sm) == len(Pm)
    assert all(Sm[i] for i in Pm)
    Pn = _judge_double_check_pruned_index(Sn)
    assert len(Sn) == len(Pn) == nb_pru
    assert all(i == j for i, j in zip(sorted(Sn), Pn))
    assert all(i in Sn for i in Pn)
    assert all(i in Pn for i in Sn)

    tr_ans = _MRMREP_mutual_information_of_whole_subset(tr_y, tr_yt, Sm)
    bi_ans = _MRMREP_mutual_information_of_whole_subset(bi_y, bi_yt, Sm)
    mu_ans = _MRMREP_mutual_information_of_whole_subset(mu_y, mu_yt, Sm)
    assert 0 <= tr_ans == bi_ans <= 1
    assert 0 <= mu_ans <= 1 and isinstance(mu_ans, float)

    tr_ans = _MRMREP_mutual_information_of_whole_subset(tr_y, tr_yt, Sn)
    bi_ans = _MRMREP_mutual_information_of_whole_subset(bi_y, bi_yt, Sn)
    mu_ans = _MRMREP_mutual_information_of_whole_subset(mu_y, mu_yt, Sn)
    assert 0 <= tr_ans == bi_ans <= 1
    assert 0 <= mu_ans <= 1 and isinstance(mu_ans, float)

    tr_ans = _MRMREP_F_statistic(tr_y, tr_ha)
    bi_ans = _MRMREP_F_statistic(bi_y, bi_ha)
    mu_ans = _MRMREP_F_statistic(mu_y, mu_ha)
    assert all(isinstance(i, float) for i in [tr_ans, bi_ans, mu_ans])

    from sklearn import preprocessing as prep
    ep_a, ep_b = 2, 7  # endpoint
    trbimu_hc = np.random.rand(nb_inst) * (ep_b - ep_a) + ep_a
    trbimu_hc = trbimu_hc.tolist()
    CONST_DIFF = 1e-8  # 1e-12

    ans_prop = _normalization_min_max(trbimu_hc)
    ans_comp = prep.minmax_scale(trbimu_hc)
    assert np.all((np.array(ans_prop) - ans_comp) < CONST_DIFF)
    ans_prop = _normalization_z_score(trbimu_hc)
    ans_comp = prep.scale(trbimu_hc)
    assert np.all((np.array(ans_prop) - ans_comp) < CONST_DIFF)

    alpha = 0.4  # I(.) of `tr,bi` are the same, but not F(.)
    tr_ans = _MRMREP_capability_of_classifiers(tr_y, tr_yt, Sm, alpha)
    bi_ans = _MRMREP_capability_of_classifiers(bi_y, bi_yt, Sm, alpha)
    mu_ans = _MRMREP_capability_of_classifiers(mu_y, mu_yt, Sm, alpha)
    assert all(isinstance(i, float) for i in [tr_ans, bi_ans, mu_ans])
    assert all((0 <= i <= 1) for i in [tr_ans, bi_ans, mu_ans])
    tr_ans = _MRMREP_capability_of_classifiers(tr_y, tr_yt, Sn, alpha)
    bi_ans = _MRMREP_capability_of_classifiers(bi_y, bi_yt, Sn, alpha)
    mu_ans = _MRMREP_capability_of_classifiers(mu_y, mu_yt, Sn, alpha)
    assert all(isinstance(i, float) for i in [tr_ans, bi_ans, mu_ans])
    assert all((0 <= i <= 1) for i in [tr_ans, bi_ans, mu_ans])


# 3.4.2 Pruning redundant classifiers

def test_MRMREP_redundant():
    from pyfair.granite.ensem_prulatest import (
        _MRMREP_Hamming_distance,
        _MRMREP_distance_of_each_pair,
        _MRMREP_target_function_of_second_stage,
        _MRMREP_final_target_function)
    tr_ans = _MRMREP_Hamming_distance(tr_ha, tr_hb)
    bi_ans = _MRMREP_Hamming_distance(bi_ha, bi_hb)
    mu_ans = _MRMREP_Hamming_distance(mu_ha, mu_hb)
    assert 0 <= tr_ans == bi_ans <= nb_inst
    assert 0 <= mu_ans <= nb_inst and isinstance(mu_ans, int)

    tr_ans = _MRMREP_distance_of_each_pair(tr_yt)
    bi_ans = _MRMREP_distance_of_each_pair(tr_yt)
    assert id(tr_ans) != id(bi_ans)
    bi_ans = _MRMREP_distance_of_each_pair(bi_yt)
    mu_ans = _MRMREP_distance_of_each_pair(mu_yt)
    assert np.all(np.equal(tr_ans, bi_ans))
    tmp = nb_inst * nb_cls * (nb_cls - 1)
    assert 0 <= np.sum(tr_ans) == np.sum(bi_ans) <= tmp
    del tmp
    assert 0 <= np.sum(mu_ans) <= nb_inst * nb_cls * (nb_cls - 1)

    alpha = 0.6
    from pyfair.facil.utils_const import check_equal
    for S in [Sm, Sn]:
        number_S = len(S)
        maximum = nb_inst * (number_S - 1.) / number_S

        tr_ans = _MRMREP_target_function_of_second_stage(tr_yt, S)
        bi_ans = _MRMREP_target_function_of_second_stage(bi_yt, S)
        mu_ans = _MRMREP_target_function_of_second_stage(mu_yt, S)
        assert 0 <= tr_ans == bi_ans <= maximum
        assert 0 <= mu_ans <= maximum

        tr_ans = _MRMREP_final_target_function(tr_y, tr_yt, S, alpha)
        bi_ans = _MRMREP_final_target_function(bi_y, bi_yt, S, alpha)
        mu_ans = _MRMREP_final_target_function(mu_y, mu_yt, S, alpha)
        # assert -nb_inst < tr_ans == bi_ans <= 1
        assert -nb_inst < mu_ans <= 1
        assert all((-nb_inst < i <= 1) for i in [tr_ans, bi_ans])
        assert check_equal(tr_ans, bi_ans)


# 3.5 Classifier fusion

def test_MRMREP_fusion():
    from pyfair.granite.ensem_prulatest import (
        _subroute_MRMREP_cover, _subroute_MRMREP_untie,
        _subroute_MRMREP_init, _MRMREP_selected_subset,
        MRMREP_Pruning)
    alpha = 0.7

    tr_ci = _subroute_MRMREP_cover(tr_y, tr_yt, alpha, Sm)
    bi_ci = _subroute_MRMREP_cover(bi_y, bi_yt, alpha, Sm)
    mu_ci = _subroute_MRMREP_cover(mu_y, mu_yt, alpha, Sm)
    # idx = nb_cls - sum(Sm)
    assert -1 <= tr_ci == bi_ci < nb_cls  # idx
    assert -1 <= mu_ci < nb_cls and isinstance(mu_ci, int)  # np.int64
    assert (not Sm[tr_ci]) and (not Sm[bi_ci]) and (not Sm[mu_ci])

    tr_ci = _subroute_MRMREP_untie(tr_y, tr_yt, alpha, Sm)
    bi_ci = _subroute_MRMREP_untie(bi_y, bi_yt, alpha, Sm)
    mu_ci = _subroute_MRMREP_untie(mu_y, mu_yt, alpha, Sm)
    # idx = sum(Sm)
    assert -1 <= tr_ci == bi_ci < nb_cls  # idx
    assert -1 <= mu_ci < nb_cls and isinstance(mu_ci, int)  # np.int64
    # pdb.set_trace()  # tt = np.zeros_like(Sm, dtype='bool')
    assert Sm[tr_ci] and Sm[bi_ci] and Sm[mu_ci]

    tr_ci = _subroute_MRMREP_init(tr_y, tr_yt, alpha)
    bi_ci = _subroute_MRMREP_init(bi_y, bi_yt, alpha)
    mu_ci = _subroute_MRMREP_init(mu_y, mu_yt, alpha)
    assert 0 <= tr_ci == bi_ci < nb_cls  # TODO: BUG?
    assert 0 <= mu_ci < nb_cls

    L, R = 3, 2
    tr_rank = _MRMREP_selected_subset(
        tr_y, tr_yt, nb_cls, L, R, alpha)
    bi_rank = _MRMREP_selected_subset(
        tr_y, tr_yt, nb_cls, L, R, alpha)
    assert id(tr_rank) != id(bi_rank)
    bi_rank = _MRMREP_selected_subset(
        bi_y, bi_yt, nb_cls, L, R, alpha)
    mu_rank = _MRMREP_selected_subset(
        mu_y, mu_yt, nb_cls, L, R, alpha)
    assert np.all(np.equal(tr_rank, bi_rank))
    assert all(isinstance(i, int) for i in tr_rank)
    assert all(isinstance(i, int) for i in bi_rank)
    assert all(isinstance(i, int) for i in bi_rank)
    assert len(set(tr_rank)) == nb_cls
    assert len(set(bi_rank)) == nb_cls
    assert len(set(mu_rank)) == nb_cls

    tr_P, tr_seq, tr_rank = MRMREP_Pruning(
        tr_y, tr_yt, nb_cls, nb_pru, L, R, alpha)
    bi_P, bi_seq, bi_rank = MRMREP_Pruning(
        tr_y, tr_yt, nb_cls, nb_pru, L, R, alpha)
    assert id(tr_P) != id(bi_P)
    assert id(tr_seq) != id(bi_seq)
    assert id(tr_rank) != id(bi_rank)

    bi_P, bi_seq, bi_rank = MRMREP_Pruning(
        bi_y, bi_yt, nb_cls, nb_pru, L, R, alpha)
    mu_P, mu_seq, mu_rank = MRMREP_Pruning(
        mu_y, mu_yt, nb_cls, nb_pru, L, R, alpha)
    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_seq, bi_seq))
    assert np.all(np.equal(tr_rank, bi_rank))
    assert len(set(tr_rank)) == nb_cls
    assert len(set(bi_rank)) == nb_cls
    assert len(set(mu_rank)) == nb_cls
    assert sum(tr_P) == len(tr_seq) == sum(
        bi_P) == len(bi_seq) == nb_pru
    assert sum(mu_P) == len(mu_seq) == nb_pru


# ----------------------------------
# ali2019classification
# cao2018optimizing


def test_mRMR_Disc():
    from pyfair.granite.ensem_prulatest import (
        mRMR_ensemble_pruning, Disc_ensemble_pruning)
    tr_P, tr_seq = mRMR_ensemble_pruning(tr_y, tr_yt, nb_cls, nb_pru)
    bi_P, bi_seq = mRMR_ensemble_pruning(tr_y, tr_yt, nb_cls, nb_pru)
    assert id(tr_P) != id(bi_P)
    assert id(tr_seq) != id(bi_seq)

    bi_P, bi_seq = mRMR_ensemble_pruning(bi_y, bi_yt, nb_cls, nb_pru)
    mu_P, mu_seq = mRMR_ensemble_pruning(mu_y, mu_yt, nb_cls, nb_pru)
    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_seq, bi_seq))

    assert sum(tr_P) == len(tr_seq) == sum(
        bi_P) == len(bi_seq) == nb_pru
    assert sum(mu_P) == len(mu_seq) == nb_pru

    tr_P, tr_seq = Disc_ensemble_pruning(tr_y, tr_yt, nb_cls, nb_pru)
    bi_P, bi_seq = Disc_ensemble_pruning(tr_y, tr_yt, nb_cls, nb_pru)
    assert id(tr_P) != id(bi_P)
    assert id(tr_seq) != id(bi_seq)

    bi_P, bi_seq = Disc_ensemble_pruning(bi_y, bi_yt, nb_cls, nb_pru)
    mu_P, mu_seq = Disc_ensemble_pruning(mu_y, mu_yt, nb_cls, nb_pru)
    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_seq, bi_seq))

    assert sum(tr_P) == len(tr_seq) == sum(
        bi_P) == len(bi_seq) == nb_pru
    assert sum(mu_P) == len(mu_seq) == nb_pru


# ----------------------------------
# zhang2019two


def test_twostage_accuracy():
    from pyfair.granite.ensem_prulatest import (
        _subroute_TwoStage_AccuracyBased,
        TwoStagePruning_AccuracyBasedPruning)
    tr_ans = _subroute_TwoStage_AccuracyBased(tr_y, tr_yt, OB_i)
    bi_ans = _subroute_TwoStage_AccuracyBased(tr_y, tr_yt, OB_i)
    assert id(tr_ans) != id(bi_ans)

    bi_ans = _subroute_TwoStage_AccuracyBased(bi_y, bi_yt, OB_i)
    mu_ans = _subroute_TwoStage_AccuracyBased(mu_y, mu_yt, OB_i)
    assert np.all(np.equal(tr_ans, bi_ans))
    assert all(0 <= i <= 1 for i in tr_ans)
    assert all(0 <= i <= 1 for i in bi_ans)
    assert all(0 <= i <= 1 for i in mu_ans)

    ta = 7
    tr_P, tr_seq = TwoStagePruning_AccuracyBasedPruning(
        tr_y, tr_yt, nb_cls, indices, ta)
    bi_P, bi_seq = TwoStagePruning_AccuracyBasedPruning(
        tr_y, tr_yt, nb_cls, indices, ta)
    assert id(tr_P) != id(bi_P) and id(tr_seq) != id(bi_seq)

    bi_P, bi_seq = TwoStagePruning_AccuracyBasedPruning(
        bi_y, bi_yt, nb_cls, indices, ta)
    mu_P, mu_seq = TwoStagePruning_AccuracyBasedPruning(
        mu_y, mu_yt, nb_cls, indices, ta)
    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_seq, bi_seq))
    assert 1 < sum(tr_P) == len(tr_seq) == sum(
        bi_P) == len(bi_seq) < nb_cls
    assert 1 < sum(mu_P) == len(mu_seq) < nb_cls

    for ta in np.arange(0, 10, 1):
        mu_P, mu_seq = TwoStagePruning_AccuracyBasedPruning(
            mu_y, mu_yt, nb_cls, indices, ta)
        if ta == 0:
            assert len(mu_seq) == nb_cls
        assert 1 <= sum(mu_P) == len(mu_seq) <= nb_cls


def test_twostage_distance():
    from pyfair.granite.ensem_prulatest import \
        _subroute_TwoStage_DistanceBased as subr_dist
    from pyfair.granite.ensem_prulatest import \
        _subroute_TwoStage_DistanceBased_inst as subr_inst
    from pyfair.granite.ensem_prulatest import \
        TwoStagePruning_DistanceBasedPruning as TSP_DBP

    tr_ans, tr_res = subr_dist(tr_y, tr_yt, OB_i, tr_ycast)
    bi_ans, bi_res = subr_dist(tr_y, tr_yt, OB_i, tr_ycast)
    assert id(tr_ans) != id(bi_ans)
    assert id(tr_res) != id(bi_res)

    bi_ans, bi_res = subr_dist(bi_y, bi_yt, OB_i, bi_ycast)
    mu_ans, mu_res = subr_dist(mu_y, mu_yt, OB_i, mu_ycast)
    # NOT TRUE# assert np.all(np.equal(tr_ans, bi_ans))
    # NOT TRUE# assert np.all(np.equal(tr_res, bi_res))
    # because need to use values of class/label, and {-1,+1} \neq {0,1}

    assert np.all(np.equal(
        np.argsort(tr_res), np.argsort(bi_res)))
    for tr_tmp, bi_tmp in zip(tr_ans, bi_ans):  # the same as line 411
        assert np.all(np.equal(
            np.argsort(tr_tmp), np.argsort(bi_tmp)))
    assert np.all(np.equal(
        np.argsort(tr_ans), np.argsort(bi_ans)))  # same
    # line 413 is not making sense as much as line 411 (each element in OB_i)
    assert np.all(np.equal(
        np.argsort(tr_ans, axis=0), np.argsort(bi_ans, axis=0)))

    C_i, d_i = subr_inst(X_trn, X_val, OB_i)
    C_p, d_p = subr_inst(X_trn, X_val, OB_i)
    assert id(C_i) != id(C_p)
    assert id(d_i) != id(d_p)

    td = 3
    tr_P, tr_seq = TSP_DBP(tr_y, tr_yt, nb_cls, indices, td, tr_ycast)
    bi_P, bi_seq = TSP_DBP(tr_y, tr_yt, nb_cls, indices, td, tr_ycast)
    assert id(tr_P) != id(bi_P) and id(tr_seq) != id(bi_seq)

    bi_P, bi_seq = TSP_DBP(bi_y, bi_yt, nb_cls, indices, td, bi_ycast)
    mu_P, mu_seq = TSP_DBP(mu_y, mu_yt, nb_cls, indices, td, mu_ycast)
    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_seq, bi_seq))
    assert 1 < sum(tr_P) == len(tr_seq) == sum(bi_P) < nb_cls
    assert 1 < sum(tr_P) == sum(bi_P) == len(bi_seq) < nb_cls
    assert 1 < sum(mu_P) == len(mu_seq) < nb_cls

    for td in np.arange(1, 11, 1):
        mu_P, mu_seq = TSP_DBP(mu_y, mu_yt, nb_cls, indices, td, mu_ycast)
        if td == 10:
            assert len(mu_seq) == nb_cls
        assert 1 <= sum(mu_P) == len(mu_seq) <= nb_cls


def test_twostage_prelim():
    from pyfair.granite.ensem_prulatest import (
        _subroute_TwoStage_OBi,
        _subroute_TwoStage_checkAC,
        _subroute_TwoStage_checkDIS)
    tr_ans = _subroute_TwoStage_OBi(tr_y, indices)
    bi_ans = _subroute_TwoStage_OBi(tr_y, indices)
    assert id(tr_ans) != id(bi_ans)

    bi_ans = _subroute_TwoStage_OBi(bi_y, indices)
    mu_ans = _subroute_TwoStage_OBi(mu_y, indices)
    for i, j, k in zip(tr_ans, bi_ans, mu_ans):
        assert np.all(np.equal(i, j))
        assert np.all(np.equal(j, k))
    # NOTICE: VisibleDeprecationWarning
    # assert np.all(np.equal(tr_ans, bi_ans))
    # assert np.all(np.equal(bi_ans, mu_ans))

    for ta in range(-1, 9 + 2):
        ta = _subroute_TwoStage_checkAC(ta)
        assert 0 <= ta <= 9
    for td in range(0, 10 + 2):
        td = _subroute_TwoStage_checkDIS(td)
        assert 1 <= td <= 10


def test_two_stage_plus():
    from pyfair.granite.ensem_prulatest import (
        TwoStagePruning_APplusDP,
        TwoStagePruning_DPplusAP)
    ta, td = 3, 7  # or 4, 6 # ta, td = 7, 3

    tr_P, tr_M, tr_N = TwoStagePruning_APplusDP(
        tr_y, tr_yt, nb_cls, indices, ta, td, tr_ycast)
    bi_P, bi_M, bi_N = TwoStagePruning_APplusDP(
        tr_y, tr_yt, nb_cls, indices, ta, td, tr_ycast)
    assert id(tr_P) != id(bi_P)
    assert id(tr_M) != id(bi_M)
    assert id(tr_N) != id(bi_N)

    bi_P, bi_M, bi_N = TwoStagePruning_APplusDP(
        bi_y, bi_yt, nb_cls, indices, ta, td, bi_ycast)
    mu_P, mu_M, mu_N = TwoStagePruning_APplusDP(
        mu_y, mu_yt, nb_cls, indices, ta, td, mu_ycast)
    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_M, bi_M))
    assert np.all(np.equal(tr_N, bi_N))

    assert 1 <= sum(tr_P) == len(tr_N) < len(tr_M) < nb_cls
    assert 1 <= sum(bi_P) == len(bi_N) < len(bi_M) < nb_cls
    assert 1 <= sum(mu_P) == len(mu_N) < len(mu_M) < nb_cls

    tr_P, tr_M, tr_N = TwoStagePruning_DPplusAP(
        tr_y, tr_yt, nb_cls, indices, ta, td, tr_ycast)
    bi_P, bi_M, bi_N = TwoStagePruning_DPplusAP(
        tr_y, tr_yt, nb_cls, indices, ta, td, tr_ycast)
    assert id(tr_P) != id(bi_P)
    assert id(tr_M) != id(bi_M)
    assert id(tr_N) != id(bi_N)

    bi_P, bi_M, bi_N = TwoStagePruning_DPplusAP(
        bi_y, bi_yt, nb_cls, indices, ta, td, bi_ycast)
    mu_P, mu_M, mu_N = TwoStagePruning_DPplusAP(
        mu_y, mu_yt, nb_cls, indices, ta, td, mu_ycast)
    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_M, bi_M))
    assert np.all(np.equal(tr_N, bi_N))

    assert 1 <= sum(tr_P) == len(tr_N) < len(tr_M) < nb_cls
    assert 1 <= sum(bi_P) == len(bi_N) < len(bi_M) < nb_cls
    assert 1 <= sum(mu_P) == len(mu_N) < len(mu_M) < nb_cls


# ----------------------------------
# zhang2019two, exactly in paper


def test_twostage_previous():
    from pyfair.granite.ensem_prulatest import (
        TwoStagePrev_DistanceBasedPruning,
        TwoStagePreviously_AP_plus_DP,
        TwoStagePreviously_DP_plus_AP)
    ta, td = 4, 6

    tr_P, tr_seq = TwoStagePrev_DistanceBasedPruning(
        tr_y, tr_yt, nb_cls, indices, td, X_trn, X_val)
    bi_P, bi_seq = TwoStagePrev_DistanceBasedPruning(
        tr_y, tr_yt, nb_cls, indices, td, X_trn, X_val)
    assert id(tr_P) != id(bi_P) and id(tr_seq) != id(bi_seq)
    bi_P, bi_seq = TwoStagePrev_DistanceBasedPruning(
        bi_y, bi_yt, nb_cls, indices, td, X_trn, X_val)
    mu_P, mu_seq = TwoStagePrev_DistanceBasedPruning(
        mu_y, mu_yt, nb_cls, indices, td, X_trn, X_val)
    assert np.all(np.equal(tr_seq, bi_seq))
    assert np.all(np.equal(bi_seq, mu_seq))
    # because it is related to X_feat only, independent of `yt`
    assert 1 < len(mu_seq) == sum(mu_P) < nb_cls

    tr_P, tr_M, tr_N = TwoStagePreviously_AP_plus_DP(
        tr_y, tr_yt, nb_cls, indices, ta, td, X_trn, X_val)
    bi_P, bi_M, bi_N = TwoStagePreviously_AP_plus_DP(
        tr_y, tr_yt, nb_cls, indices, ta, td, X_trn, X_val)
    assert id(tr_P) != id(bi_P)
    assert id(tr_M) != id(bi_M) and id(tr_N) != id(bi_N)
    bi_P, bi_M, bi_N = TwoStagePreviously_AP_plus_DP(
        bi_y, bi_yt, nb_cls, indices, ta, td, X_trn, X_val)
    mu_P, mu_M, mu_N = TwoStagePreviously_AP_plus_DP(
        mu_y, mu_yt, nb_cls, indices, ta, td, X_trn, X_val)
    assert np.all(np.equal(tr_M, bi_M))
    assert np.all(np.equal(tr_N, bi_N))

    tr_P, tr_M, tr_N = TwoStagePreviously_DP_plus_AP(
        tr_y, tr_yt, nb_cls, indices, ta, td, X_trn, X_val)
    bi_P, bi_M, bi_N = TwoStagePreviously_DP_plus_AP(
        tr_y, tr_yt, nb_cls, indices, ta, td, X_trn, X_val)
    assert id(tr_P) != id(bi_P)
    assert id(tr_M) != id(bi_M) and id(tr_N) != id(bi_N)
    bi_P, bi_M, bi_N = TwoStagePreviously_DP_plus_AP(
        bi_y, bi_yt, nb_cls, indices, ta, td, X_trn, X_val)
    mu_P, mu_M, mu_N = TwoStagePreviously_DP_plus_AP(
        mu_y, mu_yt, nb_cls, indices, ta, td, X_trn, X_val)
    assert np.all(np.equal(tr_M, bi_M))
    assert np.all(np.equal(tr_N, bi_N))


# ----------------------------------
# contrastive

# Cannot fix the size of pruned sub-ensemble:
#
# Basically, all `TwoStage` related:
#     "TSP-AP", "TSP-DP", "TSP-AP+DP", "TSP-DP+AP",
#     "TSPrev-DP", "TSPrev-AD", "TSPrev-DA",
#


def test_contrastive():
    from pyfair.granite.ensem_prulatest import contrastive_pruning_lately
    from pyfair.facil.utils_remark import LATEST_NAME_PRUNE
    # from copy import deepcopy
    L, R, alpha = 3, 2, 0.5
    # argv = {'indices': indices}
    # argc = {"indices": indices, "X_trn": X_trn, "X_val": X_val}

    for name_pru in LATEST_NAME_PRUNE:
        kwargs = {}
        if name_pru.startswith("TSP"):
            # kwargs = deepcopy(argv)
            kwargs["indices"] = indices
        if name_pru.startswith("TSPrev"):
            kwargs["X_trn"] = X_trn
            kwargs["X_val"] = X_val

        tr_ytrn, tr_yval, tr_P, tr_seq = contrastive_pruning_lately(
            name_pru, nb_cls, nb_pru, tr_y, [], tr_yt, [],
            alpha, L, R, **kwargs)
        bi_ytrn, bi_yval, bi_P, bi_seq = contrastive_pruning_lately(
            name_pru, nb_cls, nb_pru, tr_y, [], tr_yt, [],
            alpha, L, R, **kwargs)
        assert id(tr_ytrn) != id(bi_ytrn)
        assert id(tr_yval) != id(bi_yval)
        assert id(tr_P) != id(bi_P)
        assert id(tr_seq) != id(bi_seq)
        assert tr_yval == bi_yval == []

        bi_ytrn, bi_yval, bi_P, bi_seq = contrastive_pruning_lately(
            name_pru, nb_cls, nb_pru, bi_y, [], bi_yt, [],
            alpha, L, R, **kwargs)
        _, _, mu_P, mu_seq = contrastive_pruning_lately(
            name_pru, nb_cls, nb_pru, mu_y, [], mu_yt, [],
            alpha, L, R, **kwargs)  # mu_ytrn,mu_yval,
        assert sum(mu_P) == len(mu_seq)
        assert np.all(np.equal(tr_P, bi_P))  # TODO: BUG?
        assert np.all(np.equal(tr_seq, bi_seq))
        assert sum(tr_P) == len(tr_seq) == sum(bi_P) == len(bi_seq)
        assert 1 <= len(tr_seq) == len(bi_seq) < nb_cls
        assert 1 <= len(mu_seq) < nb_cls


def test_compared_utus():
    from pyfair.granite.ensem_prulatest import \
        contrastive_pruning_lately_validate
    from pyfair.facil.utils_remark import LATEST_NAME_PRUNE
    from sklearn import tree
    L, R, alpha = 4, 3, 0.5
    clfs = [tree.DecisionTreeClassifier() for _ in range(nb_cls)]
    coef = np.random.rand(nb_cls)
    coef /= np.sum(coef)
    coef = coef.tolist()

    for name_pru in LATEST_NAME_PRUNE:
        kwargs = {}
        if name_pru.startswith("TSP"):
            kwargs["indices"] = indices
        if name_pru.startswith("TSPrev"):
            kwargs["X_trn"] = X_trn
            kwargs["X_val"] = X_val

        (_, _, _, ys_cast, _, _, _,
         # opt_coef,opt_clfs,ys_insp,,ys_pred,ut,us,
         tr_P, tr_seq) = contrastive_pruning_lately_validate(
            name_pru, nb_cls, nb_pru, tr_y, [], tr_yt, [],
            tr_ycast, coef, clfs, alpha, L, R, **kwargs)
        assert ys_cast == []
        assert sum(tr_P) == len(tr_seq)

        (_, _, _, _, _, _, _, bi_P,  # ut,us,
         bi_seq) = contrastive_pruning_lately_validate(
            name_pru, nb_cls, nb_pru, bi_y, [], bi_yt, [],
            bi_ycast, coef, clfs, alpha, L, R, **kwargs)
        assert np.all(np.equal(tr_seq, bi_seq))  # TODO: BUG?
        assert sum(bi_P) == len(bi_seq)

        (_, _, _, _, _, _, _, mu_P,  # ut,us,
         mu_seq) = contrastive_pruning_lately_validate(
            name_pru, nb_cls, nb_pru, mu_y, [], mu_yt, [],
            mu_ycast, coef, clfs, alpha, L, R, **kwargs)
        assert 1 <= sum(mu_P) == len(mu_seq) < nb_cls


# ----------------------------------
# NOTICE:
# When numpy==1.19.5
#      scipy==1.5.4
#      scikit-learn==0.24.1
#      pathos==0.2.7
#      Pympler==0.9
#      Pillow==8.1.2
#      matplotlib==3.3.4
#      pytest==6.2.2
#
# =============================== warnings summary ===============================
# tests/core/test_prulatest.py::test_contrastive
#   /home/ubuntu/Software/anaconda3/envs/ensem/lib/python3.6/importlib/_bootstrap.py:219:
#   RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility.
#   Expected 192 from C header, got 216 from PyObject
#     return f(*args, **kwds)
#
# -- Docs: https://docs.pytest.org/en/stable/warnings.html
#


# ----------------------------------
# ----------------------------------

# ----------------------------------
# ----------------------------------


# ----------------------------------
# ----------------------------------

# ----------------------------------
# ----------------------------------
