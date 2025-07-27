# coding: utf-8

import numpy as np
from pyfair.facil.utils_const import check_equal, synthetic_set

nb_inst, nb_labl, nb_cls = 100, 3, 7
bi_y, bi_yt, bi_c = synthetic_set(2, nb_inst, nb_cls)
mu_y, mu_yt, mu_c = synthetic_set(nb_labl, nb_inst, nb_cls)
tr_yt = np.array(bi_yt) * 2 - 1
tr_yt = tr_yt.tolist()
tr_y = [i * 2 - 1 for i in bi_y]
ki, kj = 0, 1  # idx


def test_contingency_table():
    from pyfair.marble.diver_pairwise import (
        contingency_table_binary,
        contingency_table_multi,
        contingency_table_multiclass)
    # ki, kj = 0, 1  # idx
    tr_a, tr_b, tr_c, tr_d = \
        contingency_table_binary(tr_yt[ki], tr_yt[kj])
    bi_a, bi_b, bi_c, bi_d = \
        contingency_table_binary(bi_yt[ki], bi_yt[kj])
    assert (tr_a + tr_b + tr_c + tr_d) == nb_inst
    assert (bi_a + bi_b + bi_c + bi_d) == nb_inst
    assert tr_a == bi_a and tr_d == bi_d
    assert tr_b == bi_b and tr_c == bi_c

    tr_Cij = contingency_table_multi(tr_yt[ki], tr_yt[kj], tr_y)
    bi_Cij = contingency_table_multi(bi_yt[ki], bi_yt[kj], bi_y)
    assert np.all(np.equal(tr_Cij, bi_Cij))
    assert bi_Cij[0][0] == bi_d and bi_Cij[1][1] == bi_a
    assert bi_Cij[1][0] == bi_b and bi_Cij[0][1] == bi_c
    assert np.sum(bi_Cij) == np.sum(tr_Cij) == nb_inst

    rta, rtb, rtc, rtd = contingency_table_multiclass(
        tr_yt[ki], tr_yt[kj], tr_y)
    rba, rbb, rbc, rbd = contingency_table_multiclass(
        bi_yt[ki], bi_yt[kj], bi_y)
    assert (rta + rtb + rtc + rtd) == nb_inst
    assert (rba + rbb + rbc + rbd) == nb_inst
    assert rta == rba and rtd == rbd and rtb == rbb and rtc == rbc

    assert (rba + rbd) == (bi_a + bi_d)
    assert (rbb + rbc) == (bi_b + bi_c)

    mu_Cij = contingency_table_multi(mu_yt[ki], mu_yt[kj], mu_y)
    mu_a, mu_b, mu_c, mu_d = \
        contingency_table_multiclass(mu_yt[ki], mu_yt[kj], mu_y)
    assert np.sum(mu_Cij) == (mu_a + mu_b + mu_c + mu_d) == nb_inst


def test_number_individual():
    from pyfair.marble.diver_nonpairwise import (
        number_individuals_correctly,
        number_individuals_fall_through)
    rho_tr = number_individuals_correctly(tr_yt, tr_y)
    rho_bi = number_individuals_correctly(bi_yt, bi_y)
    rho_mu = number_individuals_correctly(mu_yt, mu_y)
    assert np.all(np.equal(rho_tr, rho_bi))
    assert np.sum(rho_tr) == np.sum(rho_bi) <= nb_inst * nb_cls
    assert np.sum(rho_mu) <= nb_inst * nb_cls

    pi_tr = number_individuals_fall_through(tr_yt, tr_y, nb_cls)
    pi_bi = number_individuals_fall_through(bi_yt, bi_y, nb_cls)
    pi_mu = number_individuals_fall_through(mu_yt, mu_y, nb_cls)
    assert np.all(np.equal(pi_tr, pi_bi))
    assert check_equal(np.sum(pi_tr), 1.)
    assert check_equal(np.sum(pi_bi), 1.)
    assert check_equal(np.sum(pi_mu), 1.)

    pi_tr = np.array(pi_tr)
    pi_bi = np.array(pi_bi)
    pi_mu = np.array(pi_mu)
    assert np.all((pi_tr >= 0) & (pi_tr <= 1))
    assert np.all((pi_bi >= 0) & (pi_tr <= 1))
    assert np.all((pi_mu >= 0) & (pi_mu <= 1))


# ------------------------------
# Pairwise


def test_q_statistic():
    from pyfair.marble.diver_pairwise import (
        Q_statistic_multiclass,
        Q_Statistic_binary,
        Q_Statistic_multi)
    # ki, kj = 0, 1

    bi_ans_bi = Q_Statistic_binary(bi_yt[ki], bi_yt[kj])
    tr_ans_bi = Q_Statistic_binary(tr_yt[ki], tr_yt[kj])
    bi_ans_mu = Q_Statistic_multi(bi_yt[ki], bi_yt[kj], bi_y)
    tr_ans_mu = Q_Statistic_multi(tr_yt[ki], tr_yt[kj], tr_y)
    bi_ans_re = Q_statistic_multiclass(bi_yt[ki], bi_yt[kj], bi_y)
    tr_ans_re = Q_statistic_multiclass(tr_yt[ki], tr_yt[kj], tr_y)

    assert bi_ans_bi == bi_ans_mu == tr_ans_bi == tr_ans_mu
    assert bi_ans_re == tr_ans_re
    assert abs(tr_ans_re) == abs(bi_ans_re) <= 1
    assert abs(tr_ans_bi) == abs(bi_ans_bi) <= 1
    # assert tr_ans_re * tr_ans_bi >= 0  # the same sign
    # NOTICE: It's possible that they are not the same sign

    mu_ans_mu = Q_Statistic_multi(mu_yt[ki], mu_yt[kj], mu_y)
    mu_ans_re = Q_statistic_multiclass(mu_yt[ki], mu_yt[kj], mu_y)
    assert -1 <= mu_ans_mu <= 1
    assert -1 <= mu_ans_re <= 1


def test_kappa_statistic():
    from pyfair.marble.diver_pairwise import \
        kappa_statistic_multiclass as kpa_multiclass
    from pyfair.marble.diver_pairwise import \
        Kappa_Statistic_binary as KStat_binary
    from pyfair.marble.diver_pairwise import \
        Kappa_Statistic_multi as KStat_multi

    bi_ans_bi = KStat_binary(bi_yt[ki], bi_yt[kj], nb_inst)
    tr_ans_bi = KStat_binary(tr_yt[ki], tr_yt[kj], nb_inst)
    bi_ans_mu = KStat_multi(bi_yt[ki], bi_yt[kj], bi_y, nb_inst)
    tr_ans_mu = KStat_multi(tr_yt[ki], tr_yt[kj], tr_y, nb_inst)
    bi_ans_re = kpa_multiclass(bi_yt[ki], bi_yt[kj], bi_y, nb_inst)
    tr_ans_re = kpa_multiclass(tr_yt[ki], tr_yt[kj], tr_y, nb_inst)

    assert bi_ans_bi == bi_ans_mu[0] == tr_ans_bi == tr_ans_mu[0]
    assert bi_ans_re == tr_ans_re
    assert abs(tr_ans_re) <= 1
    assert abs(tr_ans_bi) <= 1
    assert 0 <= bi_ans_mu[1] == tr_ans_mu[1] <= 1  # theta1
    assert 0 <= bi_ans_mu[2] == tr_ans_mu[2] <= 1  # theta2
    # assert np.sign(tr_ans_bi) == np.sign(tr_ans_re)
    # NOTICE: It's possible that they are not the same sign

    mu_ans_re = kpa_multiclass(mu_yt[ki], mu_yt[kj], mu_y, nb_inst)
    mu_ans_mu = KStat_multi(mu_yt[ki], mu_yt[kj], mu_y, nb_inst)
    assert -1 <= mu_ans_re <= 1
    assert -1 <= mu_ans_mu[0] <= 1  # kappa_p
    assert 0 <= mu_ans_mu[1] <= 1  # theta1
    assert 0 <= mu_ans_mu[2] <= 1  # theta2
    # It's possible that np.sign(mu_ans_re) != np.sign(mu_ans_mu[0])


def test_disagreement():
    from pyfair.marble.diver_pairwise import \
        disagreement_measure_multiclass as dis_multiclass
    from pyfair.marble.diver_pairwise import \
        Disagreement_Measure_binary as Disag_binary
    from pyfair.marble.diver_pairwise import \
        Disagreement_Measure_multi as Disag_multi

    bi_ans_bi = Disag_binary(bi_yt[ki], bi_yt[kj], nb_inst)
    tr_ans_bi = Disag_binary(tr_yt[ki], tr_yt[kj], nb_inst)
    bi_ans_mu = Disag_multi(bi_yt[ki], bi_yt[kj], nb_inst)
    tr_ans_mu = Disag_multi(tr_yt[ki], tr_yt[kj], nb_inst)
    bi_ans_re = dis_multiclass(bi_yt[ki], bi_yt[kj], bi_y, nb_inst)
    tr_ans_re = dis_multiclass(tr_yt[ki], tr_yt[kj], tr_y, nb_inst)

    assert bi_ans_bi == bi_ans_mu == tr_ans_bi == tr_ans_mu
    assert bi_ans_re == tr_ans_re == tr_ans_bi
    assert 0 <= tr_ans_re <= 1

    mu_ans_re = dis_multiclass(mu_yt[ki], mu_yt[kj], mu_y, nb_inst)
    mu_ans_mu = Disag_multi(mu_yt[ki], mu_yt[kj], nb_inst)
    assert 0 <= mu_ans_re <= 1
    assert 0 <= mu_ans_mu <= 1


def test_correlation_coefficient():
    from pyfair.marble.diver_pairwise import \
        correlation_coefficient_multiclass as cor_multiclass
    from pyfair.marble.diver_pairwise import \
        Correlation_Coefficient_binary as Corre_binary
    from pyfair.marble.diver_pairwise import \
        Correlation_Coefficient_multi as Corre_multi
    from pyfair.marble.diver_pairwise import (
        Q_statistic_multiclass, Q_Statistic_binary,
        Q_Statistic_multi)

    bi_ans_bi = Corre_binary(bi_yt[ki], bi_yt[kj])
    tr_ans_bi = Corre_binary(tr_yt[ki], tr_yt[kj])
    bi_ans_mu = Corre_multi(bi_yt[ki], bi_yt[kj], bi_y)
    tr_ans_mu = Corre_multi(tr_yt[ki], tr_yt[kj], tr_y)
    bi_ans_re = cor_multiclass(bi_yt[ki], bi_yt[kj], bi_y)
    tr_ans_re = cor_multiclass(tr_yt[ki], tr_yt[kj], tr_y)

    assert bi_ans_bi == bi_ans_mu == tr_ans_bi == tr_ans_mu
    assert bi_ans_re == tr_ans_re
    assert abs(tr_ans_bi) == abs(bi_ans_bi) <= 1
    assert abs(tr_ans_re) == abs(bi_ans_re) <= 1
    # It's possible that np.sign(tr_ans_bi) != np.sign(tr_ans_re)

    mu_ans_re = cor_multiclass(mu_yt[ki], mu_yt[kj], mu_y)
    mu_ans_mu = Corre_multi(mu_yt[ki], mu_yt[kj], mu_y)
    assert np.abs(mu_ans_re) <= 1
    assert np.abs(mu_ans_mu) <= 1
    # It's possible that np.sign(mu_ans_re) != np.sign(mu_ans_mu)

    # from pyfair.junior.diver_pairwise import (
    #     Q_statistic_multiclass, Q_Statistic_binary)
    tem_re = Q_statistic_multiclass(mu_yt[ki], mu_yt[kj], mu_y)
    assert abs(tem_re) >= abs(mu_ans_re)
    tem_re = Q_Statistic_binary(bi_yt[ki], bi_yt[ki])
    assert abs(tem_re) >= abs(bi_ans_bi)

    # from pyfair.junior.diver_pairwise import Q_Statistic_multi
    tem_re = Q_Statistic_multi(mu_yt[ki], mu_yt[kj], mu_y)
    assert abs(tem_re) >= abs(mu_ans_mu)


def test_double_fault():
    from pyfair.marble.diver_pairwise import \
        double_fault_measure_multiclass as dbl_multiclass
    from pyfair.marble.diver_pairwise import \
        Double_Fault_Measure_binary_multi as DoubF_multi

    bi_ans_bi = DoubF_multi(bi_yt[ki], bi_yt[kj], bi_y, nb_inst)
    tr_ans_bi = DoubF_multi(tr_yt[ki], tr_yt[kj], tr_y, nb_inst)
    bi_ans_re = dbl_multiclass(bi_yt[ki], bi_yt[kj], bi_y, nb_inst)
    tr_ans_re = dbl_multiclass(tr_yt[ki], tr_yt[kj], tr_y, nb_inst)

    assert bi_ans_bi == tr_ans_bi and bi_ans_re == tr_ans_re
    assert 0 <= bi_ans_bi == bi_ans_re == tr_ans_bi == tr_ans_re <= 1

    mu_ans_re = dbl_multiclass(mu_yt[ki], mu_yt[kj], mu_y, nb_inst)
    mu_ans_mu = DoubF_multi(mu_yt[ki], mu_yt[kj], mu_y, nb_inst)
    assert 0 <= mu_ans_re == mu_ans_mu <= 1


# ------------------------------
# Non-Pairwise


def test_KW_variance():
    from pyfair.marble.diver_nonpairwise import \
        Kohavi_Wolpert_variance_multiclass as KWVar_multiclass
    bi_ans = KWVar_multiclass(bi_yt, bi_y, nb_inst, nb_cls)
    tr_ans = KWVar_multiclass(tr_yt, tr_y, nb_inst, nb_cls)
    mu_ans = KWVar_multiclass(mu_yt, mu_y, nb_inst, nb_cls)
    assert 0 <= bi_ans == tr_ans <= 1. / 4
    assert 0 <= mu_ans <= 1. / 4


def test_interrater_agreement():
    from pyfair.marble.diver_nonpairwise import \
        interrater_agreement_multiclass as Inter_multiclass
    bi_ans = Inter_multiclass(bi_yt, bi_y, nb_inst, nb_cls)
    tr_ans = Inter_multiclass(tr_yt, tr_y, nb_inst, nb_cls)
    mu_ans = Inter_multiclass(mu_yt, mu_y, nb_inst, nb_cls)
    assert bi_ans == tr_ans <= 1
    assert mu_ans <= 1


def test_entropy_cc_sk():
    from pyfair.marble.diver_nonpairwise import (
        Entropy_cc_multiclass, Entropy_sk_multiclass)

    bi_ans = Entropy_cc_multiclass(bi_yt, bi_y)
    tr_ans = Entropy_cc_multiclass(tr_yt, tr_y)
    mu_ans = Entropy_cc_multiclass(mu_yt, mu_y)
    assert 0 <= bi_ans == tr_ans <= 1
    assert 0 < mu_ans <= 1

    bi_ans = Entropy_sk_multiclass(bi_yt, bi_y, nb_cls)
    tr_ans = Entropy_sk_multiclass(tr_yt, tr_y, nb_cls)
    mu_ans = Entropy_sk_multiclass(mu_yt, mu_y, nb_cls)
    assert 0 <= bi_ans == tr_ans <= 1
    assert 0 <= mu_ans <= 1


def test_difficulty():
    from pyfair.marble.diver_nonpairwise import difficulty_multiclass
    bi_ans = difficulty_multiclass(bi_yt, bi_y)
    tr_ans = difficulty_multiclass(tr_yt, tr_y)
    mu_ans = difficulty_multiclass(mu_yt, mu_y)
    assert 0 <= bi_ans == tr_ans <= 1
    assert 0 <= mu_ans <= 1


def test_generalized():
    from pyfair.marble.diver_nonpairwise import \
        generalized_diversity_multiclass as GeneD_multiclass
    from pyfair.marble.diver_nonpairwise import \
        Generalized_Diversity_multi as GeneD_multi

    bi_ans_re = GeneD_multiclass(bi_yt, bi_y, nb_cls)
    tr_ans_re = GeneD_multiclass(tr_yt, tr_y, nb_cls)
    mu_ans_re = GeneD_multiclass(mu_yt, mu_y, nb_cls)
    assert bi_ans_re == tr_ans_re <= 1
    assert mu_ans_re <= 1

    bi_ans_mu = GeneD_multi(bi_yt, bi_y, nb_inst, nb_cls)
    tr_ans_mu = GeneD_multi(tr_yt, tr_y, nb_inst, nb_cls)
    mu_ans_mu = GeneD_multi(mu_yt, mu_y, nb_inst, nb_cls)
    assert bi_ans_mu == tr_ans_mu <= 1
    assert mu_ans_re <= 1

    assert bi_ans_re == bi_ans_mu == tr_ans_re == tr_ans_mu <= 1
    assert mu_ans_re == mu_ans_mu <= 1


def test_coincident_failure():
    from pyfair.marble.diver_nonpairwise import \
        coincident_failure_multiclass
    bi_ans = coincident_failure_multiclass(bi_yt, bi_y, nb_cls)
    tr_ans = coincident_failure_multiclass(tr_yt, tr_y, nb_cls)
    mu_ans = coincident_failure_multiclass(mu_yt, mu_y, nb_cls)
    assert 0 <= bi_ans == tr_ans <= 1
    assert 0 <= mu_ans <= 1


# ------------------------------
# General


def test_pairwise_item():
    from pyfair.facil.utils_remark import PAIRWISE
    from pyfair.marble.diver_pairwise import (
        pairwise_measure_item_multiclass,
        pairwise_measure_item_binary,
        pairwise_measure_item_multi)
    for name_div in sorted(PAIRWISE.keys()):
        tr_ans_re = pairwise_measure_item_multiclass(
            name_div, tr_yt[ki], tr_yt[kj], tr_y, nb_inst)
        bi_ans_re = pairwise_measure_item_multiclass(
            name_div, bi_yt[ki], bi_yt[kj], bi_y, nb_inst)
        mu_ans_re = pairwise_measure_item_multiclass(
            name_div, mu_yt[ki], mu_yt[kj], mu_y, nb_inst)

        tr_ans_bi = pairwise_measure_item_binary(
            PAIRWISE[name_div], tr_yt[ki], tr_yt[kj], tr_y, nb_inst)
        bi_ans_bi = pairwise_measure_item_binary(
            PAIRWISE[name_div], bi_yt[ki], bi_yt[kj], bi_y, nb_inst)
        tr_ans_mu = pairwise_measure_item_multi(
            PAIRWISE[name_div], tr_yt[ki], tr_yt[kj], tr_y, nb_inst)
        bi_ans_mu = pairwise_measure_item_multi(
            PAIRWISE[name_div], bi_yt[ki], bi_yt[kj], bi_y, nb_inst)
        mu_ans_mu = pairwise_measure_item_multi(
            PAIRWISE[name_div], mu_yt[ki], mu_yt[kj], mu_y, nb_inst)

        if name_div in ['Disag', 'DoubF']:
            assert tr_ans_re == tr_ans_bi == tr_ans_mu
            assert bi_ans_re == bi_ans_bi == bi_ans_mu
        else:
            assert tr_ans_bi == tr_ans_mu == bi_ans_bi == bi_ans_mu
        assert tr_ans_re == bi_ans_re
        assert all(isinstance(
            i, float) for i in [tr_ans_re, bi_ans_re, mu_ans_re])
        assert all(isinstance(
            i, float) for i in [tr_ans_bi, bi_ans_bi])
        assert all(isinstance(
            i, float) for i in [tr_ans_mu, bi_ans_mu, mu_ans_mu])


def test_pairwise_gather():
    from pyfair.facil.utils_remark import PAIRWISE
    from pyfair.marble.diver_pairwise import (
        pairwise_measure_gather_multiclass,
        pairwise_measure_whole_binary,
        pairwise_measure_whole_multi)
    for name_div in sorted(PAIRWISE.keys()):
        tr_ans_re = pairwise_measure_gather_multiclass(
            name_div, tr_yt, tr_y, nb_inst, nb_cls)
        bi_ans_re = pairwise_measure_gather_multiclass(
            name_div, bi_yt, bi_y, nb_inst, nb_cls)
        mu_ans_re = pairwise_measure_gather_multiclass(
            name_div, mu_yt, mu_y, nb_inst, nb_cls)

        tr_ans_bi = pairwise_measure_whole_binary(
            PAIRWISE[name_div], tr_yt, tr_y, nb_inst, nb_cls)
        bi_ans_bi = pairwise_measure_whole_binary(
            PAIRWISE[name_div], bi_yt, bi_y, nb_inst, nb_cls)
        tr_ans_mu = pairwise_measure_whole_multi(
            PAIRWISE[name_div], tr_yt, tr_y, nb_inst, nb_cls)
        bi_ans_mu = pairwise_measure_whole_multi(
            PAIRWISE[name_div], bi_yt, bi_y, nb_inst, nb_cls)
        mu_ans_mu = pairwise_measure_whole_multi(
            PAIRWISE[name_div], mu_yt, mu_y, nb_inst, nb_cls)

        assert tr_ans_re == bi_ans_re
        assert tr_ans_bi == tr_ans_mu == bi_ans_bi == bi_ans_mu
        if name_div in ['Disag', 'DoubF']:
            assert tr_ans_re == tr_ans_bi == tr_ans_mu
            assert bi_ans_re == bi_ans_bi == bi_ans_mu
        assert all(isinstance(
            i, float) for i in [tr_ans_re, tr_ans_bi, tr_ans_mu])
        assert all(isinstance(
            i, float) for i in [bi_ans_re, bi_ans_bi, bi_ans_mu])
        assert all(isinstance(
            i, float) for i in [mu_ans_re, mu_ans_mu])


def test_nonpairwise():
    from pyfair.facil.utils_remark import NONPAIRWISE
    from pyfair.marble.diver_nonpairwise import (
        nonpairwise_measure_gather_multiclass,
        nonpairwise_measure_item_multiclass)
    for name_div in sorted(NONPAIRWISE.keys()):
        tr_ans_re = nonpairwise_measure_gather_multiclass(
            name_div, tr_yt, tr_y, nb_inst, nb_cls)
        bi_ans_re = nonpairwise_measure_gather_multiclass(
            name_div, bi_yt, bi_y, nb_inst, nb_cls)
        mu_ans_re = nonpairwise_measure_gather_multiclass(
            name_div, mu_yt, mu_y, nb_inst, nb_cls)

        tr_ans_it = nonpairwise_measure_item_multiclass(
            name_div, tr_yt[ki], tr_yt[kj], tr_y, nb_inst)
        bi_ans_it = nonpairwise_measure_item_multiclass(
            name_div, bi_yt[ki], bi_yt[kj], bi_y, nb_inst)
        mu_ans_it = nonpairwise_measure_item_multiclass(
            name_div, mu_yt[ki], mu_yt[kj], mu_y, nb_inst)

        assert all(isinstance(
            i, float) for i in [tr_ans_re, bi_ans_re, mu_ans_re])
        assert all(isinstance(
            i, float) for i in [tr_ans_it, bi_ans_it, mu_ans_it])


def test_contrastive():
    from pyfair.facil.utils_remark import AVAILABLE_NAME_DIVER
    tr_ha, tr_hb = tr_yt[ki], tr_yt[kj]
    bi_ha, bi_hb = bi_yt[ki], bi_yt[kj]
    mu_ha, mu_hb = mu_yt[ki], mu_yt[kj]

    from pyfair.granite.ensem_diversity import \
        contrastive_diversity_gather_multiclass as gather_mc
    from pyfair.granite.ensem_diversity import \
        contrastive_diversity_item_multiclass as item_mc
    from pyfair.granite.ensem_diversity import \
        contrastive_diversity_by_instance_multiclass as inst_mc

    from pyfair.granite.ensem_diversity import \
        contrastive_diversity_whole_binary as whole_bi
    from pyfair.granite.ensem_diversity import \
        contrastive_diversity_whole_multi as whole_mu

    for name_div in AVAILABLE_NAME_DIVER:
        tr_ans_re = gather_mc(name_div, tr_y, tr_yt)
        bi_ans_re = gather_mc(name_div, bi_y, bi_yt)
        mu_ans_re = gather_mc(name_div, mu_y, mu_yt)
        assert all(isinstance(
            i, float) for i in [tr_ans_re, bi_ans_re, mu_ans_re])

        tr_ans_bi = whole_bi(name_div, tr_y, tr_yt)
        bi_ans_bi = whole_bi(name_div, bi_y, bi_yt)
        tr_ans_mu = whole_mu(name_div, tr_y, tr_yt)
        bi_ans_mu = whole_mu(name_div, bi_y, bi_yt)
        mu_ans_mu = whole_mu(name_div, mu_y, mu_yt)
        assert tr_ans_bi == tr_ans_mu == bi_ans_bi == bi_ans_mu
        assert tr_ans_re == bi_ans_re
        if name_div in ['Disag', 'DoubF']:
            assert tr_ans_re == tr_ans_bi == tr_ans_mu
            assert bi_ans_re == bi_ans_bi == bi_ans_mu
        assert all(isinstance(
            i, float) for i in [tr_ans_bi, bi_ans_bi])
        assert all(isinstance(
            i, float) for i in [tr_ans_mu, bi_ans_mu, mu_ans_mu])

        tr_ans_re = item_mc(name_div, tr_y, tr_ha, tr_hb)
        bi_ans_re = item_mc(name_div, bi_y, bi_ha, bi_hb)
        mu_ans_re = item_mc(name_div, mu_y, mu_ha, mu_hb)
        assert all(isinstance(
            i, float) for i in [tr_ans_re, bi_ans_re, mu_ans_re])

        tr_ans_re = inst_mc(name_div, tr_y, tr_yt)
        bi_ans_re = inst_mc(name_div, bi_y, bi_yt)
        mu_ans_re = inst_mc(name_div, mu_y, mu_yt)
        assert np.all(np.equal(tr_ans_re, bi_ans_re))
        assert all(isinstance(i, float) for i in tr_ans_re)
        assert all(isinstance(i, float) for i in bi_ans_re)
        assert all(isinstance(i, float) for i in mu_ans_re)
