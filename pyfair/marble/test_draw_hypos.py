# coding: utf-8

import numpy as np
from pyfair.facil.utils_const import check_equal
# from prgm.nucleus.utils_hypos import *
from pyfair.marble.draw_hypos import (
    _regulate_vals, _regulate_sign,
    _Friedman_sequential, _Friedman_successive,
    binomial_test, t_test, scipy_ttest_for1, scipy_ttest_for2,
    paired_5x2cv_test, paired_t_tests, scipy_ttest_pair,
    McNemar_test, Friedman_init, Friedman_test, Nememyi_posthoc_test,
    cmp_paired_avg, cmp_paired_wtl, comp_t_init, comp_t_prep,
    Pearson_correlation)


accs = [0.7, 0.8, 0.6, 0.85, 0.65]
errs = _regulate_vals(accs, 'acc')
k = 5

# err_1 = [0.1, 0.2, 0.15, 0.21, 0.3]
# err_2 = [0.3, 0.2, 0.16, 0.17, 0.13]
err_1 = [0.15, 0.21, 0.1, 0.2, 0.3]
err_2 = [0.3, 0.17, 0.2, 0.16, 0.13]


def test_prelim():
    assert all([i + j == 1 for i, j in zip(errs, accs)])
    vals = _regulate_vals(errs, 'acc')  # Note if you want 1-x
    assert all([i == j for i, j in zip(vals, accs)])

    assert _regulate_sign(True).startswith('Accept')
    assert _regulate_sign(False).startswith('Reject')


def test_subsubsec_241():
    # hypothetical_test

    mark, tau_b = binomial_test(max(errs), 11, .3)
    assert 'H0' in mark and 0 <= tau_b <= 1
    # assert mark.endswith('H0') and 0 <= tau_b <= 1

    mark, tau_b = binomial_test(max(errs), 7, .4)
    assert 'H0' in mark and 0 <= tau_b <= 1
    # assert mark.endswith('H0') and 0 <= tau_b <= 1

    _, tau_t, mu, s2 = t_test(errs, k, .3, 0.05)
    _, tau_t, mu, s2 = t_test(errs, k, .4, 0.05)
    _, tau_t, mu, s2 = t_test(errs, k, .4, 0.10)
    # assert 'H0' in mark  # _:mark
    assert isinstance(mu, float)

    for alpha in [.05, .1]:
        mark, tau_t, mu, s2 = t_test(errs, k, .18, alpha)
        clue = scipy_ttest_for1(errs, .18, alpha)
        assert mark == clue[0] and tau_t == clue[1]


def test_subsubsec_242():

    for alpha in [.05, .1]:
        mark, tau_t = paired_t_tests(err_1, err_2, k, alpha)
        clue = scipy_ttest_pair(err_2, err_1, alpha)
        assert mark == clue[0] and check_equal(tau_t, clue[1])

        cnew = scipy_ttest_pair(err_2, err_1, alpha, 'sg')
        assert mark == cnew[0] and check_equal(tau_t, cnew[1])
        assert clue == cnew

        clue = scipy_ttest_for2(err_1, err_2, alpha)
        assert mark == clue[0] and check_equal(tau_t, clue[1])

    for alpha in [.05, .1]:
        mark, tau_t = paired_5x2cv_test(
            err_1 + err_2, err_2 + err_1, k=k, alpha=alpha)
        clue = paired_5x2cv_test(
            err_2 + err_1, err_1 + err_2, k=k, alpha=alpha)
        assert mark == clue[0] and -tau_t == clue[1]

        paired_5x2cv_test(err_1 + err_1, err_2 + err_2, k, alpha)
        paired_5x2cv_test(err_2 + err_2, err_1 + err_1, k, alpha)


def test_subsubsec_243():
    nb_inst, nb_labl = 11, 2
    y = np.random.randint(nb_labl, size=nb_inst)
    ha = np.random.randint(nb_labl, size=nb_inst)
    hb = np.random.randint(nb_labl, size=nb_inst)

    for alpha in [.05, .1]:
        mark, tau_t = McNemar_test(ha, hb, y, alpha=alpha)
        assert 0 <= tau_t  # <= 1
        assert mark[:6] in ('Accept', 'Reject')  # ' H0'


# '''
# U = np.arange(24)
# np.random.shuffle(U)
# U.resize(6, 4)
#
# U[5, 3] = U[5, 1]
# U[1, 0] = U[1, 2]
# U = U.astype('float')
# '''


def _generate_U(N=6, k=4):
    U = np.arange(N * k)
    np.random.shuffle(U)
    U.resize(N, k)

    U[N - 1, k - 1] = U[N - 1, 1]
    U[1, 0] = U[1, 2]
    U = U.astype('float')

    return U


def test_subsubsec_244():
    U = _generate_U(6, 4)
    rank = _Friedman_sequential(U, 'descend')
    idx_bar = _Friedman_successive(U, rank)
    assert np.all(np.equal(np.sum(idx_bar, axis=1),
                           np.sum(rank, axis=1)))

    for sz in [(5, 4), (4, 3)]:
        U = _generate_U(*sz)
        # rank, idx_bar = Friedman_init(U, alpha=.05)
        # mark, = Friedman_test(U, 'descend', .05)

        _, idx_bar = Friedman_init(U, 'descend')
        # mark, tau_F, tau_chi2, CD = Friedman_test(idx_bar, .05)

        mark, tau_F, tau_chi2 = Friedman_test(idx_bar, .05)
        CD = Nememyi_posthoc_test(idx_bar, .05)
        assert len(CD) == 2

        mark, tau_F, tau_chi2 = Friedman_test(idx_bar, .1)
        CD, q_alpha = Nememyi_posthoc_test(idx_bar, .1)
        assert isinstance(CD, float) and isinstance(q_alpha, float)


def test_paired_t():
    # import numpy as np
    # from fairml.widget.utils_const import check_equal
    # nb_inst, ep_a, ep_b = 21, .7, .3
    k, ep_a, ep_b = 5, .7, .3  # nb_cv

    acc_1 = np.random.rand(k) * ep_b + ep_a
    acc_2 = np.random.rand(k) * ep_b + ep_a
    err_1 = 1 - acc_1
    err_2 = 1 - acc_2
    pct_1 = acc_1 * 100
    pct_2 = acc_2 * 100
    # error rate, percentage

    acc_1 = acc_1.tolist()
    acc_2 = acc_2.tolist()
    err_1 = err_1.tolist()
    err_2 = err_2.tolist()
    pct_1 = pct_1.tolist()
    pct_2 = pct_2.tolist()

    _, _, GA_a, GB_a = comp_t_init(acc_1, acc_2)
    _, _, GA_e, GB_e = comp_t_init(err_1, err_2)
    _, _, GA_p, GB_p = comp_t_init(pct_1, pct_2)
    # assert GA_a[0] + GA_e[0] == 1 and GA_a[1] == GA_e[1]
    assert check_equal(GA_a[0] + GA_e[0], 1)
    assert check_equal(GB_a[0] + GB_e[0], 1)
    assert check_equal(GA_a[1], GA_e[1])
    assert check_equal(GB_a[1], GB_e[1])
    # assert GB_a[0] + GB_e[0] == 1 and GB_a[1] == GB_e[1]
    assert check_equal(GA_a[0] * 100, GA_p[0])
    assert check_equal(GB_a[0] * 100, GB_p[0])
    assert check_equal(GA_a[1] * 100, GA_p[1])
    assert check_equal(GB_a[1] * 100, GB_p[1])

    res_a = comp_t_prep(acc_1, acc_2)
    res_e = comp_t_prep(err_1, err_2)
    res_p = comp_t_prep(pct_1, pct_2)
    assert res_a[0] == res_e[0] == res_p[0]
    assert res_a[1] == res_e[1] == res_p[1]
    ans_a = comp_t_prep(acc_1, acc_2, method='scipy')
    assert res_a[0] == ans_a[0] and res_a[1] == ans_a[1]

    ans_a = cmp_paired_avg(GA_a, GB_a)
    ans_p = cmp_paired_avg(GA_p, GB_p)
    ans_e = cmp_paired_avg(GA_e, GB_e, 'ascend')
    assert ans_a == ans_p == ans_e
    res_a = cmp_paired_wtl(GA_a, GB_a, res_a[0], res_a[1], 'descend')
    res_p = cmp_paired_wtl(GA_p, GB_p, res_p[0], res_p[1], 'descend')
    res_e = cmp_paired_wtl(GA_e, GB_e, res_e[0], res_e[1])
    assert res_a == res_p == res_e


def test_Pearson():
    n = 11
    X = np.random.rand(n)
    Y = np.random.rand(n)
    pear, cov = Pearson_correlation(X, Y)
    corr = np.corrcoef(X, Y)
    assert check_equal(pear, [corr[1, 0], corr[0, 1]])
    corr = np.corrcoef(Y, X)
    assert check_equal(pear, [corr[1, 0], corr[0, 1]])
    corr = Pearson_correlation(Y, X)
    assert check_equal(pear, corr[0])
    assert check_equal(cov, corr[1])
    # pdb.set_trace()
    return
