# coding: utf-8
# fairml.widget. (asst/past/utils)
# pyfairness.plain.


def test_saver():
    from pyfair.facil.utils_saver import (
        get_elogger, rm_ehandler, console_out,
        elegant_print)

    console_out('filtest.txt')
    logger, fm, fh = get_elogger('logtest', 'filtest.txt')
    elegant_print('test print')
    elegant_print('test print log', logger)
    rm_ehandler(logger, fm, fh)

    import os
    os.remove('filtest.txt')
    return


def test_timer():
    from pyfair.facil.utils_timer import (
        fantasy_durat, elegant_durat, elegant_dated,
        fantasy_durat_major)
    import time
    pi = 3.141592653589793

    for verb in [True, False]:
        print(fantasy_durat(pi - 3, verb))

        # print(fantasy_durat(pi, verb, False))
        print(fantasy_durat(pi, verb, True))
        print(fantasy_durat_major(pi, True))  # verb,True))

        print(elegant_durat(pi, verb))  # same
        print(fantasy_durat_major(pi, True))  # verb,False))

    tim = time.time()
    print(elegant_dated(tim, 'num'))
    print(elegant_dated(tim, 'txt'))
    print(elegant_dated(tim, 'day'))
    print(elegant_dated(tim, 'wks'))
    return


def test_const():
    from pyfair.facil.utils_const import (
        CONST_ZERO, CONST_DIFF, check_zero, check_equal,
        check_signed_zero, unique_column, judge_transform_need,
        judge_mathcal_Y, np_sum, np_prod,  # renew_rand_seed,
        synthetic_lbl, synthetic_dat, synthetic_set, synthetic_clf,
        random_seed_generator)
    # from fairml.widget.utils_const import random_seed_generator as renew_rand_seed
    import numpy as np
    renew_rand_seed = random_seed_generator

    assert check_zero(0) == CONST_ZERO
    assert check_equal(0, CONST_DIFF / 2)
    assert not check_equal(0, CONST_DIFF)
    assert check_zero(1e-8) and check_zero(0.) != 0.
    assert check_equal(1e-8, 1e-9)
    assert not check_equal(1e-5, 1e-6)
    assert check_signed_zero(0) == 0
    assert check_signed_zero(1e-19) == CONST_ZERO
    assert check_signed_zero(-1e-19) == -CONST_ZERO

    y = np.random.randint(4, size=17).tolist()
    vY, dY = judge_transform_need(y)
    assert dY <= 4
    tmp = judge_mathcal_Y(dY)
    # assert all([i == j for i, j in zip(vY, tmp)])
    # assert check_equal(vY, tmp)
    assert len(vY) == len(tmp) <= 4

    y = np.random.randint(2, size=17).tolist()
    vY, dY = judge_transform_need(y)
    assert dY == 2  # 1
    tmp = judge_mathcal_Y(dY)
    # assert all([i == j for i, j in zip(vY, tmp)])
    assert check_equal(vY, tmp)

    nb_col = 26 + 26**2 + 26**3
    alphabet = [chr(i) for i in range(97, 123)]
    group_AZ_upper = unique_column(nb_col)
    group_az_lower = unique_column(nb_col, alphabet)
    assert all(65 <= ord(i) <= 90 for i in group_AZ_upper[:26])
    assert all(97 <= ord(i) <= 122 for i in group_az_lower[:26])
    answer = unique_column(26)
    answer = np.array([ord(i) for i in answer])
    assert np.all(answer >= 65)
    assert np.all(answer <= 91)
    assert np_sum(range(1, 11)) == 55
    assert np_prod(range(1, 5)) == 24

    _, prng = renew_rand_seed('fixed_tim')  # psed,
    nb_lbl, nb_spl, nb_ftr, nb_clf = 3, 21, 4, 7
    X, y = synthetic_dat(nb_lbl, nb_spl, nb_ftr)
    assert np.shape(X) == (nb_spl, nb_ftr) and len(y) == nb_spl
    assert 0 <= min(y) < max(y) <= nb_lbl - 1
    y, yt, coef = synthetic_set(nb_lbl, nb_spl, nb_clf)
    assert np.shape(yt) == (nb_clf, nb_spl)
    assert len(y) == nb_spl and nb_clf == len(coef)
    assert 0 < sum(coef) <= 1.0000000000000002
    err = .2
    y_spl = synthetic_lbl(nb_lbl, nb_spl, prng)
    yt = synthetic_clf(y_spl, nb_clf, err, prng=prng)
    acc_opposite = np.mean(np.not_equal(y_spl, yt), axis=1)
    acc_opposite = acc_opposite.tolist()
    assert all([0 <= i <= err for i in acc_opposite])
    del X, y, yt, coef, y_spl, err, nb_lbl, nb_spl, nb_ftr, nb_clf
    return


def test_simulator():
    from pyfair.facil.utils_const import (
        synthetic_lbl, synthetic_dat, synthetic_set, synthetic_clf)
    from pyfair.facil.utils_const import \
        random_seed_generator as renew_rand_seed
    import numpy as np

    _, prng = renew_rand_seed('fixed_tim')  # psed,
    nb_lbl, nb_spl, nb_ftr, nb_clf = 3, 21, 4, 7
    X, y = synthetic_dat(nb_lbl, nb_spl, nb_ftr)
    assert np.shape(X) == (nb_spl, nb_ftr) and len(y) == nb_spl
    assert 0 <= min(y) < max(y) <= nb_lbl - 1
    y, yt, coef = synthetic_set(nb_lbl, nb_spl, nb_clf)
    assert np.shape(yt) == (nb_clf, nb_spl)
    assert len(y) == nb_spl and nb_clf == len(coef)
    assert 0 < sum(coef) <= 1.0000000000000004  # 1.0000000000000002
    err = .2
    y_spl = synthetic_lbl(nb_lbl, nb_spl, prng)
    yt = synthetic_clf(y_spl, nb_clf, err, prng=prng)
    acc_opposite = np.mean(np.not_equal(y_spl, yt), axis=1)
    acc_opposite = acc_opposite.tolist()
    assert all([0 <= i <= err for i in acc_opposite])
    del X, y, yt, coef, y_spl, err, nb_lbl, nb_spl, nb_ftr, nb_clf
    return
