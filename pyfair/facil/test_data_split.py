# coding: utf-8


import numpy as np
# import pdb
from pyfair.facil.utils_const import synthetic_dat

from pyfair.facil.data_split import (
    sklearn_k_fold_cv, sklearn_stratify, manual_repetitive,
    scale_normalize_helper, scale_normalize_data,
    get_splited_set_acdy, sitch_cross_validation,
    situation_split1, situation_split2, situation_split3,
    manual_cross_valid)


nb_inst, nb_lbl, nb_feat = 21, 3, 5
nb_cv, k = 2, 1  # or 2,3,5
X, y = synthetic_dat(nb_lbl, nb_inst, nb_feat)


def test_sklearn():
    si = sklearn_k_fold_cv(nb_cv, y)
    assert len(si) == nb_cv
    assert all([len(j) + len(k) == nb_inst for j, k in si])
    si = sklearn_stratify(nb_cv, y, X)
    assert len(si) == nb_cv
    assert all([len(j) + len(k) == nb_inst for j, k in si])

    i_trn, i_tst = si[k]
    (_, X_val, _,
     _, y_val, _) = get_splited_set_acdy(
        np.array(X), np.array(y), [i_trn, [], i_tst])
    assert not (X_val or y_val)

    for gen in [False, True]:
        si = manual_repetitive(nb_cv, y, gen)
        assert np.shape(si) == (nb_cv, nb_inst)

    for typ in ['standard', 'min_max', 'min_abs', 'normalize']:
        scaler = scale_normalize_helper(typ)  # , X)
        scaler, X_trn, X_val, X_tst = scale_normalize_data(
            scaler, X, [], X)
        assert np.shape(X_trn) == np.shape(X_tst)
    return


def test_CV_new():
    si = sitch_cross_validation(nb_cv, y, 'cv2')
    assert all([len(j) + len(k) == nb_inst for j, k in si])
    # pdb.set_trace()
    si = sitch_cross_validation(nb_cv, y, 'cv3')
    assert all([len(i) + len(j) + len(
        k) == nb_inst for i, j, k in si])
    # pdb.set_trace()
    si = manual_cross_valid(5, y)
    assert len(si) == 5
    # pdb.set_trace()

    pr_trn, pr_tst = .7, .2
    si = situation_split1(y, pr_trn, None)    # si,=
    assert len(si[0][0]) + len(si[0][1]) == nb_inst
    # pdb.set_trace()
    si = situation_split1(y, pr_trn, pr_tst)  # si,=
    assert sum([len(i) for i in si[0]]) == nb_inst
    # pdb.set_trace()

    si = situation_split2(pr_trn, nb_cv, y)
    assert all([len(j) + len(k) == nb_inst for j, k in si])
    # pdb.set_trace()
    si = situation_split3(pr_trn, pr_tst, nb_cv, y)
    assert all([len(i) + len(j) + len(
        k) == nb_inst for i, j, k in si])

    # pdb.set_trace()
    return


# =================================
# -------------------------------------
# fairml/widget/data_split.py


pr_trn = .8
pr_tst = .1
nb_iter = 5

nb_lbl = 4
nb_spl = nb_lbl * 10
nb_feat = 4

y_trn = np.random.randint(nb_lbl, size=nb_spl).tolist()
y_val = np.random.randint(nb_lbl, size=nb_spl).tolist()
y_tst = np.random.randint(nb_lbl, size=nb_spl).tolist()
y_prime = np.concatenate([y_trn, y_val, y_tst], axis=0).tolist()

X_trn = np.random.rand(nb_spl, nb_feat)  # .tolist()
X_val = np.random.rand(nb_spl, nb_feat)  # .tolist()
X_tst = np.random.rand(nb_spl, nb_feat)  # .tolist()


def test_sp():
    # from fairml.widget.data_split import (
    #     situation_split2, situation_split3)
    for nb_iter in [2, 3, 5]:

        split_idx = situation_split2(pr_trn, nb_iter, y_trn)
        for i_trn, i_val in split_idx:
            z_trn = set(i_trn)
            z_val = set(i_val)
            assert len(i_trn) == len(z_trn) >= 1
            assert len(i_val) == len(z_val) >= 1
            assert len(i_trn) + len(i_val) == nb_spl
        split_tmp = situation_split2(pr_trn, nb_iter, y_trn)
        assert id(split_tmp) != id(split_idx)

        split_idx = situation_split3(
            pr_trn, pr_tst, nb_iter, y_trn)
        for i_trn, i_val, i_tst in split_idx:
            z_trn = set(i_trn)
            z_val = set(i_val)
            z_tst = set(i_tst)
            assert len(i_trn) == len(z_trn) >= 1
            assert len(i_val) == len(z_val) >= 1
            assert len(i_tst) == len(z_tst) >= 1
            assert len(i_trn) + len(i_val) + len(i_tst) == nb_spl
        split_tmp = situation_split3(
            pr_trn, pr_tst, nb_iter, y_trn)
        assert id(split_idx) != id(split_tmp)


def test_cv():
    # from fairml.widget.data_split import (
    #     situation_cross_validation, situation_split1)
    # y = np.concatenate([y_trn, y_val, y_tst], axis=0).tolist()
    for nb_iter in [2, 3, 5]:

        split_type = "cross_valid_v2"
        split_idx = sitch_cross_validation(
            nb_iter, y_prime, split_type)
        for i_trn, i_tst in split_idx:
            z_trn, z_tst = set(i_trn), set(i_tst)
            assert len(z_trn) + len(z_tst) == nb_spl * 3
            assert len(i_trn) == len(z_trn) >= 1
            assert len(i_tst) == len(z_tst) >= 1
        split_tmp = sitch_cross_validation(
            nb_iter, y_prime, split_type)
        assert id(split_idx) != id(split_tmp)

        split_type = "cross_valid_v3"
        split_idx = sitch_cross_validation(
            nb_iter, y_prime, split_type)
        for i_trn, i_val, i_tst in split_idx:
            z_trn, z_val, z_tst = set(
                i_trn), set(i_val), set(i_tst)
            assert len(z_trn) + len(
                z_val) + len(z_tst) == nb_spl * 3
            assert len(i_trn) == len(z_trn) >= 1
            assert len(i_val) == len(z_val) >= 1
            assert len(i_tst) == len(z_tst) >= 1
        split_tmp = sitch_cross_validation(
            nb_iter, y_prime, split_type)
        assert id(split_idx) != id(split_tmp)

    y = np.random.randint(4, size=21)
    split_idx = situation_split1(y, 0.6, 0.2)
    split_idx = situation_split1(y, 0.8, 0.1)
    split_idx = situation_split1(y, 0.6)
    split_idx = situation_split1(y, 0.8)
    split_idx = situation_split1(y, 0.9)
    assert len(split_idx) == 1


def test_re():
    # from fairml.widget.data_split import (
    #     manual_repetitive, manual_cross_valid,
    #     scale_normalize_helper, scale_normalize_dataset)

    X_prime = np.concatenate([
        X_trn, X_val, X_tst], axis=0).tolist()
    assert np.shape(X_prime) == (nb_spl * 3, nb_feat)
    for nb_iter in [2, 3, 5]:
        split_idx = manual_repetitive(nb_iter, y_prime)
        assert all([len(i) == nb_spl * 3 for i in split_idx])
        split_idx = manual_repetitive(nb_iter, y_prime, True)
        assert all([len(i) == nb_spl * 3 for i in split_idx])
        split_idx = manual_cross_valid(nb_iter, y_prime)  # , X)
        assert all([len(xy) + len(
            z) == nb_spl * 3 for xy, z in split_idx])

    Xl_trn = X_trn.tolist()
    Xl_val = X_val.tolist()
    Xl_tst = X_tst.tolist()
    for scale_type in ["standard", "min_max", "normalize"]:
        scaler = scale_normalize_helper(scale_type)
        scatmp = scale_normalize_helper(scale_type)
        assert id(scaler) != id(scatmp)
        resler, X1, X2, X3 = scale_normalize_data(
            scaler, Xl_trn, Xl_val, Xl_tst)  # scale_type
        restmp, X4, X5, X6 = scale_normalize_data(
            scatmp, Xl_trn, Xl_val, Xl_tst)  # scale_type
        assert id(resler) != id(restmp)
        assert len(set(map(id, [X1, X2, X3, X4, X5, X6]))) == 6
        assert id(scaler) == id(resler)
        assert id(scatmp) == id(restmp)
