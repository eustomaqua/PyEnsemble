# coding: utf-8


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
# import pdb

from pyfair.marble.data_classify import (
    BaggingEnsembleAlgorithm, AdaBoostEnsembleAlgorithm,
    BoostingEnsemble_multiclass, calc_acc_sing_pl_and_pr)
from pyfair.facil.utils_const import check_zero


data = datasets.load_iris()
X = data.data
y = data.target
X_trn, X_tst, y_trn, y_tst = train_test_split(
    X, y, test_size=.3, random_state=651)
nb_trn = len(X_trn)
name_cls = tree.DecisionTreeClassifier()  # clf
nb_cls = 7

nb_lbl = 2  # 3
y_trn_tmp = np.random.randint(nb_lbl, size=nb_trn).tolist()
w_trn = np.random.rand(nb_trn)
w_trn /= check_zero(np.sum(w_trn).tolist())
w_trn = w_trn.tolist()


def test_bagging():
    from pyfair.marble.data_classify import _BaggingSelectTraining
    wX, wy, ti = _BaggingSelectTraining(X_trn, y_trn)
    assert all(np.unique(y_trn) == np.unique(wy))
    assert len(ti) == nb_trn == len(wX)
    assert len(ti) >= len(set(ti))  # pdb.set_trace()

    coef, clfs, ti = BaggingEnsembleAlgorithm(
        X_trn, y_trn, name_cls, nb_cls)
    address = [id(i) for i in clfs]
    assert len(set(address)) == nb_cls
    assert len(set([id(i) for i in ti])) == nb_cls
    assert len(coef) == len(ti) == nb_cls
    assert len(np.unique(coef)) == 1
    # assert all([len(i) == nb_trn for i in ti])
    assert np.shape(ti) == (nb_cls, nb_trn)

    # y_pred = [i.predict(X_tst).tolist() for i in clfs]
    # hens = calc_acc_sing_pl_and_pr(y_tst, y_pred, coef)
    # pdb.set_trace()
    return


def test_adaboost():
    from pyfair.marble.data_classify import _resample

    wX, wy, ti = _resample(X_trn, y_trn_tmp, w_trn)
    assert 1 <= len(np.unique(wy)) <= nb_lbl
    assert len(ti) == nb_trn == len(wX)
    assert all(np.unique(y_trn_tmp) == np.unique(wy))

    coef, clfs, ti = AdaBoostEnsembleAlgorithm(
        X_trn, y_trn_tmp, name_cls, nb_cls)
    address = [id(i) for i in clfs]
    assert len(set(address)) == nb_cls
    assert len(set([id(i) for i in ti])) == nb_cls
    assert len(coef) == len(ti) == nb_cls

    y_pred = [i.predict(X_tst).tolist() for i in clfs]
    hens = calc_acc_sing_pl_and_pr(y_tst, y_pred, coef)
    assert len(hens[0]) == nb_cls
    # pdb.set_trace()
    return


def test_boosting():
    from pyfair.marble.data_classify import (
        _resample, _AdaBoostSelectTraining)

    wX, wy, ti = _resample(X_trn, y_trn, w_trn)
    assert 1 <= len(np.unique(wy)) <= len(set(y_trn))
    assert len(ti) == nb_trn
    assert all(np.unique(y_trn) == np.unique(wy))
    assert len(ti) >= len(set(ti))  # pdb.set_trace()

    wX, wy, ti = _AdaBoostSelectTraining(
        X_trn, y_trn, w_trn)
    assert all(np.unique(y_trn) == np.unique(wy))
    assert len(ti) == len(wX) == nb_trn
    assert len(ti) >= len(set(ti))  # pdb.set_trace()

    for name_ens in ["AdaBoostM1", "SAMME"]:
        coef, clfs, ti = BoostingEnsemble_multiclass(
            X_trn, y_trn, name_cls, nb_cls, name_ens)
        address = [id(i) for i in clfs]
        assert len(set(address)) == nb_cls
        assert len(coef) == len(ti) == nb_cls
        assert np.shape(ti) == (nb_cls, nb_trn)

        # pdb.set_trace()
    return


# pdb.set_trace()
