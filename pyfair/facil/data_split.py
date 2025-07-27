# coding: utf-8
# Author: Yijun
#
# Target:
#   Split data for experiments
#   Split one dataset into "training/validation/test" datasets
#
#   Research and Applications of Diversity in Ensemble Classification
#   Oracle bounds regarding fairness for weighted voting
#


from copy import deepcopy
import gc
import numpy as np

from sklearn import preprocessing
from sklearn import model_selection

gc.enable()


# =================================
# split the data by sklearn
# =================================


# Split situations 2 & 3
# Cross-Validation (CV)
# -------------------------------------
# different ways to split data


# split the data


def sklearn_k_fold_cv(num_cv, y):
    kf = model_selection.KFold(n_splits=num_cv)
    kf.get_n_splits(y)  # kf.get_n_splits(X, y)
    # rkf = model_selection.RepeatedKFold(
    #     n_splits=num_cv, n_repeats=2)
    split_idx = []
    for trn, tst in kf.split(y):
        # split_idx.append((i_trn, i_tst))
        split_idx.append([trn.tolist(), tst.tolist()])
    return split_idx  # element: np.ndarray


def sklearn_stratify(num_cv, y, X):
    skf = model_selection.StratifiedKFold(n_splits=num_cv)
    skf.get_n_splits(X, y)
    split_idx = []
    for trn, tst in skf.split(X, y):
        split_idx.append([trn.tolist(), tst.tolist()])
    return split_idx  # element: not np.ndarray


# @numba.jit(nopython=True)
def manual_repetitive(nb_cv, y, gen=False):
    num = len(y)
    if not gen:
        split_idx = [list(range(num)) for _ in range(nb_cv)]
        for i in range(nb_cv):
            np.random.shuffle(split_idx[i])
        return split_idx
    split_idx = [np.random.randint(
        num, size=num).tolist() for _ in range(nb_cv)]
    return split_idx


# data regularisation
# preprocessing for feature normalisation
# ----------------------------------


def scale_normalize_helper(scale_type):
    assert scale_type in [
        "standard", "min_max", "min_abs", "normalize",
        "normalise",
    ], LookupError("Correct the `scale_type` please.")

    if scale_type == "min_max":    # min_max_scaler
        scaler = preprocessing.MinMaxScaler()
    elif scale_type == "min_abs":  # max_abs_scaler
        scaler = preprocessing.MaxAbsScaler()
    elif scale_type in ["normalize", "normalise"]:
        # pdb.set_trace()          # normaliser
        scaler = preprocessing.Normalizer()
    else:
        scaler = preprocessing.StandardScaler()
    return scaler


def scale_normalize_data(scaler, X_trn, X_val, X_tst):
    # def scale_normalize_data():
    # scaler = scale_normalize_helper(scale_type)
    # scaler.fit(X_trn)  # id(scaler) would not change
    scaler = scaler.fit(X_trn)  # .fit_transform()
    X_trn = scaler.transform(X_trn)
    X_val = [] if not X_val else scaler.transform(X_val)
    X_tst = scaler.transform(X_tst)
    X_trn, X_tst = X_trn.tolist(), X_tst.tolist()
    X_val = X_val.tolist() if len(X_val) > 0 else []
    return scaler, X_trn, X_val, X_tst  # deepcopy


# get splited datasets


def get_splited_set_acdy(X, y, split_idx_item):
    # def according_index_split_train_valid_test():
    # X, y: np.ndarray
    idx_trn, idx_val, idx_tst = split_idx_item
    X_trn, y_trn = X[idx_trn], y[idx_trn]
    X_tst, y_tst = X[idx_tst], y[idx_tst]
    X_val, y_val = [], []
    if len(idx_val) > 0:
        X_val = X[idx_val].tolist()
        y_val = y[idx_val].tolist()
    return X_trn.tolist(), X_val, X_tst.tolist(), \
        y_trn.tolist(), y_val, y_tst.tolist()


# =================================
# split the data (cross validation) manually
# =================================


def _sub_sp_indices(y):
    y = np.array(y)
    vY = np.unique(y).tolist()
    dY = len(vY)
    iY = [np.where(y == j)[0] for j in vY]  # indices
    lY = [len(j) for j in iY]               # length
    # tY = [np.arange(j) or np.copy(j) for j in lY]
    # tY = deepcopy(iY)  # tY = iY.copy()
    tY = [deepcopy(j) for j in iY]        # tmp_index
    for j in tY:
        np.random.shuffle(j)
    tmp_idx = [np.arange(j) for j in lY]
    return dY, iY, lY, tY, tmp_idx  # CC=5


# def _sub_sp_2sets(lY, iY, dY, tmp_idx, nb_cv,
#                   pr_trn):  # nb_y, pr_trn):
def _sub_sp_2sets(dY, iY, tmp_idx, sY,
                  nb_y, nb_tst):

    i_trn = [iY[i][tmp_idx[i][:sY[i]]] for i in range(dY)]
    i_tst = [iY[i][tmp_idx[i][sY[i]:]] for i in range(dY)]
    i_trn = np.concatenate(i_trn, axis=0).tolist()
    i_tst = np.concatenate(i_tst, axis=0).tolist()

    if len(i_tst) == 0:
        i_tst = list(set(
            np.random.randint(nb_y, size=nb_tst)))

    # tmp = (i_trn, i_tst)
    # split_idx.append(tmp)
    # del i_trn, i_tst, tmp
    return (i_trn, i_tst)  # (deepcopy(i_trn), deepcopy(i_tst))


# def _sub_sp_3sets(lY, iY, dY, tmp_idx, nb_cv,
#                   # nb_y, pr_trn, pr_tst):
#                   pr_trn, pr_tst):
def _sub_sp_3sets(dY, iY, tmp_idx, sY,
                  nb_y, nb_tst, nb_val):

    i_trn = [iY[i][tmp_idx[i][: sY[i][0]]] for i in range(dY)]
    i_tst = [iY[i][tmp_idx[i][
        sY[i][0]: sY[i][1]]] for i in range(dY)]
    i_val = [iY[i][tmp_idx[i][sY[i][1]:]] for i in range(dY)]
    i_trn = np.concatenate(i_trn, axis=0).tolist()
    i_tst = np.concatenate(i_tst, axis=0).tolist()
    i_val = np.concatenate(i_val, axis=0).tolist()

    if len(i_tst) == 0:
        i_tst = list(set(
            np.random.randint(nb_y, size=nb_tst)))
    if len(i_val) == 0:
        i_val = list(set(
            np.random.randint(nb_y, size=nb_val)))

    # tmp = (i_trn, i_val, i_tst)
    # split_idx.append(tmp)
    # del i_trn, i_val, i_tst, tmp
    return (i_trn, i_val, i_tst)
    # return (deepcopy(i_trn), deepcopy(i_val), deepcopy(i_tst))


def _sub_sp_alt_2set(dY, tY, sY):
    i_trn, i_tst = [], []
    for i in range(dY):
        i_trn.append(tY[i][: sY[i]])
        i_tst.append(tY[i][sY[i]:])
    i_trn = np.concatenate(i_trn, axis=0).tolist()
    i_tst = np.concatenate(i_tst, axis=0).tolist()
    return (i_trn, i_tst)


def _sub_sp_alt_3set(dY, tY, sY):
    i_trn, i_val, i_tst = [], [], []
    for i in range(dY):
        i_trn.append(tY[i][: sY[i][0]])
        i_tst.append(tY[i][sY[i][0]: sY[i][1]])
        i_val.append(tY[i][sY[i][1]:])
    i_trn = np.concatenate(i_trn, axis=0).tolist()
    i_tst = np.concatenate(i_tst, axis=0).tolist()
    i_val = np.concatenate(i_val, axis=0).tolist()
    return (i_trn, i_val, i_tst)


def _sub_sp_alt_cv(dY, tY, sY, k, nb_cv):
    i_trn, i_val, i_tst = [], [], []
    for i in range(dY):
        k_former = sY[i] * (k - 1)
        k_middle = sY[i] * k
        k_latter = sY[i] * (k + 1) if k != nb_cv else sY[i]

        i_tst.append(tY[i][k_former: k_middle])
        if k != nb_cv:
            i_val.append(tY[i][k_middle: k_latter])
            i_trn.append(np.concatenate([
                tY[i][k_latter:], tY[i][: k_former]], axis=0))
        else:
            i_val.append(tY[i][: k_latter])
            i_trn.append(np.concatenate([
                tY[i][k_middle:],
                tY[i][k_latter: k_former]], axis=0))

    i_tst = np.concatenate(i_tst, axis=0).tolist()
    i_val = np.concatenate(i_val, axis=0).tolist()
    i_trn = np.concatenate(i_trn, axis=0).tolist()
    return i_trn, i_val, i_tst


# Cross-Validation
# ----------------------------------


def sitch_cross_validation(nb_cv, y, split_type='cv3'):
    assert split_type in [
        "cross_valid_v3", "cross_valid_v2", "cross_validation",
        "cross_valid", "cv3", "cv2"], UserWarning(
            "Check the number of sets in the cross-validation.")

    # y, vY = np.array(y), np.unique(y).tolist()
    # dY = len(vY)
    # iY = [np.where(y == j)[0] for j in vY]  # indices
    # lY = [len(j) for j in iY]               # length
    # # tY = [np.copy(j) for j in iY]  # np.arange(j),temp_index
    # tY = deepcopy(iY)     # tY = iY.copy()  # tmp_index
    # for j in tY:
    #     np.random.shuffle(j)
    # sY = [int(np.floor(j / nb_cv)) for j in lY]  # split length
    # if nb_cv in [2, 3, 1]:
    #     sY = [int(np.floor(j / (nb_cv + 1))) for j in lY]
    #
    # split_idx = []
    # for k in range(1, nb_cv + 1):
    #     i_tst, i_val, i_trn = [], [], []
    #     for i in range(dY):
    #         k_former = sY[i] * (k - 1)
    #         k_middle = sY[i] * k
    #         k_latter = sY[i] * (k + 1) if k != nb_cv else sY[i]
    #
    #         i_tst.append(tY[i][k_former: k_middle])
    #         if k != nb_cv:
    #             i_val.append(tY[i][k_middle: k_latter])
    #             i_trn.append(np.concatenate([
    #                 tY[i][k_latter:], tY[i][: k_former]], axis=0))
    #         else:
    #             i_val.append(tY[i][: k_latter])
    #             i_trn.append(np.concatenate([
    #                 tY[i][k_middle:],
    #                 tY[i][k_latter: k_former]], axis=0))
    #
    #     i_tst = np.concatenate(i_tst, axis=0).tolist()
    #     i_val = np.concatenate(i_val, axis=0).tolist()
    #     i_trn = np.concatenate(i_trn, axis=0).tolist()
    #     if split_type.endswith("v2"):
    #         # "cross_valid_v2" or "cross_validation"
    #         tmp = (i_trn + i_val, i_tst)  # deepcopy()
    #     else:
    #         tmp = (i_trn, i_val, i_tst)   # deepcopy(),i_val.copy()
    #     split_idx.append(tmp)             # deepcopy(temp_)
    # #     del k_former, k_middle, k_latter, i_tst, i_val, i_trn
    # # del k, y, vY, dY, iY, lY, tY, sY
    # gc.collect()
    # return split_idx

    dY, _, lY, tY, _ = _sub_sp_indices(y)
    sY = [int(np.floor(j / float(nb_cv))) for j in lY]
    if nb_cv in [2, 3, 1]:
        sY = [int(np.floor(j / (nb_cv + 1.))) for j in lY]
    split_idx = []  # sY_alt = []
    for k in range(1, nb_cv + 1):
        i_trn, i_val, i_tst = _sub_sp_alt_cv(dY, tY, sY, k, nb_cv)
        if split_type.endswith("v2"):
            tmp = (i_trn + i_val, i_tst)
        else:
            tmp = (i_trn, i_val, i_tst)
        split_idx.append(tmp)

        # if split_type.endswith("v2"):
        #     tmp = (deepcopy(i_trn + i_val), deepcopy(i_tst))
        # else:
        #     tmp = (deepcopy(
        #         i_trn), deepcopy(i_val), deepcopy(i_tst))
        # split_idx.append(deepcopy(tmp))
    return split_idx  # return deepcopy(split_idx)


def manual_cross_valid(nb_cv, y):
    split_idx = sitch_cross_validation(nb_cv, y)
    return [[x + y, z] for x, y, z in split_idx]


# Split situation 1:
#   i.e. one single iteration
# ----------------------------------

def situation_split1(y, pr_trn, pr_tst=None):
    # y = np.array(y)
    # vY = np.unique(y).tolist()
    # dY = len(vY)
    # iY = [np.where(y == j)[0] for j in vY]
    # lY = [len(j) for j in iY]  # index & length
    #
    # # tmp_idx = [np.arange(j) for j in lY]
    # tY = deepcopy(iY)  # [np.copy(j) for j in iY]
    # for j in tY:
    #     np.random.shuffle(j)
    # nb_y = len(y)
    # nb_trn = int(np.round(nb_y * pr_trn))
    # nb_trn = min(max(nb_trn, 1), nb_y - 1)
    #
    # if pr_tst is None:
    #     sY = [int(np.max(
    #         [np.round(j * pr_trn), 1])) for j in lY]
    #     nb_tst = nb_y - nb_trn
    # else:
    #     # pr_val = 1. - pr_trn - pr_tst
    #     sY = [[int(np.max([np.round(j * i), 1])) for i in (
    #         pr_trn, pr_trn + pr_tst)] for j in lY]
    #     nb_tst = int(np.round(nb_y * pr_tst))
    #     nb_tst = min(max(nb_tst, 1), nb_y - 1)
    #     # del pr_val
    #
    # i_tst, i_val, i_trn = [], [], []
    # for i in range(dY):
    #     if pr_tst is not None:
    #         i_trn.append(tY[i][: sY[i][0]])
    #         i_tst.append(tY[i][sY[i][0]: sY[i][1]])
    #         i_val.append(tY[i][sY[i][1]:])
    #     else:
    #         i_trn.append(tY[i][: sY[i]])
    #         i_tst.append(tY[i][sY[i]:])
    #
    # i_trn = np.concatenate(i_trn, axis=0).tolist()
    # i_tst = np.concatenate(i_tst, axis=0).tolist()
    # # if len(i_tst) == 0:
    # #     i_tst = list(set(
    # #         np.random.randint(nb_y, size=nb_tst)))
    # if not pr_tst:  # pr_tst is None
    #     # i_val = list(set(
    #     #     np.random.randint(nb_y, size=nb_val)))
    #     return [(i_trn, i_tst)]
    # # nb_val = nb_y - nb_trn - nb_tst
    # i_val = np.concatenate(i_val, axis=0).tolist()
    # # if len(i_val) == 0:
    # #     i_val = list(set(
    # #         np.random.randint(nb_y, size=nb_val)))
    # split_idx = [(i_trn, i_val, i_tst)]
    # return split_idx

    # nb_y = len(y)
    # dY, iY, lY, tY, _ = _sub_sp_indices(y)
    dY, _, lY, tY, _ = _sub_sp_indices(y)
    if pr_tst is None:
        sY = [int(np.max([
            np.round(j * pr_trn), 1])) for j in lY]
        # nb_tst = int(np.round(nb_y * (1. - pr_trn)))
        # nb_tst = min(max(nb_tst, 1), nb_y - 1)
        tmp = _sub_sp_alt_2set(dY, tY, sY)
        return [tmp]
    sY = [[int(np.max([
        np.round(j * i), 1])) for i in (
        pr_trn, pr_trn + pr_tst)] for j in lY]
    # nb_tst = int(np.round(nb_y * pr_tst))
    # nb_tst = min(max(nb_tst, 1), nb_y - 1)
    tmp = _sub_sp_alt_3set(dY, tY, sY)
    return [tmp]


# Split situation 2

def situation_split2(pr_trn, nb_cv, y_trn):
    # vY = np.unique(y_trn).tolist()
    # dY = len(vY)
    # iY = [np.where(np.equal(y_trn, j))[0] for j in vY]
    # lY = [len(j) for j in iY]  # length
    # sY = [int(np.max([np.round(
    #     j * pr_trn), 1])) for j in lY]    # split_loca
    # tmp_idx = [np.arange(j) for j in lY]  # tem_idx
    #
    # nb_trn = len(y_trn)
    # nb_val = int(np.round(nb_trn * (1. * pr_trn)))
    # nb_val = min(max(nb_val, 1), nb_trn - 1)
    # split_idx = []
    # for k in range(nb_cv):
    #     for i in tmp_idx:
    #         np.random.shuffle(i)
    #
    #     i_trn = [iY[i][tmp_idx[i][:sY[i]]] for i in range(dY)]
    #     i_val = [iY[i][tmp_idx[i][sY[i]:]] for i in range(dY)]
    #     i_trn = np.concatenate(i_trn, axis=0).tolist()
    #     i_val = np.concatenate(i_val, axis=0).tolist()
    #     if len(i_val) == 0:
    #         i_val = list(set(
    #             np.random.randint(nb_trn, size=nb_val)))
    #     tem = (i_trn, i_val)  # .copy(), deepcopy
    #     split_idx.append(tem)
    #
    #     del i_trn, i_val, tem, i
    # del k, tmp_idx, sY, lY, dY, vY
    # gc.collect()
    # return split_idx

    # dY, iY, lY, _, tmp_idx = _sub_sp_indices(y_trn)
    # return _sub_sp_2sets(lY, iY, dY, tmp_idx, nb_cv, pr_trn)
    nb_y = len(y_trn)
    nb_val = int(np.round(nb_y * (1. - pr_trn)))
    nb_val = min(max(nb_val, 1), nb_y - 1)

    dY, iY, lY, _, tmp_idx = _sub_sp_indices(y_trn)
    sY = [int(np.max([
        np.round(j * pr_trn), 1])) for j in lY]
    split_idx = []
    for _ in range(nb_cv):  # for k in
        for i in tmp_idx:
            np.random.shuffle(i)
        tmp = _sub_sp_2sets(dY, iY, tmp_idx, sY,
                            nb_y, nb_val)
        split_idx.append(tmp)  # deepcopy(tmp))
    del sY, dY, iY, lY, tmp_idx
    gc.collect()
    return split_idx      # deepcopy(split_idx)


# Split situation 3

def situation_split3(pr_trn, pr_tst, nb_cv, y):
    # # pr_val = 1. - pr_trn - pr_tst
    #
    # vY = np.unique(y).tolist()
    # dY = len(vY)
    # # iY = [np.where(np.array(y) == j)[0] for j in vY]
    # iY = [np.where(np.equal(y, j))[0] for j in vY]
    # lY = [len(j) for j in iY]  # length
    # sY = [[int(np.max([np.round(j * i), 1])) for i in (
    #     pr_trn, pr_trn + pr_tst)] for j in lY]
    # tmp_idx = [np.arange(j) for j in lY]
    #
    # nb_y = len(y)
    # nb_trn = int(np.round(nb_y * pr_trn))
    # nb_tst = int(np.round(nb_y * pr_tst))
    # # nb_val = int(np.round(nb_y * pr_val))
    # nb_trn = min(max(nb_trn, 1), nb_y)
    # nb_tst = min(max(nb_tst, 1), nb_y - 1)
    # nb_val = nb_y - nb_trn - nb_tst
    # nb_val = min(max(nb_val, 1), nb_y - 1)
    # # del pr_val  # pr_trn, pr_tst, y
    #
    # split_idx = []
    # for k in range(nb_cv):
    #     for i in tmp_idx:         # tem_idx
    #         np.random.shuffle(i)
    #     i_trn = [iY[i][tmp_idx[i][: sY[i][0]]] for i in range(dY)]
    #     i_tst = [iY[i][tmp_idx[i][
    #         sY[i][0]: sY[i][1]]] for i in range(dY)]
    #     i_val = [iY[i][tmp_idx[i][sY[i][1]:]] for i in range(dY)]
    #     i_trn = np.concatenate(i_trn, axis=0).tolist()
    #     i_val = np.concatenate(i_val, axis=0).tolist()
    #     i_tst = np.concatenate(i_tst, axis=0).tolist()
    #     if len(i_tst) == 0:
    #         i_tst = list(set(np.random.randint(nb_y, size=nb_tst)))
    #     if len(i_val) == 0:
    #         i_val = list(set(np.random.randint(nb_y, size=nb_val)))
    #     tem = (i_trn, i_val, i_tst)
    #     split_idx.append(tem)  # deepcopy()
    #
    #     del i_trn, i_val, i_tst, tem, i
    # del k, tmp_idx, sY, lY, iY, dY, vY
    # gc.collect()
    # return deepcopy(split_idx)

    # dY, iY, lY, tY, tmp_idx = _sub_sp_indices(y)
    # return _sub_sp_3sets(
    #     lY, iY, dY, tmp_idx, nb_cv, pr_trn, pr_tst)
    nb_y = len(y)
    nb_trn = int(np.round(nb_y * pr_trn))
    nb_tst = int(np.round(nb_y * pr_tst))
    nb_trn = min(max(nb_trn, 1), nb_y)
    nb_tst = min(max(nb_tst, 1), nb_y - 1)
    nb_val = nb_y - nb_trn - nb_tst
    nb_val = min(max(nb_val, 1), nb_y - 1)

    dY, iY, lY, _, tmp_idx = _sub_sp_indices(y)
    sY = [[int(np.max([np.round(j * i), 1])) for i in (
        pr_trn, pr_trn + pr_tst)] for j in lY]
    split_idx = []
    for _ in range(nb_cv):  # for k in
        for i in tmp_idx:
            np.random.shuffle(i)
        tmp = _sub_sp_3sets(dY, iY, tmp_idx, sY,
                            nb_y, nb_tst, nb_val)
        split_idx.append(tmp)  # deepcopy(tmp))
    del sY, dY, iY, tmp_idx
    gc.collect()
    return split_idx      # deepcopy(split_idx)


# =================================
# split the data
# =================================


def split_into_sets(split_type, *split_args):
    # def split_into_train_validation_test():
    assert split_type.endswith(
        "split") or split_type.startswith(
        "cross_valid"), "Error occurred when splitting the data."
    # raise UserWarning()
    # "Error occurred in `split_into_train_validation_test`."

    if split_type == "2split":
        pr_trn, nb_iter, y_trn = split_args
        split_idx = situation_split2(pr_trn, nb_iter, y_trn)
        del y_trn
    elif split_type == "3split":
        pr_trn, pr_tst, nb_iter, y = split_args
        split_idx = situation_split3(pr_trn, pr_tst, nb_iter, y)
        del y
    else:  # elif split_type == "cross_validation":
        nb_iter, y = split_args
        split_idx = situation_cross_validation(nb_iter, y, split_type)
        del y
    gc.collect()
    return deepcopy(split_idx)  # list


# INTERFACE
# Preliminaries
# ----------------------------------
# obtain data
# split data

# from core.fetch_utils import different_type_of_data
# different ways to split data  # carry_split.py
# def _spl2_step1():
#     pass

# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_validate
# sklearn
# https://scikit-learn.org/stable/modules/ensemble.html
# https://scikit-learn.org/0.24/modules/cross_validation.html
# https://scikit-learn.org/0.24/model_selection.html
