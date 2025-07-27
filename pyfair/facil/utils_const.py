# coding: utf-8
# Author: Yijun

import numpy as np


# ==========================
# Constants
# Helper functions


# -------------------------------------
# Constants (by default)


DL_DTY_FLT = 'float32'
DL_DTY_INT = 'int32'
ML_DTY_FLT = 'float64'
ML_DTY_INT = 'int64'

DTY_FLT = 'float'
DTY_INT = 'int'
DTY_BOL = 'bool'
DTY_PLT = '.pdf'
# DTY_PLT = '.eps'


CONST_ZERO = 1e-13
CONST_DIFF = 1e-7
RANDOM_SEED = None
FIXED_SEED = 7654

GAP_INF = 2 ** 31 - 1
GAP_MID = 1e8
GAP_NAN = 1e-16


# -------------------------------------
# Help(er) functions


def check_belong(tmp, *args):
    for v in args:
        if isinstance(tmp, v):
            return True
    return False


def check_zero(tmp, diff=CONST_ZERO):
    if check_belong(tmp, list, tuple, np.ndarray):
        return [check_zero(i, diff) for i in tmp]
    # namely, robust_zero()
    return tmp if tmp != 0. else diff


def non_negative(tmp):
    if check_belong(tmp, list, tuple, np.ndarray):
        return [non_negative(i) for i in tmp]
    return tmp if tmp >= 0 else 0.


def check_equal(tmp_a, tmp_b, diff=CONST_DIFF):
    # flag_a = check_belong(tmp_a, list, tuple, np.ndarray)
    # flag_b = check_belong(tmp_b, list, tuple, np.ndarray)
    # if not (flag_a or flag_b):
    #     return True if abs(tmp_a - tmp_b) < diff else False
    #
    # if flag_a and check_belong(tmp_b, int, float):
    #     tmp = [abs(i - tmp_b) < diff for i in tmp_a]
    # elif flag_b and check_belong(tmp_a, int, float):
    #     tmp = [abs(tmp_a - i) < diff for i in tmp_b]
    # elif flag_a and flag_b:
    #     # tmp = [i == j for i, j in zip(tmp_a, tmp_b)]
    #     tmp = [abs(i - j) < diff for i, j in zip(tmp_a, tmp_b)]

    flag_a = isinstance(tmp_a, (list, tuple, np.ndarray))
    flag_b = isinstance(tmp_b, (list, tuple, np.ndarray))
    if not (flag_a or flag_b):
        return True if abs(tmp_a - tmp_b) < diff else False
    if isinstance(tmp_b, (int, float)):    # flag_a and
        return all([abs(i - tmp_b) < diff for i in tmp_a])
    elif isinstance(tmp_a, (int, float)):  # flag_b and
        return all([abs(tmp_a - i) < diff for i in tmp_b])
    # else:                     # elif flag_a and flag_b:
    tmp = [abs(i - j) < diff for i, j in zip(tmp_a, tmp_b)]
    return all(tmp)


def check_signed_zero(x, diff=CONST_ZERO):
    if abs(x) > CONST_ZERO:
        return x
    elif x > 0:
        return CONST_ZERO
    elif x < 0:
        return -CONST_ZERO
    return 0.0


# -------------------------------------
# Help(er) functions cont.


def judge_transform_need(y):
    vY = sorted(set(y))  # list(set(y))
    dY = len(vY)
    if dY == 2 and (-1 in vY) and (1 in vY):
        # if dY == 2 and (-1 in vY):
        dY = 1
    return vY, dY  # 2, or ...


def judge_mathcal_Y(nc=1):
    # vY: list(range(nc)) if nc >= 2 else [-1, +1]
    if nc == 1:
        return [-1, +1]
    return list(range(nc))


def _sub_ABC(nb_col, alphabet, double):
    triple = [i + j + k for i in alphabet for j in alphabet for k in alphabet]
    index = nb_col - 26 - 26**2
    if index <= 26**3:
        return alphabet + double + triple[: index]
    return triple


def unique_column(nb_col, alphabet=None):
    # i.e., generate_unique_column_name
    #
    # alphabet = [chr(i) for i in range(97, 123)]  # 'a' etc.
    # alphabet = [chr(i) for i in range(65, 91)]  ## 'A' etc.
    # import string
    # string.ascii_lowercase

    if alphabet is None:
        alphabet = [chr(i) for i in range(65, 91)]
    if nb_col <= 26:
        return alphabet[: nb_col]

    double = [i + j for i in alphabet for j in alphabet]
    index = nb_col - 26
    if index <= 26**2:
        return alphabet + double[: index]

    # return list()
    return _sub_ABC(nb_col, alphabet, double)


# RuntimeWarning:
# overflow encountered in long_scalars
# import decimal

def np_prod(x):
    y = 1
    for i in x:
        y *= i
    return y


def np_sum(x):
    y = 0
    for i in x:
        y += i
    return y


def _get_tmp_name_ens(name_ens):
    tmp = name_ens[: 3]
    if name_ens == "AdaBoostM1":  # "AoM", "ABM"
        tmp = tmp[: 1] + name_ens[-2:]
    return tmp


def _get_tmp_document(name_ens, nb_cls):
    nmens_tmp = _get_tmp_name_ens(name_ens)
    return nmens_tmp + str(nb_cls)


# -----------------------
# Synthetic data (simulator.py)
#
#   nb_lbl: number of labels/classes
#   nb_spl: number of instances
#   nb_ftr: number of features
#   nb_clf: number of classifiers


def synthetic_lbl(nb_lbl, nb_spl, prng=None):
    y_inst = np.repeat(range(nb_lbl), nb_spl / nb_lbl + 1)
    y_inst = y_inst.reshape(nb_lbl, -1).T.reshape(-1)
    y_inst = y_inst[: nb_spl]
    if not prng:
        np.random.shuffle(y_inst)
    else:
        prng.shuffle(y_inst)
    return y_inst.tolist()


def synthetic_dat(nb_lbl, nb_spl, nb_ftr, prng=None):
    if not prng:
        X_inst = np.random.rand(nb_spl, nb_ftr)
        y_inst = np.random.randint(nb_lbl, size=nb_spl)
    else:
        X_inst = prng.rand(nb_spl, nb_ftr)
        y_inst = prng.randint(nb_lbl, size=nb_spl)
    return X_inst.tolist(), y_inst.tolist()


def synthetic_set(nb_lbl, nb_spl, nb_clf, prng=None):
    if not prng:
        prng = np.random
    y_inst = prng.randint(nb_lbl, size=nb_spl)
    yt_cls = prng.randint(nb_lbl, size=(nb_clf, nb_spl))
    coef = prng.rand(nb_clf)
    coef /= check_zero(np.sum(coef))
    return y_inst.tolist(), yt_cls.tolist(), coef.tolist()


def synthetic_clf(y_inst, nb_clf, err=.1, prng=None):
    if not prng:
        prng = np.random
    nb_spl, nb_lbl = len(y_inst), len(set(y_inst))
    yt_clf = np.repeat(y_inst, repeats=nb_clf, axis=0)
    yt_clf = yt_clf.reshape(-1, nb_clf).T
    num = int(nb_spl * err)
    for k in range(nb_clf):
        for _ in range(num):
            i = prng.randint(nb_spl)
            yt_clf[k][i] = nb_lbl - 1 - yt_clf[k][i]
    return yt_clf.tolist()


def random_seed_generator(psed='fixed_tseed'):  # _tim
    if (psed is not None) or (not isinstance(psed, int)):
        import time
        psed = int(time.time() * GAP_MID % GAP_INF)
    prng = np.random.RandomState(seed=psed)
    return psed, prng


# -------------------------------------
