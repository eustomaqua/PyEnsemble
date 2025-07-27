# coding: utf-8
#
# Target:
#   Combination after ensemble classification
#


from copy import deepcopy
import gc

import numpy as np
from pympler.asizeof import asizeof

gc.enable()


# -------------------------------------
#  Ensemble Voting
# -------------------------------------


# def plurality_voting(y, yt):
#     # vY = np.c_[y.reshape(-1, 1), yt.T].T
#     # vY = np.concatenate(([y], yt))
#     vY = np.unique(y).tolist() + np.unique(yt).tolist()
#     vY = np.unique(vY)

def plurality_voting(yt):
    vY = np.unique(yt).tolist()

    vote = [np.sum(np.equal(
        yt, i), axis=0).tolist() for i in vY]
    loca = np.argmax(vote, axis=0)  # vote.argmax(axis=0)
    fens = [vY[i] for i in loca]
    del vY, vote, loca
    gc.collect()
    return deepcopy(fens)  # fens.copy()


# def majority_voting(y, yt):
#     vY = np.unique(np.vstack([y, yt]))
#
def majority_voting(yt):
    vY = np.unique(yt).tolist()
    vote = [np.sum(np.equal(
        yt, i), axis=0).tolist() for i in vY]

    nb_cls = len(yt)
    half = int(np.ceil(nb_cls / 2.))
    vts = np.array(vote).T  # transpose

    loca = [np.where(j > half)[0] for j in vts]  # strictly
    # loca = [np.where(j >= half)[0] for j in vts]
    loca = [j[0] if len(j) > 0 else -1 for j in loca]
    fens = [vY[i] if i != -1 else -1 for i in loca]

    del vY, vote, half, vts, loca, nb_cls
    gc.collect()
    return deepcopy(fens)


# def weighted_voting(y, yt, coef):
#     vY = np.unique(np.concatenate([[y], yt]))
#
def weighted_voting(yt, coef):
    vY = np.unique(yt).tolist()
    # coef = np.array(np.mat(coef).T)
    # coef = np.array(coef).reshape(-1, 1)
    coef = np.array([coef]).transpose()

    weig = [np.sum(coef * np.equal(
        yt, i), axis=0).tolist() for i in vY]
    loca = np.array(weig).argmax(axis=0).tolist()
    fens = [vY[i] for i in loca]

    del vY, coef, weig, loca
    gc.collect()
    return deepcopy(fens)


# -------------------------------------
#  Ensemble Methods
# -------------------------------------


def get_accuracy_of_multiclass(y, yt, coef=None):
    # that is, get_accuracy_of_multiclassification()
    if not coef:
        nb_cls = len(yt)
        coef = [1. / nb_cls for _ in range(nb_cls)]
    # plurality_voting
    fens = weighted_voting(yt, coef)  # y,

    accpl = [np.mean(np.equal(t, y)) for t in yt]
    accsg = np.mean(np.equal(fens, y))

    accpr = np.mean(accsg > np.array(accpl))
    return deepcopy(fens), deepcopy(accpl), accsg, accpr


# -------------------------------------
# 1st_diversity, a tie
# -------------------------------------


# def tie_with_weight_plurality(y, yt, coef=None, nc=1):
#     if nc == 2:
#         y = [2 * i - 1 for i in y]
#         yt = [[2 * i - 1 for i in fx] for fx in yt]
#     if nc <= 2:
#         # if coef is None:
#         #     nb_cls = len(yt)
#         #     coef = [1. / nb_cls] * nb_cls
#         if coef is None:
#             fens = np.sum(yt, axis=0).tolist()
#             # fens = np.mean(yt, axis=0).tolist()
#         else:
#             weig = np.array([coef]).T
#             yt = np.array(yt)
#             fens = np.sum(weig * yt, axis=0)
#         ans = np.sign(fens)
#         if nc == 2:
#             ans = (ans + 1.) / 2.
#         return ans.tolist()
#     #   #   #
#     if coef is None:
#         fens = plurality_voting(y, yt)
#     else:
#         fens = weighted_voting(y, yt, coef)
#     return fens


def tie_with_weight_plurality(yt, coef=None, nc=1):
    if nc > 2:
        fens = plurality_voting(yt) if (
            coef is None) else weighted_voting(yt, coef)
        return fens
    if nc == 2:
        # y = [2 * i - 1 for i in y]
        yt = [[2 * i - 1 for i in fx] for fx in yt]
    if coef is None:
        fens = np.sum(yt, axis=0).tolist()  # .mean()
    else:
        weig = np.array([coef]).T
        yt = np.array(yt)
        fens = np.sum(weig * yt, axis=0)
    ans = np.sign(fens)
    if nc == 2:
        ans = (ans + 1.) / 2.
    return ans.tolist()


# ----------------------------------------
# obtain performance
# ----------------------------------------


def get_pruned_subensemble(y, yt, coef, P):
    ys = np.array(yt)[P].tolist()  # yt[P]
    cs = np.array(coef)[P].tolist()  # coef[P]
    _, accpl, accsg, accpr = get_accuracy_of_multiclass(y, ys, cs)
    return deepcopy(accpl), accsg, accpr


def get_pruned_space_cost(coef, clfs, P):
    opt_coef = np.array(coef)[P].tolist()  # coef[P]
    opt_clfs = [v for k, v in zip(P, clfs) if k]
    space_cost__ = asizeof(opt_clfs) + asizeof(opt_coef)
    del opt_coef, opt_clfs
    # train: get us_b4
    # prune: get us_af
    return space_cost__


def get_accuracy_for_pruned_version(y, yt, coef, clfs, P):
    ys = np.array(yt)[P].tolist()
    cs = np.array(coef)[P].tolist()
    _, accpl, accsg, accpr = get_accuracy_of_multiclass(y, ys, cs)

    opt_coef = cs  # np.array(coef)[P].tolist()
    opt_clfs = [clfs[k] for k, v in enumerate(P) if v]

    space_cost__ = asizeof(opt_clfs) + asizeof(opt_coef)
    del opt_coef, opt_clfs, ys, cs
    return deepcopy(accpl), accsg, accpr, space_cost__
