# coding: utf-8
#
# Target:
#   Existing diversity measures in ensemble learning
#


from copy import deepcopy
import numpy as np

from pyfair.facil.utils_const import (
    check_zero, judge_transform_need, DTY_INT)
from pyfair.facil.utils_remark import (
    PAIRWISE, NONPAIRWISE, AVAILABLE_NAME_DIVER)

from pyfair.marble.diver_pairwise import (
    pairwise_measure_gather_multiclass,
    pairwise_measure_item_multiclass,
    pairwise_measure_whole_binary, pairwise_measure_whole_multi)
from pyfair.marble.diver_nonpairwise import (
    nonpairwise_measure_gather_multiclass,
    nonpairwise_measure_item_multiclass)

# from fairml.widget.utils_const import (
#     check_zero, DTY_FLT, DTY_INT, judge_transform_need)


# ==================================
#  General
# ==================================


# ----------------------------------
# Data Set
# ----------------------------------
#
# Instance  :  \mathcal{Y} \in \{c_1,...,c_{\ell}\} = {0,1,...,n_c-1}
# m (const) :  number of instances
# n (const) :  number of individual classifiers
# n_c (const:  number of classes / labels
#
# Data Set  :  \mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^m
# Classifier:  \mathcal{H} = \{h_j\}_{j=1}^n
#


# """
# def contingency_table_binary(hi, hj):
#     if not (len(hi) == len(hj)):  # number of instances/samples
#         raise AssertionError("These two individual classifiers have"
#                              " two different shapes.")
#     tem = np.concatenate([hi, hj])
#     _, dY = judge_transform_need(tem)
#     del tem
#     if dY > 2:
#         raise AssertionError("contingency_table only works for binary"
#                              " classification.")  # works for only.
#     elif dY == 2:
#         hi = [i * 2 - 1 for i in hi]
#         hj = [i * 2 - 1 for i in hj]
#     #   #   #
#     hi = np.array(hi, dtype=DTY_INT)
#     hj = np.array(hj, dtype=DTY_INT)
#     a = np.sum((hi == 1) & (hj == 1))
#     b = np.sum((hi == 1) & (hj == -1))
#     c = np.sum((hi == -1) & (hj == 1))
#     d = np.sum((hi == -1) & (hj == -1))
#     # return int(a), int(b), int(c), int(d)
#     return a, b, c, d
#
#
# def contingency_table_multi(hi, hj, y):
#     tem = np.concatenate([hi, hj, y])
#     # vY = np.unique(tem)
#     # dY = len(vY)
#     vY, dY = judge_transform_need(tem)
#     del tem
#     if dY == 1:
#         dY = 2
#     ha, hb = np.array(hi), np.array(hj)  # y=np.array(y)
#     # construct a contingency table
#     Cij = np.zeros(shape=(dY, dY), dtype=DTY_INT)
#     for i in range(dY):
#         for j in range(dY):
#             Cij[i, j] = np.sum((ha == vY[i]) & (hb == vY[j]))
#     #   #   #
#     # return Cij.tolist()  # list
#     return Cij.copy()  # Cij, np.ndarray
#
#
# def contingency_table_multiclass(ha, hb, y):
#     # construct a contingency table, Cij
#     a = np.sum(np.logical_and(np.equal(ha, y), np.equal(hb, y)))
#     c = np.sum(np.logical_and(np.equal(ha, y), np.not_equal(hb, y)))
#     b = np.sum(np.logical_and(np.not_equal(ha, y), np.equal(hb, y)))
#     d = np.sum(np.logical_and(np.not_equal(ha, y), np.not_equal(hb, y)))
#     # a,b,c,d are `np.integer` (not `int`), a/b/c/d.tolist() gets `int`
#     return int(a), int(b), int(c), int(d)
# """


# ==================================
#  General
# ==================================
#
# zhou2012ensemble     : binary (multi: self defined)
# kuncheva2003diversity: multiclass
#


# ----------------------------------
# General Overall


def contrastive_diversity_gather_multiclass(name_div, y, yt):
    m = len(y)  # number of instances
    nb_cls = len(yt)  # number of individuals
    assert name_div in AVAILABLE_NAME_DIVER
    if name_div in PAIRWISE.keys():
        return pairwise_measure_gather_multiclass(
            name_div, yt, y, m, nb_cls)
    elif name_div in NONPAIRWISE.keys():
        return nonpairwise_measure_gather_multiclass(
            name_div, yt, y, m, nb_cls)
    raise ValueError(
        "LookupError! Double check the `name_div` please.")


def contrastive_diversity_item_multiclass(name_div, y, ha, hb):
    m = len(y)  # if m is None else m
    # number of individual classifiers
    assert name_div in AVAILABLE_NAME_DIVER
    if name_div in PAIRWISE.keys():
        return pairwise_measure_item_multiclass(
            name_div, ha, hb, y, m)
    elif name_div in NONPAIRWISE.keys():
        return nonpairwise_measure_item_multiclass(
            name_div, ha, hb, y, m)
    raise ValueError(
        "LookupError! Double check the `name_div` please.")


def contrastive_diversity_by_instance_multiclass(name_div, y, yt):
    # nb_cls = len(yt)  # number of individual / weak classifiers
    nb_inst = len(y)  # =m, number of instances/samples in the data set
    answer = []
    for k in range(nb_inst):
        h = [y[k]]
        ht = [[fx[k]] for fx in yt]
        ans = contrastive_diversity_gather_multiclass(name_div, h, ht)
        answer.append(ans)
    return deepcopy(answer)


# ----------------------------------


def contrastive_diversity_whole_binary(name_div, y, yt):
    m, nb_cls = len(y), len(yt)
    if name_div in PAIRWISE.keys():
        return pairwise_measure_whole_binary(
            PAIRWISE[name_div], yt, y, m, nb_cls)
    elif name_div in NONPAIRWISE.keys():
        return nonpairwise_measure_gather_multiclass(
            name_div, yt, y, m, nb_cls)
    raise ValueError("Incorrect `name_div`.")


def contrastive_diversity_whole_multi(name_div, y, yt):
    m, nb_cls = len(y), len(yt)
    if name_div in PAIRWISE.keys():
        return pairwise_measure_whole_multi(
            PAIRWISE[name_div], yt, y, m, nb_cls)
    elif name_div in NONPAIRWISE.keys():
        return nonpairwise_measure_gather_multiclass(
            name_div, yt, y, m, nb_cls)
    raise ValueError("Incorrect `name_div`.")


# ----------------------------------


# '''
# def div_inst_item_cont_tab(ha, hb, vY, dY, change="mu"):
#     if (change == "bi") or (dY == 2):  # dY == 2
#         ha = ha * 2 - 1  # ha = [i * 2 - 1 for i in ha]
#         hb = hb * 2 - 1  # hb = [i * 2 - 1 for i in hb]
#     if (change in ["bi", "tr"]) or (dY in [1, 2]):
#         a = np.sum(np.equal(ha, 1) & np.equal(hb, 1))
#         b = np.sum(np.equal(ha, 1) & np.equal(hb, -1))
#         c = np.sum(np.equal(ha, -1) & np.equal(hb, 1))
#         d = np.sum(np.equal(ha, -1) & np.equal(hb, -1))
#         return a, b, c, d
#     elif (change == "mu") or (dY >= 3):
#         Cij = np.zeros(shape=(dY, dY), dtype=DTY_INT)
#         for i in range(dY):
#             for j in range(dY):
#                 Cij[i, j] = np.sum(
#                     np.equal(ha, vY[i]) & np.equal(hb, vY[j]))
#         return Cij.copy()
#     raise ValueError(
#         "Check `change`, it should belong to {tr,bi,mu}.")
# '''


def div_inst_item_cont_tab(ha, hb, vY, dY, change="mu"):
    assert change in ('tr', 'bi', 'mu'), "ValueError, check `change`."
    if (change == 'mu') or (dY >= 3):
        Cij = np.zeros(shape=(dY, dY), dtype=DTY_INT)  # 'int')
        for i in range(dY):
            for j in range(dY):
                Cij[i, j] = np.sum(
                    np.equal(ha, vY[i]) & np.equal(hb, vY[j]))
        return Cij  # Cij.copy()  # contg_tab_mu_type3(ha, hb, vY)
    if (change == "bi") or (dY == 2):  # dY == 2
        ha = ha * 2 - 1  # ha = [i * 2 - 1 for i in ha]
        hb = hb * 2 - 1  # hb = [i * 2 - 1 for i in hb]
    # if (change in ["bi", "tr"]) or (dY in [1, 2]):
    a = np.sum(np.equal(ha, 1) & np.equal(hb, 1))
    b = np.sum(np.equal(ha, 1) & np.equal(hb, -1))
    c = np.sum(np.equal(ha, -1) & np.equal(hb, 1))
    d = np.sum(np.equal(ha, -1) & np.equal(hb, -1))
    return a, b, c, d


def div_inst_item_pairwise(name_div, h, ha, hb, vY, dY):
    # if change in ["tr", "bi"]:
    change = "tr" if dY == 1 else "bi"
    a, b, c, d = div_inst_item_cont_tab(ha, hb, vY, dY, change)
    # if change != "mu":
    #   #   #
    if name_div == "QStat":
        return (a * d - b * c) / check_zero(a * d + b * c)
    elif name_div == "KStat":
        m = a + b + c + d
        Theta_1 = float(a + d) / m
        Theta_2 = (
            (a + b) * (a + c) + (c + d) * (b + d)) / float(m ** 2)
        return (Theta_1 - Theta_2) / check_zero(1. - Theta_2)
    elif name_div == "Disag":
        return float(b + c) / (a + b + c + d)
    elif name_div == "Corre":
        denominator = (a + b) * (a + c) * (c + d) * (b + d)
        denominator = np.sqrt(denominator)
        return (a * d - b * c) / check_zero(denominator)
    elif name_div == "DoubF":
        e = np.sum(np.not_equal(ha, h) & np.not_equal(hb, h))
        return float(e) / (a + b + c + d)
    #   #
    raise ValueError(
        "Check `name_div`, not a pairwise measure of diversity.")


# def contrastive_diversity_institem_mubi(name_div, y, ha, hb, change="mu"):
def div_inst_item_mubi(name_div, h, ha, hb, vY, dY):
    # elif change == "mu":
    Cij = div_inst_item_cont_tab(ha, hb, vY, dY, change="mu")
    m = 1  # m = len(h)

    # '''
    # if name_div == "QStat":
    #     axd = np.prod([Cij[i][i] for i in range(dY)])
    #     bxc = np.prod([Cij[i][dY - 1 - i] for i in range(dY)])
    #     return (axd - bxc) / check_zero(axd + bxc)
    # elif name_div == "KStat":
    #     Theta_1 = np.sum([Cij[i, i] for i in range(dY)]) / float(m)
    #     Theta_2 = np.sum(Cij, axis=1) * np.sum(Cij, axis=0)
    #     Theta_2 = np.sum(Theta_2) / float(m ** 2)
    #     return (Theta_1 - Theta_2) / check_zero(1. - Theta_2)
    # elif name_div == "Disag":
    #     # return np.sum(np.not_equal(ha, hb)) / float(a + b + c + d)
    #     return np.sum(np.not_equal(ha, hb)) / float(m)
    # elif name_div == "Corre":
    #     axd = np.prod([Cij[i, i] for i in range(dY)])
    #     bxc = np.prod([Cij[i, dY - 1 - i] for i in range(dY)])
    #     denominator = np.multiply(
    #         np.sum(Cij, axis=1), np.sum(Cij, axis=0))
    #     denominator = np.sqrt(np.prod(denominator))
    #     # denominator = np.sqrt(np_prod(denominator.tolist()))
    #     return (axd - bxc) / check_zero(denominator)
    # elif name_div == "DoubF":
    #     e = np.sum(np.not_equal(ha, h) & np.not_equal(hb, h))
    #     return float(e) / m
    # '''

    Cij_ixi = [Cij[i, i] for i in range(dY)]  # np.diag(Cij)
    Cij_minus = [Cij[i, dY - 1 - i] for i in range(dY)]
    if name_div == "QStat":
        axd = np.prod(Cij_ixi)
        bxc = np.prod(Cij_minus)
        return (axd - bxc) / check_zero(axd + bxc)
    elif name_div == "KStat":
        Theta_1 = np.sum(Cij_ixi) / float(m)
        Theta_2 = np.sum(Cij, axis=1) * np.sum(Cij, axis=0)
        Theta_2 = np.sum(Theta_2) / float(m ** 2)
        return (Theta_1 - Theta_2) / check_zero(1. - Theta_2)
    elif name_div == "Disag":
        return np.sum(np.not_equal(ha, hb)) / float(m)
    elif name_div == "Corre":
        axd = np.prod(Cij_ixi)
        bxc = np.prod(Cij_minus)
        denominator = np.multiply(
            np.sum(Cij, axis=1), np.sum(Cij, axis=0))
        denominator = np.sqrt(np.prod(denominator))
        return (axd - bxc) / check_zero(denominator)
    elif name_div == "DoubF":
        e = np.sum(np.not_equal(ha, h) & np.not_equal(hb, h))
        return float(e) / m

    # if name_div not in PAIRWISE:  # i.e., PAIRWISE.keys()
    raise ValueError(
        "Check `name_div`, it should be a pairwise measure.")


def contrastive_diversity_instance_mubi(name_div, y, yt):  # , change="mu"
    vY = np.concatenate(
        [[y], yt], axis=0).reshape(-1).tolist()
    vY, dY = judge_transform_need(vY)
    change = "mu" if dY >= 3 else ("bi" if dY == 2 else "tr")  # dY==1
    nb_inst, nb_cls, answer = len(y), len(yt), []
    for k in range(nb_inst):
        h = y[k]  # h = [y[k]]
        ht = [fx[k] for fx in yt]  # ht = [[fx[k]] for fx in yt]
        # if change == "mu":
        #     ans = contrastive_diversity_whole_multi(name_div, h, ht)
        # elif change in ["bi", "tr"]:
        #     ans = contrastive_diversity_whole_binary(name_div, h, ht)
        # else:
        #     raise ValueError("Check the `change` parameter please.")
        if name_div in PAIRWISE:
            res = 0.
            if change in ["tr", "bi"]:
                for ia in range(nb_cls - 1):
                    for ib in range(ia + 1, nb_cls):
                        res += div_inst_item_pairwise(
                            name_div, h, ht[ia], ht[ib], vY, dY)
            elif change in ["mu"]:
                for ia in range(nb_cls - 1):
                    for ib in range(ia + 1, nb_cls):
                        res += div_inst_item_mubi(
                            name_div, h, ht[ia], ht[ib], vY, dY)
            else:
                raise ValueError(
                    "Check `change`, it should belong to {tr,bi,mu}.")
            ans = res * 2. / check_zero(nb_cls * (nb_cls - 1.))
        elif name_div in NONPAIRWISE:
            # i.e.,  contrastive_diversity_whole_multi(name_div, h, ht)
            ht = [[fx] for fx in ht]
            ans = contrastive_diversity_gather_multiclass(
                name_div, [h], ht)
            # NOTICE: 'GeneD', 'CFail/CoinF' might have bug:
            #       ## ZeroDivisionError: float division by zero
        else:
            raise ValueError("Check `name_div`, pairwise/nonpairwise?")
        answer.append(ans)
    return deepcopy(answer)
