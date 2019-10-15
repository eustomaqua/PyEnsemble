# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pyensemble.utils_const import DTY_INT


from pyensemble.diversity.pairwise import Disagreement_Measure_multiclass
from pyensemble.diversity.pairwise import Q_Statistic_binary
from pyensemble.diversity.pairwise import Correlation_Coefficient_binary
from pyensemble.diversity.pairwise import Kappa_Statistic_multiclass
from pyensemble.diversity.pairwise import Double_Fault_Measure_multiclass

from pyensemble.diversity.nonpairwise import Kohavi_Wolpert_Variance_multiclass
from pyensemble.diversity.nonpairwise import Interrater_agreement_multiclass
from pyensemble.diversity.nonpairwise import Entropy_cc_multiclass
from pyensemble.diversity.nonpairwise import Entropy_sk_multiclass
from pyensemble.diversity.nonpairwise import Difficulty_multiclass
from pyensemble.diversity.nonpairwise import Generalized_Diversity_multiclass
from pyensemble.diversity.nonpairwise import Coincident_Failure_multiclass



#-------------------------------------------
# Overall interface
#-------------------------------------------



# Pairwise Measure

def pairwise_measure_for_item(name_div, hi, hj, y, m):
    if name_div in ['Q_statistic', 'Correlation']:
        ya = np.array(hi) == np.array(y)
        yb = np.array(hj) == np.array(y)
        ya = np.array(ya, dtype=DTY_INT)
        yb = np.array(yb, dtype=DTY_INT)
    if name_div in ['Q_statistic', 'Correlation']:
        vY = np.unique(np.concatenate([y, hi, hj]))
        if len(vY) > 2:
            hi = ya.tolist();   hj = yb.tolist()
    #   #
    if name_div == 'Disagreement':
        ans = Disagreement_Measure_multiclass(hi, hj, m)
    elif name_div == 'Q_statistic':
        ans = Q_Statistic_binary(hi, hj)
        # ans = Q_Statistic_binary(ya, yb)
    elif name_div == 'Correlation':
        ans = Correlation_Coefficient_binary(hi, hj)
        # ans = Correlation_Coefficient_binary(ya, yb)
    elif name_div == 'K_statistic':
        ans = Kappa_Statistic_multiclass(hi, hj, y, m)
        ans = ans[0]
    elif name_div == 'Double_fault':
        ans = Double_Fault_Measure_multiclass(hi, hj, y, m)
    else:
        raise UserWarning("LookupError! Check the `name_diver` for pairwise_measure.")
    #   #
    return ans


def pairwise_measure_overall_value(name_div, yt, y, m, nb_cls):
    ans = 0.
    for i in range(nb_cls - 1):
        for j in range(i + 1, nb_cls):
            tem = pairwise_measure_for_item(name_div, yt[i], yt[j], y, m)
            ans += tem
    return ans * 2. / (nb_cls * (nb_cls - 1.))



# Non-Pairwise Measure

def nonpairwise_measure_overall_value(name_div, yt, y, m, nb_cls):
    if name_div == 'KWVariance':
        ans = Kohavi_Wolpert_Variance_multiclass(yt, y, m, nb_cls)
    elif name_div == 'Interrater':
        ans = Interrater_agreement_multiclass(yt, y, m, nb_cls)
    elif name_div == 'EntropyCC':
        ans = Entropy_cc_multiclass(yt, y, m, nb_cls)
    elif name_div == 'EntropySK':
        ans = Entropy_sk_multiclass(yt, y, m, nb_cls)
    elif name_div == 'Difficulty':
        ans = Difficulty_multiclass(yt, y, nb_cls)
    elif name_div == 'Generalized':
        ans = Generalized_Diversity_multiclass(yt, y, m, nb_cls)
    elif name_div == 'CoinFailure':
        ans = Coincident_Failure_multiclass(yt, y, m, nb_cls)
    else:
        raise UserWarning("LookupError! Check the `name_diver` for non-pairwise measure.")
    return ans


def nonpairwise_measure_for_item(name_div, hi, hj, y, m):
    yt = [hi, hj];  nb_cls = 2
    return nonpairwise_measure_overall_value(name_div, yt, y, m, nb_cls)



# Overall

def calculate_overall_diversity(name_div, yt, y, m, nb_cls):
    if name_div in ['Disagreement', 'Q_statistic', 'Correlation', 'K_statistic', 'Double_fault']:
        return pairwise_measure_overall_value(name_div, yt, y, m, nb_cls)
    if name_div in ['KWVariance', 'Interrater', 'EntropyCC', 'EntropySK', 'Difficulty', 'Generalized', 'CoinFailure']:
        return nonpairwise_measure_overall_value(name_div, yt, y, m, nb_cls)
    raise UserWarning("LookupError! Check the `name_diver`.")


def calculate_item_in_diversity(name_div, hi, hj, y, m):
    if name_div in ['Disagreement', 'Q_statistic', 'Correlation', 'K_statistic', 'Double_fault']:
        return pairwise_measure_for_item(name_div, hi, hj, y, m)
    if name_div in ['KWVariance', 'Interrater', 'EntropyCC', 'EntropySK', 'Difficulty', 'Generalized', 'CoinFailure']:
        return nonpairwise_measure_for_item(name_div, hi, hj, y, m)
    raise UserWarning("LookupError! Check the `name_diver`.")



