# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from typing import List


# Ensemble Pruning
#-------------------------------------
#
# X_trn, y_trn, X_tst, y_tst
# nb_trn, nb_tst, nb_feat
# pr_feat, pr_pru
# k1,m1,lam1, k2,m2,lam2
#
# k?:   the number of selected objects (classifiers / features)
# m?:   the number of machiens doing ensemble pruning / feature selection
# \lambda:  tradeoff
#
# yt:   predicted results, list, [[nb_y] nb_cls]
# yo:   pruned results,    list, [[nb_y] nb_pru]
#


RANKING_BASED = [
    'ES',    # Early Stopping
    'KL',    # KL divergence Pruning
    'KP',    # Kappa Pruning
    'OO',    # Orientation Ordering Pruning
    'RE',    # Reduce Error Pruning
    'KL+',   # KL divergence Pruning (modified version of mine)
    'OEP',   # OEP in Pareto Ensemble Pruning
]  # ORDERING_BASED

CLUSTERING_BASED = []

OPTIMIZATION_BASED = [
    'DREP',  # DREP Pruning
    'SEP',   # SEP in Pareto Ensemble Pruning
    'PEP',   # PEP in Pareto Ensemble Pruning
    'PEP+',  # PEP (modified version of mine)
]


COMPOSABLE_CORE_SETS = [
    'GMM',   # GMM_Algorithm
    'LCS',   # Local_Search
]  # DIVERSITY MAXIMIZATION
# modified by me to make them suitable for ensemble pruning problems


AVAILABLE_NAME_PRUNE = RANKING_BASED + OPTIMIZATION_BASED
