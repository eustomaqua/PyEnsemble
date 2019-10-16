# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from typing import List


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
