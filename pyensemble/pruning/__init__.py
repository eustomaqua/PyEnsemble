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
    'GMA',   # GMM_Algorithm
    'LCS',   # Local_Search
]  # DIVERSITY MAXIMIZATION
# modified by me to make them suitable for ensemble pruning problems


AVAILABLE_NAME_PRUNE = RANKING_BASED + OPTIMIZATION_BASED



__all__ = ['RANKING_BASED', 'CLUSTERING_BASED', 'OPTIMIZATION_BASED',
           'COMPOSABLE_CORE_SETS', 'AVAILABLE_NAME_PRUNE']
from . import ranking_based as ranking
from . import optimization_based as optimizing
from . import composable
# __all__.extend(['ranking_based', 'optimization_based', 'composable'])
__all__.extend(['ranking', 'optimizing', 'composable'])

from .ranking_based.Early_Stopping import Early_Stopping
from .ranking_based.Kappa_Pruning import Kappa_Pruning
from .ranking_based.KL_divergence_Pruning import KL_divergence_Pruning
from .ranking_based.Reduce_Error_Pruning import Reduce_Error_Pruning
from .ranking_based.Orientation_Ordering_Pruning import Orientation_Ordering_Pruning
from .ranking_based.OEP_inPEP import PEP_OEP as OEP_Pruning
__all__.extend(['Early_Stopping',
                'Kappa_Pruning',
                'KL_divergence_Pruning',
                'Reduce_Error_Pruning',
                'Orientation_Ordering_Pruning',
                'OEP_Pruning'])

from .optimization_based.DREP import DREP_Pruning
from .optimization_based.SEP_inPEP import PEP_SEP as SEP_Pruning
from .optimization_based.PEP_inPEP import PEP_PEP as PEP_Pruning
__all__.extend(['DREP_Pruning', 'SEP_Pruning', 'PEP_Pruning'])

from .composable.GMM_Algorithm import GMM_Algorithm
from .composable.Local_Search_Alg import Local_Search
__all__.extend(['GMM_Algorithm', 'Local_Search'])

