# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Diversity Measures
#-------------------------------------
#
# hi, hj:   list, not np.ndarray, with shape [m,]
# y \in {0, 1} -> {-1, +1}
# yt:       list, [[m,] nb_cls]
#
# m = nb_y
#


PAIRWISE = [
    'Disagreement',  # Disagreement Measure [Skalak, 1996, Ho, 1998]
    'Q_statistic',   # Q-Statistic [Yule, 1900]
    'Correlation',   # Correlation Coefficient [Sneath and Sokal, 1973]
    'K_statistic',   # Kappa-Statistic [Cohen, 1960]
    'Double_fault',  # Double-Fault Measure [Giacinto and Roli, 2001]
]

NONPAIRWISE = [
    'KWVariance',    # Kohavi-Wolpert Variance [Kohavi and Wolpert, 1996]
    #                #                         [Kuncheva and Whitaker, 2003]
    'Interrater',    # Interrater agreement [Fleiss, 1981]
    #                #                      [Kuncheva and Whitaker, 2003]
    'EntropyCC',     # Entropy, $Ent_{cc}$, [Cunningham and Carney, 2000]
    'EntropySK',     # Entropy, $Ent_{sk}$, [Shipp and Kuncheva, 2002]
    'Difficulty',    # Difficulty, [Hansen and Salamon, 1990]
    #                #             [Kuncheva and Whitaker, 2003]
    'Generalized',   # Generalized Diversity [Partridge and Krzanowski, 1997]
    'CoinFailure',   # Coincident Failure [Partridge and Krzanowski, 1997]
]


AVAILABLE_NAME_DIVER = PAIRWISE + NONPAIRWISE



__all__ = ['PAIRWISE', 'NONPAIRWISE',
           'AVAILABLE_NAME_DIVER']
from . import pairwise
from . import nonpairwise
from . import overall
from . import utils_diver
__all__.extend(['pairwise', 'nonpairwise',
                'overall', 'utils_diver'])

