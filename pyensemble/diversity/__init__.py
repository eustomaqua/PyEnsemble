# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


PAIRWISE = [
    'Disagreement',
    'Q_statistic',
    'Correlation',
    'K_statistic',
    'Double_fault'
]

NONPAIRWISE = [
    'KWVariance',
    'Interrater',
    'EntropyCC',
    'EntropySK',
    'Difficulty',
    'Generalized',
    'CoinFailure'
]


AVAILABLE_NAME_DIVER = PAIRWISE + NONPAIRWISE
