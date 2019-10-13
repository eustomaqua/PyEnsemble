# coding: utf8
# Aim to:  constants for experiments


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


DTY_FLT = 'float32'
DTY_INT = 'int32'
DTY_BOL = 'bool'

CONST_ZERO = 1e-18

GAP_INF = 2 ** 31 - 1
GAP_MID = 1e12
GAP_NAN = 1e-12

RANDOM_SEED = None


def check_zero(tem):
    return tem if tem != 0.0 else CONST_ZERO

