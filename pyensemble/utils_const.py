# coding: utf8
# Aim to:  constants for experiments


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


DTY_FLT = 'float32'
DTY_INT = 'int32'
DTY_BOL = 'bool'

CONST_ZERO = 1e-18

GAP_INF = 2 ** 32 - 1
GAP_MID = 1e24
GAP_NAN = 1e-12

RANDOM_SEED = None
FIXED_SEED = 4579


def check_zero(tem):
    return tem if (tem != 0.0) else CONST_ZERO

def check_equal(tem_A, tem_B):
    return True if (abs(tem_A - tem_B) <= 1e-6) else False


def individual(name_cls, wX, wy):
    return name_cls.fit(wX, wy)
# works for list and np.ndarray

