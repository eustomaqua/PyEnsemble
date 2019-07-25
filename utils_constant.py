# coding: utf8
# Aim to: constants for experiments (utils_const.py)


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
import time

import numpy as np 
# import multiprocessing as mp 
# from pathos import multiprocessing as pp
# from pympler.asizeof import asizeof
from PIL import Image as pil_image

gc.enable()




DTY_FLT = 'float32'
DTY_INT = 'int32'
DTY_BOL = 'bool'
CONST_ZERO = 1e-12


GAP_INF = 2 ** 31 - 1 
GAP_MID = 1e12 
GAP_NAN = 1e-12 


RAND_SEED = None



def check_zero(tem):
    return tem if tem != 0.0 else CONST_ZERO

