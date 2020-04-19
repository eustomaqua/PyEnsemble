# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from copy import deepcopy
import gc
import time

import numpy as np
from pathos import multiprocessing as pp

gc.enable()
from pyensemble.utils_const import GAP_INF, GAP_MID
from pyensemble.utils_const import DTY_FLT, DTY_INT
from pyensemble.utils_const import individual



#------------------------------------
#  Ensemble:  Bagging
#------------------------------------


def BaggingSelectTraining(X_trn, y_trn):
    X_trn = np.array(X_trn, dtype=DTY_FLT)
    y_trn = np.array(y_trn, dtype=DTY_INT)
    vY = np.unique(y_trn);  dY = len(vY)
    stack_X = [];   stack_y = []    # temporal

    randseed = int(time.time() * GAP_MID % GAP_INF)
    prng = np.random.RandomState(randseed)

    for k in range(dY):
        idx = (y_trn == vY[k])
        tem_X = X_trn[idx]
        tem_y = y_trn[idx]
        idx = prng.randint(0, len(tem_y), size=len(tem_y))
        tem_X = tem_X[idx].tolist()
        tem_y = tem_y[idx].tolist()
        stack_X.append( deepcopy(tem_X) )
        stack_y.append( deepcopy(tem_y) )
        del idx, tem_X,tem_y

    del X_trn,y_trn, vY,dY
    tem_X = np.concatenate(stack_X, axis=0)
    tem_y = np.concatenate(stack_y, axis=0)
    idx = list(range(len(tem_y)))
    prng.shuffle(idx)
    wX = tem_X[idx].tolist()
    wy = tem_y[idx].tolist()

    del tem_X, tem_y, idx, randseed, prng
    gc.collect()
    return deepcopy(wX), deepcopy(wy)  # list


def BaggingEnsembleAlgorithm(X_trn, y_trn, name_cls, nb_cls):
    clfs = []  # initial
    for k in range(nb_cls):
        wX, wy = BaggingSelectTraining(X_trn, y_trn)
        if len(np.unique(wy)) == 1:
            wX, wy = BaggingSelectTraining(X_trn, y_trn)
        clf = individual(name_cls, wX, wy)
        clfs.append( deepcopy(clf) )
        del wX, wy, clf
        gc.collect()
    coef = [1. / nb_cls] * nb_cls
    return deepcopy(coef), deepcopy(clfs)  # list



def BaggingEnsembleParallel(X_trn, y_trn, name_cls, nb_cls, cores):
    pool = pp.ProcessingPool(nodes = cores)
    wXy = pool.map(BaggingSelectTraining,  [X_trn]*nb_cls, [y_trn]*nb_cls)
    wX, wy = zip(*wXy)  # list, [[..] nb_cls]
    clfs = pool.map(individual,  [name_cls]*nb_cls, wX, wy)
    coef = [1./nb_cls] * nb_cls  # coef = np.array([1. / nb_cls] * nb_cls, dtype=DTY_FLT)
    del pool, wXy, wX, wy
    gc.collect()
    return deepcopy(coef), deepcopy(clfs)

