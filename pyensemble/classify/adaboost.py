# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from copy import deepcopy
import gc
import time

import numpy as np
gc.enable()

from pyensemble.utils_const import GAP_INF, GAP_MID
from pyensemble.utils_const import DTY_FLT, DTY_INT
from pyensemble.utils_const import individual, check_zero



#------------------------------------
#  Ensemble:  AdaBoost
#------------------------------------


def resample(X, y, w):
    cw = np.cumsum(w).tolist()
    randseed = int(time.time() * GAP_MID % GAP_INF)
    prng = np.random.RandomState(randseed)
    value = prng.rand(len(y)).tolist()
    idx = []
    for k in range(len(y)):
        if value[k] <= cw[0]:
            idx.append(0)
            continue
        for j in range(1, len(cw)):
            if value[k] > cw[j-1] and value[k] <= cw[j]:
                idx.append(j)
                break

    if len(idx) == 0:
        idx.append( prng.randint(len(w)) )

    X = np.array(X, dtype=DTY_FLT)
    y = np.array(y, dtype=DTY_INT)
    wX = X[idx].tolist()
    wy = y[idx].tolist()
    del cw, value, idx, X,y, randseed,prng
    gc.collect()
    return deepcopy(wX), deepcopy(wy)  # list


def AdaBoostSelectTraining(X_trn, y_trn, weight):
    X_trn = np.array(X_trn, dtype=DTY_FLT)
    y_trn = np.array(y_trn, dtype=DTY_INT)
    weight = np.array(weight, dtype=DTY_FLT)
    vY = np.unique(y_trn);  dY = len(vY)
    stack_X = [];   stack_y = []    # init

    for k in range(dY):
        idx = (y_trn == vY[k])
        tem_X = X_trn[idx].tolist()
        tem_y = y_trn[idx].tolist()
        tem_w = weight[idx]
        tem_w /= check_zero( np.sum(tem_w) )
        tem_w = tem_w.tolist()
        wX, wy = resample(tem_X, tem_y, tem_w)
        stack_X.append( deepcopy(wX) )
        stack_y.append( deepcopy(wy) )
        del idx, tem_X,tem_y,tem_w, wX,wy
    del X_trn,y_trn,weight, vY,dY

    tem_X = np.concatenate(stack_X, axis=0)
    tem_y = np.concatenate(stack_y, axis=0)

    randseed = int(time.time() * GAP_MID % GAP_INF)
    prng = np.random.RandomState(randseed)
    idx = list(range(len(tem_y)))
    prng.shuffle(idx)
    wX = tem_X[idx].tolist()
    wy = tem_y[idx].tolist()
    del stack_X,stack_y, tem_X,tem_y, idx, randseed,prng
    gc.collect()
    return deepcopy(wX), deepcopy(wy)  # list


def AdaBoostEnsembleAlgorithm(X_trn, y_trn, name_cls, nb_cls):
    # Y\in {0,1}  # translate: y_trn = [ i*2-1  for i in y_trn]

    # Notice alpha here is relevant to this algorithm named AdaBoost.
    clfs = [];  nb_trn = len(y_trn)
    # initial
    weight = np.zeros((nb_cls, nb_trn), dtype=DTY_FLT)
    em = [0.0] * nb_cls;    alpha = [0.0] * nb_cls

    weight[0] = np.ones(nb_trn, dtype=DTY_FLT) / nb_trn
    for k in range(nb_cls):
        nb_count = 20
        while nb_count >= 0:
            # resample data: route wheel bat
            wX, wy = AdaBoostSelectTraining(X_trn, y_trn, weight[k].tolist() )
            # train a base classifier and run it on ORIGINAL training
            clf = individual(name_cls, wX, wy)
            inspect = clf.predict(X_trn)
            # calculate the error rate
            i_tr = (inspect != np.array(y_trn))
            em[k] = np.sum(weight[k] * i_tr)
            if em[k] >= 0. and em[k] < 0.5:
                break
            nb_count -= 1
            del wX, wy
        del nb_count

        clfs.append( deepcopy(clf) )
        # calculate alpha
        alpha[k] = 0.5 * np.log2(check_zero((1. - em[k]) / check_zero(em[k])))
        # update weights.  Notice that: y \in {-1,+1} here, transform from {0,1}
        i_tr = (np.array(y_trn) *2-1) * (inspect *2-1)
        if k+1 < nb_cls:
            weight[k+1] = weight[k] * np.exp(-1. * alpha[k] * i_tr)
            zm = np.sum(weight[k+1])
            weight[k+1] /= check_zero(zm)

    # regularization: alpha, sigma(coef)=1.
    am = np.sum(alpha)
    alpha = [i/am for i in alpha]

    del weight,em, clf, i_tr, zm,am
    gc.collect()
    return deepcopy(alpha), deepcopy(clfs)

