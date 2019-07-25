# coding: utf8
# Aim to: classify data for experiments, (codes about ensembles)
#         data_ensemble.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
# import os
import sys
import time

import numpy as np 
# import multiprocessing as mp 
from pathos import multiprocessing as pp
from pympler.asizeof import asizeof


from utils_constant import DTY_FLT
from utils_constant import DTY_INT
gc.enable()

from utils_constant import GAP_INF
from utils_constant import GAP_MID
from utils_constant import GAP_NAN

from utils_constant import CONST_ZERO
from utils_constant import check_zero



#========================================
# data_classify.py
#========================================



from sklearn import tree            # DecisionTreeClassifier()
from sklearn import naive_bayes     # GaussianNB, MultinomialNB, BernoulliNB
from sklearn import svm             # SVC, NuSVC, LinearSVC
from sklearn import neighbors       # KNeighborsClassifier(n_neighbors, weights='uniform' or 'distance')
from sklearn import linear_model    # SGDClassifier(loss='hinge', penalty='l1' or 'l2')



#----------------------------------------
#
#----------------------------------------


# X_trn, X_tst:   list, [[nb_feat] nb_trn/tst]
# y_trn, y_tst:   list, [nb_trn/tst]
# 
# Y \in {0,1}
# y_insp:         list, [[nb_trn] nb_cls],  inspect
# y_pred:         list, [[nb_tst] nb_cls],  predict
# coefficient:            list, [nb_cls]
# weights (in resample):  list, [nb_y/X]



# def individual(args):
#     name_cls = args[0]
#     wX = args[1]
#     wy = args[2]
#     return name_cls.fit(wX, wy)


def individual(name_cls, wX, wy):
    return name_cls.fit(wX, wy)



#----------------------------------------
#
#----------------------------------------



#========================================
# Obtain ensemble, for binary classification
#========================================



#----------------------------------------
# Ensemble: Bagging
#----------------------------------------


def BaggingSelectTraining(X_trn, y_trn):
    X_trn = np.array(X_trn, dtype=DTY_FLT)
    y_trn = np.array(y_trn, dtype=DTY_INT)
    #
    vY = np.unique(y_trn);  dY = len(vY)
    stack_X = [];   stack_y = []    # temporal
    # 
    randseed = int(time.time() * GAP_MID % GAP_INF)
    prng = np.random.RandomState(randseed)
    #
    for k in range(dY):
        idx = (y_trn == vY[k])
        tem_X = X_trn[idx]
        tem_y = y_trn[idx]
        idx = prng.randint(0, len(tem_y), size=len(tem_y) )
        tem_X = tem_X[idx].tolist()
        tem_y = tem_y[idx].tolist()
        stack_X.append( deepcopy(tem_X) )
        stack_y.append( deepcopy(tem_y) )
        del idx, tem_X, tem_y
    #
    del X_trn, y_trn, vY, dY
    tem_X = np.concatenate(stack_X, axis=0)
    tem_y = np.concatenate(stack_y, axis=0)
    idx = list(range(len(tem_y)))
    prng.shuffle(idx)  # np.random
    wX = tem_X[idx].tolist()
    wy = tem_y[idx].tolist()
    #
    del tem_X, tem_y, idx, randseed, prng
    gc.collect()
    return deepcopy(wX), deepcopy(wy)


def BaggingEnsembleAlgorithm(X_trn, y_trn, name_cls, nb_cls):
    clfs = []  # initial
    for k in range(nb_cls):
        wX, wy = BaggingSelectTraining(X_trn, y_trn)
        if len(np.unique(wy)) == 1:
            wX, wy = BaggingSelectTraining(X_trn, y_trn)
        #
        clf = individual(name_cls, wX, wy)  # name_cls.fit(wX, wy)
        clfs.append( deepcopy(clf) )
        del wX, wy, clf
        gc.collect()
    coef = [1./nb_cls] * nb_cls
    return deepcopy(coef), deepcopy(clfs)



def BaggingEnsembleParallel(X_trn, y_trn, name_cls, nb_cls, cores):
    pool = pp.ProcessingPool(nodes = cores)
    wXy = pool.map(BaggingSelectTraining, [X_trn]*nb_cls, [y_trn]*nb_cls)
    wX, wy = zip(*wXy)  # list(zip(*wXy))
    clfs = pool.map(individual, [name_cls]*nb_cls, wX, wy)
    coef = [1./nb_cls] * nb_cls
    del pool, wXy, wX, wy
    gc.collect()
    return deepcopy(coef), deepcopy(clfs)



#----------------------------------------
# Ensemble: AdaBoost
#----------------------------------------


def resample(X, y, w):
    cw = np.cumsum(w).tolist()
    randseed = int(time.time() * GAP_MID % GAP_INF)
    prng = np.random.RandomState(randseed)
    value = prng.rand(len(y)).tolist()  # np.random
    idx = []
    for k in range(len(y)):
        if value[k] <= cw[0]:
            idx.append(0)
            continue
        for j in range(1, len(cw)):
            if value[k] > cw[j-1] and value[k] <= cw[j]:
                idx.append(j)
                break
    #   #   #   #
    if len(idx) == 0:
        idx.append( prng.randint(len(w)) )
    #   #
    X = np.array(X, dtype=DTY_FLT)
    y = np.array(y, dtype=DTY_INT)
    wX = X[idx].tolist()
    wy = y[idx].tolist()
    del cw, value, idx, X, y, randseed, prng
    gc.collect()
    return deepcopy(wX), deepcopy(wy)


def AdaBoostSelectTraining(X_trn, y_trn, weight):
    X_trn = np.array(X_trn, dtype=DTY_FLT)
    y_trn = np.array(y_trn, dtype=DTY_INT)  # * 2 - 1
    weight = np.array(weight, dtype=DTY_FLT)
    vY = np.unique(y_trn);  dY = len(vY)
    stack_X = [];   stack_y = []    # init
    #
    for k in range(dY):
        idx = (y_trn == vY[k])
        tem_X = X_trn[idx].tolist()
        tem_y = y_trn[idx].tolist()
        tem_w = weight[idx]
        # print("np.sum(tem_w)", np.sum(tem_w), tem_w)
        tem_w /= np.max([np.sum(tem_w), CONST_ZERO])  # GAP_NAN
        tem_w = tem_w.tolist()
        wX, wy = resample(tem_X, tem_y, tem_w)
        stack_X.append( deepcopy(wX) )
        stack_y.append( deepcopy(wy) )
        del idx, tem_X, tem_y, tem_w, wX, wy
    del X_trn, y_trn, weight, vY, dY
    #
    tem_X = np.concatenate(stack_X, axis=0)
    tem_y = np.concatenate(stack_y, axis=0)
    #
    randseed = int(time.time() * GAP_MID % GAP_INF)
    # np.random.RandomState(randseed)
    prng = np.random.RandomState(randseed)
    idx = list(range(len(tem_y)))
    prng.shuffle(idx)  # np.random
    wX = tem_X[idx].tolist()
    wy = tem_y[idx].tolist()
    del stack_X, stack_y, tem_X, tem_y, idx, randseed, prng
    gc.collect()
    return deepcopy(wX), deepcopy(wy)


def AdaBoostEnsembleAlgorithm(X_trn, y_trn, name_cls, nb_cls):
    # Y \in {0, 1}  # translate: y_trn = [i*2-1 for i in y_trn]
    #
    # Notice alpha here is relevant to this algorithm named AdaBoost. 
    clfs = [];  nb_trn = len(y_trn)
    # initial
    weight = np.zeros((nb_cls, nb_trn))
    em = [0.0] * nb_cls;    alpha = [0.0] * nb_cls
    #
    weight[0] = np.ones(nb_trn, dtype=DTY_FLT) / nb_trn
    for k in range(nb_cls):
        nb_count = 20
        while nb_count >= 0:
            # resample data: route wheel bat
            wX, wy = AdaBoostSelectTraining(X_trn, y_trn, weight[k].tolist() )
            # train a base classifier and run it on ORIGINAL training
            clf = individual(name_cls, wX, wy)  # name_cls.fit(wX, wy)
            inspect = clf.predict(X_trn)
            # calculate the error rate
            i_tr = (inspect != np.array(y_trn))
            em[k] = np.sum(weight[k] * i_tr)
            if em[k] >= 0. and em[k] < 0.5:  # em[ak] < 0.5:
                break
            nb_count -= 1
            del wX, wy
        # no more than 21 times
        del nb_count
        # 
        clfs.append( deepcopy(clf) )
        # calculate alpha
        # alpha[k] = 0.5 * np.log2((1.-em[k]) / np.max([em[k], GAP_NAN]) + GAP_NAN)  # 1e-16 # log2, without error "divide zero"
        alpha[k] = (1. - em[k]) / np.max([em[k], CONST_ZERO])
        alpha[k] = 0.5 * np.log2(np.max([alpha[k], CONST_ZERO]))
        # update weights.  Notice that: y \in {-1, +1} here, transform from {0,1}
        i_tr = (np.array(y_trn) * 2 - 1) * (inspect * 2 - 1)
        if k + 1 < nb_cls:
            weight[k + 1] = weight[k] * np.exp(-1. * alpha[k] * i_tr)
            zm = np.sum(weight[k + 1])
            weight[k + 1] /= np.max([zm, CONST_ZERO]) 
    #   #   #
    # regularization: alpha, sigma (coef)=1. 
    am = np.sum(alpha)
    alpha = [i/am for i in alpha]
    # 
    del weight, em,  clf,  i_tr,  zm, am
    gc.collect()
    return deepcopy(alpha), deepcopy(clfs)



#----------------------------------------
#
#----------------------------------------



#========================================
# Obtain ensemble, for binary classification
#========================================


# for incremental/online learning
#
# from sklearn import linear_model   # Perceptron, SGDClassifier, PassiveAggressiveClassifier
# from sklearn import neural_network     # MLPClassifier
#
'''
Classification
    sklearn.naive_bayes.MultinomialNB
    sklearn.naive_bayes.BernoulliNB
    sklearn.linear_model.Perceptron
    sklearn.linear_model.SGDClassifier
    sklearn.linear_model.PassiveAggressiveClassifier
    sklearn.neural_network.MLPClassifier
Regression
    sklearn.linear_model.SGDRegressor
    sklearn.linear_model.PassiveAggressiveRegressor
    sklearn.neural_network.MLPRegressor
Clustering
    sklearn.cluster.MiniBatchKMeans
    sklearn.cluster.Birch
Decomposition / feature Extraction
    sklearn.decomposition.MiniBatchDictionaryLearning
    sklearn.decomposition.IncrementalPCA
    sklearn.decomposition.LatentDirichletAllocation
Preprocessing
    sklearn.preprocessing.StandardScaler
    sklearn.preprocessing.MinMaxScaler
    sklearn.preprocessing.MaxAbsScaler
'''



#----------------------------------------
#
#----------------------------------------



#----------------------------------------
# Acquisition of ``coef, clfs''
#----------------------------------------


# def EnsembleAlgorithm(name_ensem, name_cls, nb_cls, X_trn, y_trn):
#     pass

def EnsembleAlgorithm(name_ens, name_cls, nb_cls, X_trn, y_trn):
    #@ params:  name_alg, name_cls, nb_cls, X_trn, y_trn
    #
    if name_ens == "BagPara":
        coef, clfs = BaggingEnsembleParallel(X_trn, y_trn, name_cls, nb_cls, 4)
    elif name_ens == 'Bagging':
        coef, clfs = BaggingEnsembleAlgorithm(X_trn, y_trn, name_cls, nb_cls)
    elif name_ens == 'AdaBoost':
        coef, clfs = AdaBoostEnsembleAlgorithm(X_trn, y_trn, name_cls, nb_cls)
    else:
        raise UserWarning("LookupError! Check the `name_ens` in EnsembleAlgorithm.")
    #
    # coef, clfs:   shape, (1, nb_cls)
    return deepcopy(coef), deepcopy(clfs)



# calculate accuracies of base/weak/individual classifiers
#
def CalcBaseAccuracies(clfs, X, y):
    # options:  CalcBaseAccuracies(clfs, X_tst, y_tst) or (.., X_trn, y_trn)
    #
    # yt = [clf.predict(X).tolist() for clf in clfs]
    # accpl = [np.mean(np.array(t) != np.array(y)) for t in yt]
    #
    y = np.array(y)
    yt = [clf.predict(X) for clf in clfs]
    accpl = [np.mean(t == y) for t in yt]
    del yt, y
    gc.collect()
    return deepcopy(accpl)


# Y \in {0, 1}
# 
def CalcPerformAccuracy(coef, clfs, X, y):
    # options:  CalcPerformAccuracy(coef, clfs, X_tst, y_tst) or (.., X_trn, y_trn)
    # 
    coef = np.array(np.mat(coef).T)
    yt = [clf.predict(X).tolist() for clf in clfs]
    y = np.array(y) * 2 - 1
    yt = np.array(yt) * 2 - 1
    fcode = np.sum(yt * coef, axis=0)
    fcode = np.sign(fcode)
    #
    tie = list(map(np.sum, [fcode==0, fcode==1, fcode==-1]))
    if tie[1] >= tie[2]:
        fcode[fcode == 0] = 1
    else:
        fcode[fcode == 0] = -1
    # endif there is a tie
    del tie
    # 
    accsg = np.mean(fcode == y)  # singular
    del coef, yt, y, fcode
    gc.collect()
    return accsg



#----------------------------------------
# Acquisition of ``y_insp, y_pref''
#----------------------------------------


# Y \in {0, 1}
# 
def calc_acc_sing_pl_and_pr(y, yt, coef):
    # options:  y_trn, y_insp, coef
    #           y_tst, y_pred, coef
    #
    accpl = [np.mean(np.array(t) == np.array(y)) for t in yt]
    #
    coef = np.array(np.mat(coef).T)
    y = np.array(y) * 2 - 1
    yt = np.array(yt) * 2 - 1
    fcode = np.sum(yt * coef, axis=0)
    fcode = np.sign(fcode)
    # 
    tie = list(map(np.sum, [fcode==0, fcode==1, fcode==-1]))
    if tie[1] >= tie[2]:
        fcode[fcode == 0] = 1
    else:
        fcode[fcode == 0] = -1
    del tie
    # endif there is a tie
    # 
    accsg = np.mean(fcode == y)  # singular
    del coef, yt, y, fcode
    # 
    accpr = np.mean(accsg >= np.array(accpl))
    gc.collect()
    return deepcopy(accpl), accsg, accpr


def original_ensemble_from_train_set(name_ens, name_cls, nb_cls, X_trn, y_trn, X_tst):
    #@ params:  name_alg/name_ensem, ...
    #
    if name_ens == "Bagging":
        coef, clfs = BaggingEnsembleAlgorithm(X_trn, y_trn, name_cls, nb_cls)
    elif name_ens == "AdaBoost":
        coef, clfs = AdaBoostEnsembleAlgorithm(X_trn, y_trn, name_cls, nb_cls)
    else:
        raise UserWarning("LookupError! Check the `name_ensem`.")
    # coef, clfs:   shape [nb_cls,]
    #
    y_insp = [t.predict(X_trn).tolist() for t in clfs]
    y_pred = [t.predict(X_tst).tolist() for t in clfs]
    # y_insp/pred .shape = [[nb_trn/tst] nb_cls]
    #
    return deepcopy(y_insp), deepcopy(y_pred), deepcopy(coef), deepcopy(clfs)



#----------------------------------------
#
#----------------------------------------



#========================================
# data_prune.py
#========================================



#----------------------------------------
# initial
#----------------------------------------
#
# \citep{martinez2009analysis}, Reduce-Error Pruning
# \citep{tsoumakas2009ensemble},



#----------------------------------------
# EnsembleAlgorithm
#----------------------------------------



#----------------------------------------
# EnsembleVoting
#----------------------------------------


def plurality_voting(y, yt):
    vY = np.unique(np.unique(y).tolist() + np.unique(yt).tolist())
    dY = len(vY)
    y = np.array(y);    yt = np.array(yt)
    vote = np.array([np.sum(yt == vY[i], axis=0).tolist() for i in range(dY)])
    loca = vote.argmax(axis=0)
    fens = [vY[i] for i in loca]
    del vY,dY, y,yt, vote,loca  # location
    gc.collect()
    return deepcopy(fens)


def majority_voting(y, yt):
    vY = np.unique(np.unique(y).tolist() + np.unique(yt).tolist())
    # vY = np.unique(np.vstack((y, yt)))
    dY = len(vY)
    y = np.array(y);    yt = np.array(yt)
    vote = [np.sum(yt == vY[i], axis=0).tolist() for i in range(dY)]
    #
    nb_cls = len(yt)
    half = int(np.ceil(nb_cls / 2.))
    vts = np.array(vote).T  # transpose
    #
    loca = [np.where(j >= half)[0][0] if len(np.where(j >= half)[0]) > 0 else -1 for j in vts]
    fens = [vY[i] if i != -1 else -1 for i in loca]
    #
    del vY,dY, y,yt, vote,half,vts,loca, nb_cls
    gc.collect()
    return deepcopy(fens)



# def weighted_majority_vote(y, yt, coef):
# def weighted_plurality_vote(y, yt, coef):
# 
def weighted_voting(y, yt, coef):
    # vY = np.unique(np.unique(y).tolist() + np.unique(yt).tolist())
    # vY = np.unique(np.vstack((y, yt)))
    vY = np.unique(np.concatenate([[y], yt]))
    dY = len(vY)
    y = np.array(y);    yt = np.array(yt)
    coef = np.array(np.mat(coef).T)
    weig = [np.sum(coef * (yt == vY[i]), axis=0).tolist() for i in range(dY)]
    loca = np.array(weig).argmax(axis=0)
    fens = [vY[i] for i in loca]
    del vY,dY, y,yt,coef, weig,loca  # location
    gc.collect()
    return deepcopy(fens)



#----------------------------------------
#
#----------------------------------------




#========================================
# 
#========================================
