# coding: utf-8
# Author: Yijun
#
# Target:
#   Classify data (Bagging and AdaBoost) for experiments
#


from copy import deepcopy
import gc
import time
import numpy as np
# from pathos import multiprocessing as pp
# from pyfairness.facil.pkgs_pympler import asizeof
from pympler.asizeof import asizeof

from pyfair.facil.utils_const import (
    check_zero, judge_transform_need, random_seed_generator,
    DTY_FLT, DTY_INT,)
from pyfair.facil.utils_remark import INDIVIDUALS

gc.enable()


# ============================================
#  Obtain ensemble, for binary classification
# ============================================
#
# from sklearn import tree         # DecisionTreeClassifier()
# from sklearn import naive_bayes  # GaussianNB, MultinomialNB, BernoulliNB
# from sklearn import svm          # SVC, NuSVC, LinearSVC
#
# X_trn, X_tst, X_val:  list, [[nb_feat] nb_?]
# y_trn, y_tst, y_val:  list, [nb_trn/tst/val]
# Y \in {0, 1}
# y_insp:       list, [[nb_trn] nb_cls], inspect
# y_pred:       list, [[nb_tst] nb_cls], predict
# y_cast:       list, [[nb_val] nb_cls], verdict/validate
# coefficient:              list, [nb_cls]
# weights (in resample):    list, [nb_y/X]
#


# def individual(args):
#     name_cls = args[0]
#     wX, wy = args[1:]
#     return name_cls.fit(wX, wy)
#
def individual(name_cls, wX, wy):
    return name_cls.fit(wX, wy)
# works for list and np.ndarray


# def renew_random_seed_generator():
#   rndsed = int(time.time() * GAP_MID % GAP_INF)
#   prng = np.random.RandomState(rndsed)
#   return rndsed, prng


# ----------------------------------
#  Ensemble:  Bagging
# ----------------------------------
# return: list
# works for multi-class


def _BaggingSelectTraining(X_trn, y_trn):
    X_trn = np.array(X_trn, dtype=DTY_FLT)
    y_trn = np.array(y_trn, dtype=DTY_INT)
    vY = np.unique(y_trn)
    dY = len(vY)
    stack_X, stack_y = [], []  # temporal
    idx_stack = []

    rndsed, prng = random_seed_generator()
    for k in range(dY):
        idx = (y_trn == vY[k])
        tem_X = X_trn[idx]
        tem_y = y_trn[idx]
        # idx = prng.randint(0, len(tem_y), size=len(tem_y))

        idx_tmp = np.where(idx)[0]
        idx = prng.randint(0, len(idx_tmp), size=len(idx_tmp))
        idx_tmp = idx_tmp[idx]

        tem_X = tem_X[idx].tolist()
        tem_y = tem_y[idx].tolist()
        stack_X.append(tem_X)  # deepcopy(tem_X))
        stack_y.append(tem_y)  # deepcopy(tem_y))

        idx_stack.append(idx_tmp)
        del idx, tem_X, tem_y
    del X_trn, y_trn, vY, dY

    tem_X = np.concatenate(stack_X, axis=0)
    tem_y = np.concatenate(stack_y, axis=0)
    idx_tmp = np.concatenate(idx_stack, axis=0)
    idx = list(range(len(tem_y)))
    prng.shuffle(idx)  # np.random
    wX = tem_X[idx].tolist()
    wy = tem_y[idx].tolist()
    idx_tmp = idx_tmp[idx].tolist()
    del tem_X, tem_y, rndsed, prng
    gc.collect()
    # return deepcopy(wX), deepcopy(wy), deepcopy(idx)
    return wX, wy, idx_tmp  # wX, wy, idx


def BaggingEnsembleAlgorithm(X_trn, y_trn, name_cls, nb_cls):
    clfs, indices = [], []   # initial
    for _ in range(nb_cls):  # _:k
        wX, wy, idx = _BaggingSelectTraining(X_trn, y_trn)
        # if len(np.unique(wy)) == 1:
        #     wX, wy, idx = BaggingSelectTraining(X_trn, y_trn)
        indices.append(idx)  # deepcopy(idx))
        clf = individual(name_cls, wX, wy)  # name_cls.fit(wX, wy)
        clfs.append(deepcopy(clf))
        del wX, wy, clf
        gc.collect()
    coef = [1. / nb_cls] * nb_cls  # np.array(coef, dtype=DTY_FLT)
    # return deepcopy(coef), deepcopy(clfs), deepcopy(indices)
    return coef, clfs, indices


# def BaggingEnsembleParallel(X_trn, y_trn, name_cls, nb_cls, cores):
#     pool = pp.ProcessingPool(nodes = cores)
#     wXy = pool.map(BaggingSelectTraining, [X_trn]*nb_cls, [y_trn]*nb_cls)
#     wX, wy, ti = zip(*wXy)  # list, [[..] nb_cls]
#     clfs = pool.map(individual, [name_cls]*nb_cls, wX, wy)
#     coef = [1./nb_cls] * nb_cls  # np.ones(nb_cls) / nb_cls
#     del pool, wXy, wX, wy
#     gc.collect()
#     return deepcopy(coef), deepcopy(clfs), deepcopy(ti)


# ----------------------------------
#  Ensemble:  AdaBoost
# ----------------------------------
# return: list
# works for multi-class


def _resample(X, y, w):
    # assert len(y) == len(w)
    cw = np.cumsum(w).tolist()
    rndsed, prng = random_seed_generator()

    idx, dY = [], len(y)
    value = prng.rand(dY).tolist()
    for k in range(dY):  # len(y)
        if value[k] <= cw[0]:
            idx.append(0)
            continue
        for j in range(1, dY):  # len(cw)
            if (value[k] > cw[j - 1]) and (value[k] <= cw[j]):
                idx.append(j)
                break
    #   #   #   #   #

    if len(idx) == 0:
        idx.append(prng.randint(dY))  # len(w)
    X = np.array(X, dtype=DTY_FLT)
    y = np.array(y, dtype=DTY_INT)
    wX = X[idx].tolist()
    wy = y[idx].tolist()
    del cw, value, X, y, rndsed, prng
    gc.collect()
    # return deepcopy(wX), deepcopy(wy), deepcopy(idx)
    return wX, wy, idx


def _AdaBoostSelectTraining(X_trn, y_trn, weight):
    X_trn = np.array(X_trn, dtype=DTY_FLT)
    y_trn = np.array(y_trn, dtype=DTY_INT)
    weight = np.array(weight, dtype=DTY_FLT)
    vY = np.unique(y_trn)
    dY = len(vY)
    stack_X, stack_y, stack_idx = [], [], []  # init

    for k in range(dY):
        idx = (y_trn == vY[k])
        tem_X = X_trn[idx].tolist()
        tem_y = y_trn[idx].tolist()
        tem_w = weight[idx]
        tem_w /= check_zero(np.sum(tem_w))
        tem_w = tem_w.tolist()
        wX, wy, tem_idx = _resample(tem_X, tem_y, tem_w)

        idx_tmp = np.where(idx)[0]
        idx_tmp = idx_tmp[tem_idx]
        stack_X.append(wX)  # deepcopy(wX))
        stack_y.append(wy)  # deepcopy(wy))
        # stack_idx.append(tem_idx)  # deepcopy(tem_idx))
        stack_idx.append(idx_tmp)
        del idx, tem_X, tem_y, tem_w, wX, wy
    del vY, dY

    tem_X = np.concatenate(stack_X, axis=0)
    tem_y = np.concatenate(stack_y, axis=0)
    tem_idx = np.concatenate(stack_idx, axis=0)

    # rndsed, prng = renew_random_seed_generator()
    rndsed, prng = random_seed_generator()
    # rndsed = renew_fixed_tseed()
    # prng = renew_random_seed(rndsed)

    idx = list(range(len(tem_y)))
    prng.shuffle(idx)
    wX = tem_X[idx].tolist()
    wy = tem_y[idx].tolist()

    tem_idx = tem_idx[idx].tolist()
    if len(wX) <= 2:
        sw = np.argsort(weight)[:: -1]
        sw = sw[: (3 - len(wX))]  # why is 3?
        for si in sw:
            wX.append(X_trn[si])
            wy.append(y_trn[si])
            tem_idx.append(int(si))
        del sw, si
    # end if for robustness
    del X_trn, y_trn, weight, rndsed, prng
    del stack_X, stack_y, stack_idx, tem_X, tem_y  # ,tem_idx
    gc.collect()
    # return deepcopy(wX), deepcopy(wy), deepcopy(idx)
    return wX, wy, tem_idx  # return wX, wy, idx


# Discarded:
#   DeprecationWarning: only works for mathcal{Y} \in \{0,1\}
#
def AdaBoostEnsembleAlgorithm(X_trn, y_trn, name_cls, nb_cls):
    # Y\in {0,1}  # y_trn=[i*2-1 for i in y_trn]  # translation needed
    # Notice alpha here is relevant to this algorithm named AdaBoost.
    clfs, nb_trn = [], len(y_trn)  # initial
    weight = np.zeros((nb_cls, nb_trn), dtype=DTY_FLT)  # 'float')
    em = [0.0] * nb_cls
    alpha = [0.0] * nb_cls
    indices = []

    weight[0] = np.ones(nb_trn, dtype=DTY_FLT) / nb_trn
    for k in range(nb_cls):
        nb_count = 20
        while nb_count >= 0:
            # resample data: route wheel bat
            wX, wy, idx = _AdaBoostSelectTraining(
                X_trn, y_trn, weight[k].tolist())
            # train a base classifier and run it on ORIGINAL training
            clf = individual(name_cls, wX, wy)  # name_cls.fit(wX, wy)
            inspect = clf.predict(X_trn)
            # calculate the error rate
            i_tr = (inspect != np.array(y_trn))  # i_tr = (inspect != y_trn)
            em[k] = np.sum(weight[k] * i_tr)
            if (em[k] >= 0.) and (em[k] < 0.5):
                break  # em[ak] < 0.5
            nb_count -= 1
            del wX, wy
        # 21 times is the maximum running number
        indices.append(deepcopy(idx))
        del nb_count

        clfs.append(deepcopy(clf))
        # calculate alpha
        alpha[k] = 0.5 * np.log2(
            check_zero((1. - em[k]) / check_zero(em[k])))
        # update weights.
        # Notice that: y \in {-1,+1} here, transform from {0,1}
        i_tr = (np.array(y_trn) * 2 - 1) * (inspect * 2 - 1)
        if (k + 1) < nb_cls:
            weight[k + 1] = weight[k] * np.exp(-1. * alpha[k] * i_tr)
            zm = np.sum(weight[k + 1])
            weight[k + 1] /= check_zero(zm)
    # regularization: alpha, sigma(coef)=1.
    am = np.sum(alpha)
    alpha = [float(i / am) for i in alpha]
    del weight, em, clf, i_tr, zm, am
    gc.collect()
    # return deepcopy(alpha), deepcopy(clfs), deepcopy(indices)
    return alpha, clfs, indices


# ----------------------------------
# AdaBoost \citep{zhou2012ensemble} pp.25
#   with my modification to make it work for multi-class
# ----------------------------------
# Multiclass Extension \citep{zhou2012ensemble} pp.38-39
# The LPBoost algorithm (and others)
#
# AdaBoost.M1 [Freund and Schapire, 1997] is a very straightforward extension
#           alpha_t = \frac{1}{2} \ln( (1-\epsilon_t)/\epsilon_t )
#       the same as Figure 2.2 except that
#       the base learners now are multiclass learners instead of binary classifiers.
# SAMME [Zhu et al., 2006] is an improvement over AdaBoost.M1, which replace
#       line 6 of AdaBoost.M1 in Figure 2.2 by
#       alpha_t = \frac{1}{2} \ln( (1-\epsilon_t)/\epsilon_t ) + \ln(|\mathcal{Y}| -1)
# AdaBoost.MH [Schapire and Singer, 1999]
# AdaBoost.M2 [Freund and Schapire, 1997], later generalized as AdaBoost.MR
# AdaBoost.MR [Schapire and Singer, 1999]
#


def BoostingEnsemble_multiclass(X_trn, y_trn, name_cls, nb_cls,
                                name_ens='AdaBoostM1'):
    assert name_ens in ['AdaBoostM1', 'SAMME']
    _, dY = judge_transform_need(y_trn)  # _:vY
    if dY == 1:
        dY = 2  # \mathcal{Y} = \{-1,+1\}

    clfs, nb_trn = [], len(y_trn)  # init started
    weight = np.zeros((nb_cls, nb_trn), dtype=DTY_FLT)
    em, alpha, indices = [0.] * nb_cls, [0.] * nb_cls, []
    # Initialize the weight distribution # $\mathcal{D}_0$
    weight[0] = np.ones(nb_trn, dtype=DTY_FLT) / nb_trn
    for k in range(nb_cls):
        nb_cnt = 20
        while nb_cnt >= 0:
            # Train a classifier $h_t$ from $D$ under distribution
            # $\mathcal{D}_k$
            wX, wy, idx = _AdaBoostSelectTraining(
                X_trn, y_trn, weight[k].tolist())
            clf = individual(name_cls, wX, wy)  # name_cls.fit(wX, wy)
            # Evaluate the error of $h_t$
            inspect = clf.predict(X_trn)
            i_tr = np.not_equal(inspect, y_trn)  # inspect != y_trn
            em[k] = np.sum(weight[k] * i_tr)
            # If $\epsilon_t > 0.5$, the break
            # if (em[k] >= 0.) and (em[k] <= 0.5):
            #     break
            if 0. <= em[k] <= 0.5:
                break

            nb_cnt -= 1
            del wX, wy
        # 21 is the maximum number of running times
        indices.append(deepcopy(idx))
        del nb_cnt
        clfs.append(deepcopy(clf))
        # Determine the weight of $h_t$
        alpha[k] = 0.5 * np.log(check_zero(
            (1. - em[k]) / check_zero(em[k])))
        if name_ens == 'SAMME':
            alpha[k] += np.log(dY - 1)
        # if np.isnan(alpha[k]):
        #     alpha[k] = 0.  # for robustness
        alpha[k] = float(np.nan_to_num(alpha[k]))

        # Update the distribution, where Z_t is a normalization factor
        i_tr = np.equal(inspect, y_trn) * 2 - 1
        if (k + 1) < nb_cls:
            weight[k + 1] = weight[k] * np.exp(-1. * alpha[k] * i_tr)
            zm = np.sum(weight[k + 1])
            weight[k + 1] /= check_zero(zm)
        # Z_t which enables $\mathcal{D}_{t+1}$ to be a distribution
    am = check_zero(np.sum(alpha))
    alpha = [float(i / am) for i in alpha]
    del weight, em, clf, i_tr, zm, am
    gc.collect()
    return deepcopy(alpha), deepcopy(clfs), deepcopy(indices)
    # return alpha, clfs, indices


# ============================================
#  Obtain ensemble, for multi classification
#  Discard Algorithms Update
# ============================================
#
# for incremental/online learning
# from sklearn import linear_model    # Perceptron, SGDClassifier,
#                                     # PassiveAggressiveClassifier
# from sklearn import neural_network  # MLPClassifier
#

# '''
# Classification
#     sklearn.naive_bayes.MultinomialNB
#     sklearn.naive_bayes.BernoulliNB
#     sklearn.linear_model.Perceptron
#     sklearn.linear_model.SGDClassifier
#     sklearn.linear_model.PassiveAggressiveClassifier
#     sklearn.neural_network.MLPClassifier
# Regression
#     sklearn.linear_model.SGDRegressor
#     sklearn.linear_model.PassiveAggressiveRegressor
#     sklearn.neural_network.MLPRegressor
# Clustering
#     sklearn.cluster.MiniBatchKMeans
#     sklearn.cluster.Birch
# Decomposition / feature Extraction
#     sklearn.decomposition.MiniBatchDictionaryLearning
#     sklearn.decomposition.IncrementalPCA
#     sklearn.decomposition.LatentDirichletAllocation
# Preprocessing
#     sklearn.preprocessing.StandardScaler
#     sklearn.preprocessing.MinMaxScaler
#     sklearn.preprocessing.MaxAbsScaler
# '''


# ----------------------------------
#  Acquisition of ``coef, clfs''
# ----------------------------------
# name_ens = name_alg


def EnsembleAlgorithm(name_ens, name_cls, nb_cls, X_trn, y_trn):
    if name_ens == "Bagging":
        coef, clfs, indices = BaggingEnsembleAlgorithm(
            X_trn, y_trn, name_cls, nb_cls)
    elif name_ens in ["AdaBoostM1", "SAMME"]:
        coef, clfs, indices = BoostingEnsemble_multiclass(
            X_trn, y_trn, name_cls, nb_cls, name_ens=name_ens)
    elif name_ens == "AdaBoost":  # only works for Y \in {0,1}
        coef, clfs, indices = AdaBoostEnsembleAlgorithm(
            X_trn, y_trn, name_cls, nb_cls)
    else:
        raise ValueError(
            "Error occurred in `data_classify.py`."  # proper
            "Please select an appropriate ensemble method.")
    # coef, clfs:   shape = (1, nb_cls)
    # return deepcopy(  # coef.copy()
    return coef, clfs, indices


# """
# # calculate accuracies of base/weak/individual classifiers
# # option: or
# #   CalcBaseAccuracies(clfs, X_tst, y_tst)
# #   CalcBaseAccuracies(clfs, X_trn, y_trn)
#
# def CalcBaseAccuracies(clfs, X, y):
#   y = np.array(y)
#   yt = [clf.predict(X).tolist() for clf in clfs]
#   accpl = [np.mean(np.array(t) == y) for t in yt]
#   del yt, y
#   gc.collect()
#   return deepcopy(accpl)  # list
#
#
# # only works for binary and Y \in {0,1}
# # option: or
# #   CalcPerformAccuracy(coef, clfs, X_tst, y_tst)
# #   CalcPerformAccuracy(coef, clfs, X_trn, y_trn)
#
# def CalcPerformAccuracy(coef, clfs, X, y):
#   coef = np.transpose([coef])  # np.array(np.mat(coef).T)
#   yt = [clf.predict(X).tolist() for clf in clfs]
#   y = np.array(y) * 2 - 1
#   yt = np.array(yt) * 2 - 1
#   fcode = np.sum(yt * coef, axis=0)
#   fcode = np.sign(fcode)
#   #   #
#   tie = list(map(np.sum, [fcode == 0, fcode == 1, fcode == -1]))
#   if tie[1] > tie[2]:
#     fcode[fcode == 0] = 1
#   else:
#     fcode[fcode == 0] = -1
#   # endif there is a tie (>=)
#   del tie
#   #   #
#   accsg = np.mean(fcode == y)  # singular
#   del coef, yt, y, fcode
#   gc.collect()
#   return accsg
# """


# ----------------------------------
#  Acquisition of ``y_insp, y_pred''
# ----------------------------------


# only works for binary and Y \in {0,1}
# option: or
#   calc_acc_sing_and_pl(y_trn, y_insp, coef)
#   calc_acc_sing_and_pl(y_tst, y_pred, coef)

def calc_acc_sing_pl_and_pr(y, yt, coef):
    accpl = [np.mean(
        np.equal(t, y)).tolist() for t in yt]
    coef = np.array([coef]).transpose()

    if 0 in y and 1 in y:
        y = np.array(y) * 2 - 1
        yt = np.array(yt) * 2 - 1
    elif -1 in y and 1 in y:
        y = np.array(y)
        yt = np.array(yt)

    fcode = np.sum(yt * coef, axis=0)
    fcode = np.sign(fcode)

    tie = list(map(np.sum, [
        fcode == 0, fcode == 1, fcode == -1]))
    if tie[1] > tie[2]:
        fcode[fcode == 0] = 1
    else:
        fcode[fcode == 0] = -1
    del tie
    # endif there is a tie (>=)

    accsg = np.mean(fcode == y).tolist()  # singular
    del coef, yt, y  # ,fcode
    accpr = np.mean(accsg >= np.array(accpl))
    gc.collect()
    return deepcopy(accpl), accsg, accpr, fcode
    # return accpl, accsg, float(accpr), fcode


def original_ensemble_from_train_set(name_ens, name_cls, nb_cls,
                                     X_trn, y_trn, X_val, X_tst):
    if name_ens == "Bagging":
        coef, clfs, idx = BaggingEnsembleAlgorithm(
            X_trn, y_trn, name_cls, nb_cls)
    elif name_ens in ["AdaBoostM1", "SAMME"]:
        coef, clfs, idx = BoostingEnsemble_multiclass(
            X_trn, y_trn, name_cls, nb_cls, name_ens=name_ens)
    elif name_ens == "AdaBoost":
        coef, clfs, idx = AdaBoostEnsembleAlgorithm(
            X_trn, y_trn, name_cls, nb_cls)
    else:
        raise ValueError(
            "Error occurred in `data_classify.py`!"
            "Please select a proper ensemble method.")

    # coef, clfs:   shape = (1, nb_cls)
    y_insp = [t.predict(X_trn).tolist() for t in clfs]
    y_pred = [t.predict(X_tst).tolist() for t in clfs]
    y_cast = [] if not X_val else [
        t.predict(X_val).tolist() for t in clfs]

    # return deepcopy(
    return y_insp, y_cast, y_pred, coef, clfs, idx


# ============================================
#  Valuation Codes
# ----------------------------------


# ----------------------------------------
# obtain ensemble
# ----------------------------------------

def achieve_ensemble_from_train_set(name_ens, abbr_cls, nb_cls,
                                    X_trn, y_trn, X_val, X_tst):
    since = time.time()
    name_cls = INDIVIDUALS[abbr_cls]

    coef, clfs, indices = EnsembleAlgorithm(
        name_ens, name_cls, nb_cls, X_trn, y_trn)
    y_insp = [j.predict(X_trn).tolist() for j in clfs]  # inspect
    y_pred = [j.predict(X_tst).tolist() for j in clfs]  # predict
    y_cast = [j.predict(
        X_val).tolist() for j in clfs] if len(X_val) > 0 else []

    tim_elapsed = time.time() - since
    space_cost__ = asizeof(clfs) + asizeof(coef)

    return deepcopy(coef), clfs, indices, y_insp, y_cast, y_pred, \
        tim_elapsed, space_cost__
