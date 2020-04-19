# coding: utf8
# Aim to: classify data using classification ensembles
#       data_ensemble.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()
import numpy as np

from sklearn import tree            # DecisionTreeClassifier()
from sklearn import naive_bayes     # GaussianNB, MultinomialNB, BernoulliNB
from sklearn import svm             # SVC, NuSVC, LinearSVC
from sklearn import neighbors       # KNeighborsClassifier(n_neighbors, weights='uniform' or 'distance')
from sklearn import linear_model    # SGDClassifier(loss='hinge', penalty='l1' or 'l2')




# individual
#

# def individual(args):
#     name_cls = args[0]
#     wX = args[1]
#     wy = args[2]
#     return name_cls.fit(wX, wy)

def individual(name_cls, wX, wy):
    return name_cls.fit(wX, wy)

NAME_INDIVIDUALS = {
    'DT'  : tree.DecisionTreeClassifier(),
    'NB'  : naive_bayes.GaussianNB(),
    'SVM' : svm.SVC(gamma='scale'),
    'LSVM': svm.LinearSVC(),
    'KNNu': neighbors.KNeighborsClassifier(weights='uniform'),
    'KNNd': neighbors.KNeighborsClassifier(weights='distance'),
    'LM1' : linear_model.SGDClassifier(penalty='l1'),
    'LM2' : linear_model.SGDClassifier(penalty='l2'),
}



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
    del vY,dY, y,yt, vote,loca
    gc.collect()
    return deepcopy(fens)


def majority_voting(y, yt):
    # vY = np.unique(np.unique(y).tolist() + np.unique(yt).tolist())
    vY = np.unique(np.vstack((y, yt)))
    dY = len(vY)
    y = np.array(y);    yt = np.array(yt)
    vote = [np.sum(yt == vY[i], axis=0).tolist() for i in range(dY)]
    #
    nb_cls = len(yt)
    half = int(np.ceil(nb_cls / 2.))
    vts = np.array(vote).T  # transpose
    #
    loca = [np.where(j >= half)[0][0] if (len(np.where(j >= half)[0]) > 0) else -1 for j in vts]
    fens = [vY[i] if (i != -1) else -1 for i in loca]
    #
    del vY,dY, y,yt, vote,half,vts,loca, nb_cls
    gc.collect()
    return deepcopy(fens)


def weighted_voting(y, yt, coef):
    # vY = np.unique(np.unique(y).tolist() + np.unique(yt).tolist())
    # vY = np.unique(np.vstack((y, yt)))
    vY = np.unique(np.concatenate([[y], yt]))
    dY = len(vY)
    y = np.array(y);    yt = np.array(yt)
    coef = np.transpose([coef])  ## coef = np.array(np.mat(coef).T)
    weig = [np.sum(coef * (yt == vY[i]), axis=0).tolist() for i in range(dY)]
    loca = np.array(weig).argmax(axis=0)
    fens = [vY[i] for i in loca]
    del vY,dY, y,yt,coef, weig,loca
    gc.collect()
    return deepcopy(fens)


