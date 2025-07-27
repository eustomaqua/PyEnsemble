# coding: utf-8
#
# Target:
#   Provide names for diversity and pruning
#


# Ensemble Classifiers

from sklearn import tree            # DecisionTreeClassifier()
from sklearn import naive_bayes     # GaussianNB, MultinomialNB, BernoulliNB
from sklearn import svm             # SVC, NuSVC, LinearSVC
from sklearn import neighbors       # KNeighborsClassifier(n_neighbors,
#                                   #   weights='uniform' or 'distance')
from sklearn import linear_model    # SGDClassifier(loss='hinge', penalty='l1' or 'l2')
from sklearn import neural_network  # MLPClassifier

# import pandas as pd
# import numpy as np


# -------------------------------------
# -------------------------------------


######################################
#  Ensemble Classifiers
######################################

# X_trn, X_tst, X_val:  np.ndarray, shape=(nb_???, nb_feat) i.e., [[nb_feat] nb_???]
# y_trn, y_tst, y_val:  np.ndarray, shape=(nb_trn/tst/val,) i.e., [nb_trn/tst/val]
#
# Y \in {0, 1}
# y_insp:       np.ndarray, shape=(nb_cls, nb_trn) i.e., [[nb_trn] nb_cls], inspect
# y_pred:       np.ndarray, shape=(nb_cls, nb_tst) i.e., [[nb_tst] nb_cls], predict
# y_cast:       np.ndarray, shape=(nb_cls, nb_val) i.e., [[nb_val] nb_cls], verdict/validate
# coefficient:              np.ndarray, shape=(nb_cls,) i.e., [nb_cls]
# weights (in resample):    np.ndarray, shape=(nb_y/X,) i.e., [nb_y/X]
#

AVAILABLE_ABBR_ENSEM = ['Bagging', 'AdaBoostM1', 'SAMME']
AVAILABLE_ABBR_CLS = [
    'DT', 'NB', 'SVM', 'linSVM', 'kNNu', 'kNNd',
    'MLP', 'LR1', 'LR2', 'LM1', 'LM2',
    # 'LM1', 'LM2', 'LR1', 'LR2', 'kNNu', 'kNNd', 'MLP',
]  # ALG_NAMES                # 'lmSGD','LR', # NN,Lsvm


NAME_INDIVIDUALS = {
    'DT': tree.DecisionTreeClassifier(),
    'NB': naive_bayes.GaussianNB(),
    'SVM': svm.SVC(),
    'linSVM': svm.LinearSVC(),  # 'L-SVM', 'lSVM/Lsvm'
    'kNNu': neighbors.KNeighborsClassifier(weights='uniform'),
    'kNNd': neighbors.KNeighborsClassifier(weights='distance'),
    'LR1': linear_model.LogisticRegression(penalty='none'),
    'LR2': linear_model.LogisticRegression(penalty='l2'),  # default
    'LM1': linear_model.SGDClassifier(penalty='l1'),
    'LM2': linear_model.SGDClassifier(penalty='l2'),  # default
    'MLP': neural_network.MLPClassifier(),  # NN
}

INDIVIDUALS = NAME_INDIVIDUALS


SPEC_INDIVIDUALS = {
    'DT': tree.DecisionTreeClassifier(),
    'NB': naive_bayes.GaussianNB(),
    'SVM': svm.SVC(),
    'linSVM': svm.LinearSVC(),  # 'L-svm', 'LSVM'
    'LM1': linear_model.SGDClassifier(
        loss="hinge", penalty="l1"),
    'LM2': linear_model.SGDClassifier(
        loss="hinge", penalty="l2"),
    # 'LM*':  # max_iter=1000
    'kNNu': neighbors.KNeighborsClassifier(
        n_neighbors=3, weights="uniform"),
    'kNNd': neighbors.KNeighborsClassifier(
        n_neighbors=3, weights="distance"),
    'MLP': neural_network.MLPClassifier(
        solver='lbfgs', alpha=1e-5,  # random_state=1,
        hidden_layer_sizes=(5, 2)),  # MLP
    #   #   #   #   #   #   #   #   #   #
    'mNB': naive_bayes.MultinomialNB(),
    'bNB': naive_bayes.BernoulliNB(),
    'nuSVM': svm.NuSVC(),  # 'nusvc'
    'SVMs': svm.SVC(gamma='scale'),  # sklearn 0.22 default
    'SVMa': svm.SVC(gamma='auto'),   # sklearn 0.21.3 default
}
# bilinear, for 3rd_complex_ensem
# ref: e.g., https://scikit-learn.org/stable/modules/classes.html


######################################
#  Ensemble Diversity Measures
######################################

# Diversity Measures
# -----------------------------
#
# ha, hb:   alternative
# hi, hj:   np.ndarray, with shape [m,], not list
# yt:       np.ndarray, shape=(nb_cls, m) i.e., [[m,] nb_cls]
#
# y \in {0, 1} -> {-1, +1}
# m = nb_y
#


PAIRWISE = {
    'Disag': 'Disagreement',  # Disagreement Measure [Skalak, 1996, Ho, 1998]
    'QStat': 'Q_statistic',   # Q-Statistic [Yule, 1900]
    'Corre': 'Correlation',   # Correlation Coefficient [Sneath and Sokal, 1973]
    'KStat': 'K_statistic',   # Kappa-Statistic [Cohen, 1960]
    'DoubF': 'Double_fault',  # Double-Fault Measure [Giacinto and Roli, 2001]
}

NONPAIRWISE = {
    'KWVar': 'KWVariance',    # Kohavi-Wolpert Variance [Kohavi and Wolpert, 1996]
    #                         #                         [Kuncheva and Whitaker, 2003]
    'Inter': 'Interrater',    # Interrater agreement [Fleiss, 1981]
    #                         #                      [Kuncheva and Whitaker, 2003]
    'EntCC': 'EntropyCC',     # Entropy, $Ent_{cc}$, [Cunningham and Carney, 2000]
    'EntSK': 'EntropySK',     # Entropy, $Ent_{sk}$, [Shipp and Kuncheva, 2002]
    'Diffi': 'Difficulty',    # Difficulty, [Hansen and Salamon, 1990]
    #                         #             [Kuncheva and Whitaker, 2003]
    'GeneD': 'Generalized',   # Generalized Diversity [Partridge and Krzanowski, 1997]
    'CFail': 'CoinFailure',   # Coincident Failure [Partridge and Krzanowski, 1997]
}

# AVAILABLE_NAME_DIVER = list(PAIRWISE.keys())+list(NONPAIRWISE.keys())
AVAILABLE_NAME_DIVER = [
    'Disag', 'QStat', 'Corre', 'KStat', 'DoubF',
    'KWVar', 'Inter', 'EntCC', 'EntSK', 'Diffi', 'GeneD', 'CFail']
DIVER_PAIRWISE = ['Disag', 'QStat', 'Corre', 'KStat', 'DoubF']
DIVER_NON_PAIR = [
    'KWVar', 'Inter', 'EntCC', 'EntSK', 'Diffi', 'GeneD', 'CFail']


######################################
#  Ensemble Pruning Methods
######################################

# from typing import List

# Ensemble Pruning
# --------------------------
#
# X_trn, X_tst, X_val
# y_trn, y_tst, y_val
# nb_trn/tst/val, nb_feat
# pr_feat, pr_pru
# k1,m1,lam1, k2,m2,lam2
#
# k?:   the number of selected objects (classifiers / features)
# m?:   the number of machines doing ensemble pruning / feature selection
# \lambda:  tradeoff
#
# yt:   predicted results, np.ndarray, shape=(nb_cls, nb_y) i.e., list [[nb_y] nb_cls]
# yo:   pruned results,    np.ndarray, shape=(nb_pru, nb_y) i.e., list [[nb_y] nb_pru]
#

RANKING_BASED = [
    'ES',    # Early Stopping
    'KL',    # KL divergence Pruning
    # 'KP',  # # Kappa Pruning
    'KP',    # 'KPk', Kappa Pruning (kuncheva, multi-class)
    'KP+',   # 'KPz', Kappa Pruning (zhou2012, multi from binary)
    'OO',    # Orientation Ordering Pruning
    'RE',    # Reduce Error Pruning
    'CM',    # Complementarity Measure Pruning ??
    'KL+',   # KL divergence Pruning (modified version of mine)
    'OEP',   # OEP in Pareto Ensemble Pruning
]  # ORDERING_BASED

CLUSTERING_BASED = []

OPTIMIZATION_BASED = [
    'DREP',  # DREP Pruning  # works for [binary] only, Y\in{-1,+1}
    'SEP',   # SEP in Pareto Ensemble Pruning
    'PEP',   # PEP in Pareto Ensemble Pruning
    'PEP+',  # PEP (modified version of mine)
]


# Composable Core-sets
# --------------------------
# \citep{indyk2014composable, aghamolaei2015diversity, abbassi2013diversity}
# Composable Core-sets for Diversity and Coverage Maximization
# Diversity Maximization via Composable Coresets
# Diversity Maximization Under Matroid Constraints
# --------------------------
# works for [multi-class classification]
#

# def pruning_methods(name_func, *params_func):
#     return name_func(*params_func)
#
# Remark:
#     specially for dt.DIST in DDisMI
#     using Kappa statistic: K, theta1, theta2 = KappaMulti(ha, hb, y)
#         K = 0, different;    K = 1, completely the same
#

COMPOSABLE_CORE_SETS = [
    'GMA',   # GMM_Algorithm  # 'GMM'
    'LCS',   # Local_Search
]  # DIVERSITY MAXIMIZATION
# modified by me to make them suitable for ensemble pruning problems

# AVAILABLE_NAME_PRUNE = RANKING_BASED + OPTIMIZATION_BASED
# AVAILABLE_NAME_PRUNE = RANKING_BASED + COMPOSABLE_CORE_SETS + OPTIMIZATION_BASED
AVAILABLE_NAME_PRUNE = [
    # 'ES', 'KL', 'KL+', 'KP', 'RE', 'CM', 'OO',
    # 'GMA', 'LCS',
    # 'DREP', 'SEP', 'OEP', 'PEP', 'PEP+'
    #
    'ES', 'KL', 'KL+', 'KPk', 'KPz',  # 'KP', 'KP+',
    'RE', 'CM', 'OO', 'GMA', 'LCS',
    'DREP',   # 'DREPbi' # the most original version in li2012diversity `DREP`
    'drepm',  # `drep_multi` instead of `DREP binary (original version)`
    'SEP', 'OEP', 'PEP', 'PEP+', 'pepre', 'pepr+'  # `pep_pep_integrate`
]


# Ensemble Pruning Latest
# --------------------------
#
# instances             $\mathbf{N}$, a $n \times d$ matrix, a set of instances
# features              $\mathbf{F}$, with cardinality $d$, a set of features
# class label           $\mathbf{L}$, a $n$-dimensional vector
# classification result $\mathbf{U}$, a $n \times t$ matrix
# original ensemble     $\mathbf{T}$, with cardinality $t$, a set/pool of original
#                                     ensemble with $t$ individual classifiers
# pruned subensemble    $\mathbf{P}$, with cardinality $p$, a set of individual
#                                     classifiers after ensemble pruning
#
# name_ensem = Bagging, AdaBoost
# abbr_cls   = DT, NB, SVM, LSVM
# name_cls   =
#   nb_cls   =
#       k    = the number of selected objects (classifiers / features)
#       m    = the number of machines doing ensemble pruning / feature selection
#   \lambda  = tradeoff
#       X    = raw data
#       y    = raw label
#       yt   = predicted result, [[nb_y] nb_cls] list, `nb_cls x nb_y` array
#       yo   = pruned result, not `yp`
#
# name_prune =
# name_diver =
# KL distance between two probability distributions p and q:
#     scipy.stats.entropy(p, q)
#

LATEST_NAME_PRUNE = [
    # ordering-based, MRMC-ordered-aggregation pruning
    'MRMC-MRMR', 'MRMC-MRMC', 'MRMC-ALL',
    'MRMREP',  # optimization-based, \citep{li2018mrmr}
    'mRMR-EP', 'Disc-EP',  # \citep{cao2018optimizing}
    # \citep{zhang2019two}
    'TSP-AP', 'TSP-DP', 'TSP-AP+DP', 'TSP-DP+AP',
    "TSPrev-DP", "TSPrev-AD", "TSPrev-DA",
]


######################################
#  Datasets
######################################

# UCI Repository
# --------------------------
#

GMM_DATASET = [
    'gmm_2D_n4k', 'gmm_3D_n2k', 'gmm_10D_n1k'
]
DIVERSITY_DATA = [
    'Ames', 'card', 'heart', 'iono', 'liver', 'ringnorm',
    'sonar', 'spam', 'waveform', 'wisconsin',  # house
]

DIVERHUGE_DATA = [
    'credit', 'landsat', 'page', 'shuttle', 'wilt'
]
UCI_DATASET = [
    'ecoli+', 'ecoli', 'yeast+', 'yeast',
    'mammographic_masses', 'segmentation', 'madelon',
    # 'segmentation_data', 'segmentation_test',
    # 'madelon_train', 'madelon_valid',
    'sensor_readings_2', 'sensor_readings_4',
    'sensor_readings_24',
    # 'waveform', 'waveform_noise', 'EEGEyeState'
    "waveform_init", "waveform_noise", "EEGEyeState"
]

# Python Packages
# --------------------------
#

KERAS_DATASET = [
    "mnist", "fashion", "cifar10",
]
Keras_DATAS = [
    "mnist", "fmnist", "fashion",
    "cifar10", "cifar100f", "cifar100c",
    "reuters"  # omitted
]

SET_DATA_NAMES = {
    'Keras': Keras_DATAS,
    'catdog': '',
    'UCI': UCI_DATASET,
    'GMM': GMM_DATASET,
    'diversity': DIVERSITY_DATA,
    'diverhuge': DIVERHUGE_DATA,
    'keras': KERAS_DATASET,
}

FAIR_DATASET = [
    # 'ricci', 'german', 'adult', 'ppc', 'ppvc'
    'ricci', 'german', 'adult', 'ppr', 'ppvr'
]
