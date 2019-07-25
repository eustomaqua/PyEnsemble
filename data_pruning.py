# coding: utf8
# Aim to: prune the ensemble, (existing methods)
'''
Including:
    self._name_pru_set = ['ES','KL','KL+','KP','OO','RE', 'GMM','LCS', 'DREP','SEP','OEP','PEP','PEP+']

1) Ranking based / Ordering based
    ES   |  Early_Stopping(              yt,    nb_cls, nb_pru)            |  multi-class 
    KL   |  KL_divergence_Pruning(       yt,    nb_cls, nb_pru)            |  multi-class 
    KP   |  Kappa_Pruning(               yt, y, nb_cls, nb_pru)            |  multi-class 
    OO   |  Orientation_Ordering_Pruning(yt, y                ) with flag  |  multi-class  
    RE   |  Reduce_Error_Pruning(        yt, y, nb_cls, nb_pru)            |  multi-class  
    KL+  |  KL_divergence_Pruning_modify(yt,    nb_cls, nb_pru)            |  multi-class 
    OEP  |  PEP_OEP(                     yt, y, nb_cls        )            |  multi-class  
2) Clustering based
3) Optimization based
    DREP |  DREP_Pruning(                yt, y, nb_cls,         rho)       |  binary  
    SEP  |  PEP_SEP(                     yt, y, nb_cls,         rho)       |  multi-class  
    PEP  |  PEP_PEP(                     yt, y, nb_cls,         rho)       |  multi-class 
    PEP+ |  PEP_PEP_modify(              yt, y, nb_cls,         rho)       |  multi-class 
4) Other
6) Composable Core-sets, Diversity Maximization
    GMM  |  GMM_Algorithm(               yt, y, nb_cls, nb_pru)            |  multi-class  
    LCS  |  Local_Search(                yt, y, nb_cls, nb_pru, epsilon)   |  multi-class
'''


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
# garbage collector
import gc
gc.enable()

import os
import sys
import time

import numpy as np 

from math import isinf
from scipy import stats


from utils_constant import DTY_FLT
from utils_constant import DTY_INT
from utils_constant import DTY_BOL

from utils_constant import CONST_ZERO
from utils_constant import check_zero

from utils_constant import GAP_INF  # 2 ** 31 - 1
from utils_constant import GAP_MID  # 1e12
from utils_constant import GAP_NAN  # 1e-12


# obtain ensemble
from sklearn import tree            # DecisionTreeClassifier()
from sklearn import naive_bayes     # GaussianNB, MultinomialNB, BernoulliNB
from sklearn import svm             # SVC, NuSVC, LinearSVC
from sklearn import neighbors       # KNeighborsClassifier(n_neighbors, weights='uniform' or 'distance')
from sklearn import linear_model    # SGDClassifier(loss="hinge", penalty="l1" or "l2")



'''
instances             $\mathbf{N}$, a $n \times d$ matrix, a set of instances
features              $\mathbf{F}$, with cardinality $d$, a set of features
class label           $\mathbf{L}$, a $n$-dimensional vector
classification result $\mathbf{U}$, a $n \times t$ matrix
original ensemble     $\mathbf{T}$, with cardinality $t$, a set/pool of original ensemble with $t$ individual classifiers
pruned subensemble    $\mathbf{P}$, with cardinality $p$, a set of individual classifiers after ensemble pruning

name_ensem  = Bagging, AdaBoost
abbr_cls    = DT, NB, SVM, LSVM
name_cls    = 
nb_cls      =
k           = the number of selected objects (classifiers / features)
m           = the number of machines doing ensemble pruning / feature selection
\lambda     = tradeoff
X           = raw data
y           = raw label
yt          = predicted result,  [[nb_y] nb_cls] list, `nb_cls x nb_y' array
yo          = pruned result, not `yp'

X_trn, y_trn, X_tst, y_tst
nb_trn, nb_tst, nb_feat, 
pr_feat, pr_pru
k1,m1,lam1, k2,m2,lam2

name_prune  = ....
name_diver  = ....


KL distance between two probability distributions p and q:
scipy.stats.entropy(p, q)
'''


#==================================
# initial
#==================================


#----------------------------------
#
#----------------------------------
#
# \citep{martinez2009analysis}, Reduce-Error Pruning
# \citep{tsoumakas2009ensemblle},


from data_classify import plurality_voting
from data_classify import majority_voting
from data_classify import weighted_voting  

from data_diversity import Kappa_Statistic_multiclass
from data_diversity import Kappa_Statistic_binary



#----------------------------------
# data_entropy.py
#----------------------------------
# 
# data_entropy.py ---- Inspired by margineantu1997pruning
# KL distance between two probability distributions p and q:
# KL_distance = scipy.stats.entropy(p, q)


# the KL distance between two probability distributions p and q is:
# D(p || q) = 
#
def KLD(p, q):
    p = np.array(p, dtype=DTY_FLT)
    q = np.array(q, dtype=DTY_FLT)
    if np.sum(p) != 1.0:
        tem = np.sum(p)
        p /= check_zero(tem)
    if np.sum(q) != 1.0:
        tem = np.sum(q)
        q /= check_zero(tem)
    ans = 0.;   n = len(p) 
    for i in range(n):
        tem = p[i] / check_zero(q[i])
        tem = p[i] * np.log(check_zero(tem))
        ans += tem
    return ans


# softmax regression
#
def softmax(y):
    return np.exp(y) / np.sum(np.exp(y), axis=0)



#----------------------------------
#
#----------------------------------



#==================================
# \citep{margineantu1997pruning}
# 
# Pruning Adaptive Boosting (ICML-97)  [multi-class classification, AdaBoost]
#==================================



#----------------------------------
# Early Stopping
# works for [multi-class]
#----------------------------------


def Early_Stopping(yt, nb_cls, nb_pru):
    yo = yt[: nb_pru]
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    P[: nb_pru] = True
    P = P.tolist()
    return deepcopy(yo), deepcopy(P)



#----------------------------------
# KL-divergence Pruning
# works for [mutli-class]
#----------------------------------
# 
# KL distance between two probability distributions p and q:
# KL_distance = scipy.stats.entropy(p, q)


# X, Y = classification vectors
# 
def KLD_vectors(X, Y):
    vXY = np.unique(np.concatenate([X, Y])).tolist()
    dXY = len(vXY)
    px = np.zeros(dXY); py = np.zeros(dXY)
    X = np.array(X);    Y = np.array(Y)
    for i in range(dXY):
        px[i] = np.mean(X == vXY[i])
        py[i] = np.mean(Y == vXY[i])
    px = px.tolist();   py = py.tolist()
    del i, X, Y, dXY
    ans = KLD(px, py)
    del px, py
    gc.collect()
    return ans


def JU_set_of_vectors(U):
    ans = 0.;   u = len(U)
    for i in range(u - 1):
        for j in range(i + 1, u):
            ans += KLD_vectors(U[i], U[j])
    del i, j, u
    gc.collect()
    return ans


def U_next_idx(yt, P):
    P = np.array(P)
    not_in_p = np.where(P == False)[0]
    #
    yt = np.array(yt);  ansJ = []
    for i in not_in_p:
        ansP = deepcopy(P)
        ansP[i] = True
        ansU = yt[ansP == True].tolist()
        ansJ.append( JU_set_of_vectors(ansU) )
        del ansP, ansU
    idx = ansJ.index( np.max(ansJ) )
    del ansJ, yt, P
    gc.collect()
    return not_in_p[idx]


def KL_divergence_Pruning(yt, nb_cls, nb_pru):
    P = np.zeros(nb_cls, dtype=DTY_BOL).tolist()
    P[0] = True
    while np.sum(P) < nb_pru:
        idx = U_next_idx(yt, P)
        P[idx] = True
        del idx
    yo = np.array(yt)[P].tolist()
    return deepcopy(yo), deepcopy(P)



#----------------------------------
# Kappa Pruning [binary classification]
# ! not multi-class
# now works on multi-class
#----------------------------------


# def Kappa(nb_y, nb_c, ha, hb):
#     dY = nb_c  # dY = nb_lab  # nb_label
#     m = nb_y   # number of instances / samples


def KappaMulti(ha, hb, y):
    vY = np.unique(np.concatenate([y, ha, hb]))
    dY = len(vY)  # number of labels / classes
    ha = np.array(ha);  hb = np.array(hb)
    #
    # construct a contingency table
    Cij = np.zeros(shape=(dY, dY))
    for i in range(dY):
        for j in range(dY):
            Cij[i, j] = np.sum((ha == vY[i]) & (hb == vY[j]))
    m = len(y)  # number of instances / samples
    # 
    c_diagonal = [Cij[i][i] for i in range(dY)]  # Cij[i, i]
    theta1 = np.sum(c_diagonal) / float(m)
    c_row_sum = [np.prod([Cij[i,i] + Cij[i,j] for j in range(dY) if j!=i]) for i in range(dY)]
    c_col_sum = [np.prod([Cij[i,j] + Cij[j,j] for i in range(dY) if i!=j]) for j in range(dY)]
    theta2 = np.sum(np.multiply(c_row_sum, c_col_sum)) / (float(m) ** 2)
    # 
    ans = (theta1 - theta2) / check_zero(1. - theta2)
    del dY, ha,hb, Cij, i,j,m
    del c_row_sum, c_col_sum, c_diagonal
    gc.collect()
    return ans, theta1, theta2


def Kappa_Pruning(yt, y, nb_cls, nb_pru):
    # initial
    Kij = np.zeros(shape=(nb_cls, nb_cls), dtype=DTY_FLT)
    Kij += sys.maxsize  # sys.maxint
    # 
    for i in range(nb_cls - 1):
        for j in range(i + 1, nb_cls):
            ans, _, _ = KappaMulti(yt[i], yt[j], y)
            Kij[i, j] = ans
    # upper triangular / triangle matrix 
    # 
    # the lowest \kappa
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    nb_p = 0
    while nb_p < nb_pru:
        idx = np.where(Kij == np.min(Kij))
        try:
            row = idx[0][0];    col = idx[1][0]
        except Exception as e:
            print("Kappa_Pruning -- nb_cls {} nb_pru {} nb_p {}".format(nb_cls, nb_pru, nb_p))
            print("Kappa_Pruning -- y, yt, {} \n{:23s}{}".format(y, '', yt))
            print("Kappa_Pruning -- idx {}".format(idx))
            print("Kappa_Pruning -- Kij \n{:20s}{}".format('', Kij))
            raise e
        else:
            pass
        finally:
            pass
        #   #
        if nb_p + 1 == nb_pru:
            P[row] = True
            Kij[row, :] = sys.maxsize
            Kij[:, row] = sys.maxsize
            nb_p += 1
        else:
            P[row] = True;  P[col] = True
            Kij[row, :] = sys.maxsize
            Kij[:, col] = sys.maxsize
            Kij[:, row] = sys.maxsize
            Kij[col, :] = sys.maxsize
            nb_p += 2
        del idx, row, col
    #   #
    yo = np.array(yt)[P == True].tolist()
    P = P.tolist()
    del nb_p, Kij
    gc.collect()
    return deepcopy(yo), deepcopy(P)




#----------------------------------
# 
#----------------------------------



#==================================
# \citep{martine2006pruning}
#
# Pruning in Ordered Bagging Ensembles (ICML-06)  [multi-class classification, Bagging]
#==================================


#----------------------------------
# Ordering Bagging Ensembles
# Orientation Ordering [multi-class classification]
#----------------------------------


# a, b are vectors
# 
def angle(a, b):
    a = np.array(a);    b = np.array(b)
    # dot product, scalar product
    prod = np.sum(a * b)    # $a \cdot b$  # or: prod = np.dot(a, b)
    # norm / module
    len1 = np.sqrt(np.sum(a * a))  # $|a|, |b|$
    len2 = np.sqrt(np.sum(b * b))
    # $\cos(\theta)$
    cos_theta = prod / check_zero(len1 * len2)
    theta = np.arccos(cos_theta)
    del a,b, prod,len1,len2, cos_theta
    gc.collect()
    return theta


# $\mathbf{c}_t$, as the $N_{tr}$-dimensional vector
# the signature vector of the classifier $h_t$, for the dataset $L_{tr}$ composed of $N_{tr}$ examples
#
def signature_vector(ht, y):
    y = np.array(y);    ht = np.array(ht)
    ct = 2. * (y == ht) - 1. 
    ans = ct.tolist()
    del y, ht, ct
    gc.collect()
    return deepcopy(ans)


# $c_{ti}$ is equal to +1 if $h_t$ (the t-th unit in the ensemble) correctly classifies the i-th example of $L_{tr}$
# $\mathbf{c}_{ens}$, the average signature vector of the ensemble is 
# 
def average_signature_vector(yt, y):
    ct = [signature_vector(ht, y) for ht in yt]
    cens = np.mean(ct, axis=0)  # np.sum(ct, axis=0) / float(nb_cls)
    cens = cens.tolist()
    return deepcopy(cens), deepcopy(ct)


# This study presents an ordering criterion based on the orientation of the signature vector
# of the individual classifiers with respect to a reference direction. 
# 
# This direction, coded in a reference vector, $\mathbf{c}_{ref}$, is the projection of the 
# first quadrant diagonal onto the hyper-plane defined by $\mathbf{c}_{ens}$. 
# 
def reference_vector(nb_y, cens):
    oc = np.ones(shape=nb_y)  # $\mathbf{o}$
    # cens = np.array(average_signature_vector(yt, y))
    cens = np.array(cens)  # $\mathbf{c}_{ens}$
    lam = -1. * np.sum(oc * cens) / np.sum(cens * cens)  # $\lambda$
    cref = oc + lam * cens  # $\mathbf{c}_{ref}$
    # perpendicular: np.abs(np.sum(cref * cens) - 0.) < 1e-6 == True
    # 
    # $\mathbf{c}_{ref}$ becomes unstable when the vectors that define the projection (i.e., 
    # $\mathbf{c}_{ref}$ and the diagonal of the first quadrant) are close to each other. 
    flag = angle(cref.tolist(), oc.tolist())
    #
    ans = cref.tolist()
    del oc, cens, lam, cref
    gc.collect()
    return deepcopy(ans), flag



# The classifiers are ordered by increasing values of the angle between the signature vectors 
# of the individual classifiers and the reference vector $\mathbf{c}_{ref}$ 
#
def Orientation_Ordering_Pruning(yt, y):
    nb_y = len(y)   # number of samples / instances
    cens, ct = average_signature_vector(yt, y)
    cref, flag = reference_vector(nb_y, cens)
    theta = [angle(i, cref) for i in ct]
    #
    # P = np.array(theta) < np.pi / 2. 
    P = np.array(theta) < (np.pi / 2.)
    P = P.tolist()
    del nb_y, cens,ct,cref,theta
    # if np.abs(flag - 0.) < 1e-3:
    #     print("$\mathbf{c}_{ref}$ becomes unstable!")
    if np.sum(P) == 0:
        P[ np.random.randint(len(P)) ] = True
    yo = np.array(yt)[P].tolist()
    #
    gc.collect()
    return deepcopy(yo), deepcopy(P), flag



#----------------------------------
#
#----------------------------------



#==================================
# \citep{martinez2009analysis}
# 
# An Analysis of Ensemble Pruning Techniques Based on Ordered Aggregation 
# (TPAMI)  [multi-class classification, Bagging]
#==================================



#----------------------------------
# Reduce-Error Pruning
#----------------------------------


# need to use a pruning set, subdivided from training set, with a sub-training set
# 
def Reduce_Error_Pruning(yt, y, nb_cls, nb_pru):
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    yt = np.array(yt)   # y = np.array(y)
    # 
    # first
    err = np.mean(yt != np.array(y), axis=1)
    idx = err.argmin()  # argmax()
    P[idx] = True
    #
    # next
    while np.sum(P) < nb_pru:
        # find the next idx
        not_in_p = np.where(P == False)[0]
        anserr = [] 
        for i in not_in_p:
            temP = deepcopy(P)
            temP[i] = True
            temyt = yt[temP].tolist()
            temfens = plurality_voting(y, temyt) 
            temerr = np.mean(np.array(temfens) != np.array(y), axis=0)
            anserr.append( temerr )
            del temP, temyt, temfens, temerr
        #   #
        idx = anserr.index( np.min(anserr) )
        P[ not_in_p[idx] ] = True
        del anserr, idx, not_in_p
    #   #
    yo = yt[P].tolist()
    P = P.tolist()
    gc.collect()
    return deepcopy(yo), deepcopy(P)



#----------------------------------
# Complementarity Measure
#----------------------------------



#----------------------------------
# Concurrency thinning
#----------------------------------



#----------------------------------
# Margin Distance Minimization
#----------------------------------



#----------------------------------
# Boosting-based Ordering [multi-class classification, Bagging]
# \citep{martinez2007using}
# Using Boosting to Prune Bagging Ensembles
#----------------------------------



#----------------------------------
#
#----------------------------------



#==================================
# \citep{tsoumakas2009ensemble}
# 
# An Ensemble Pruning Primer ()
#==================================


#----------------------------------
#
#----------------------------------



#==================================
# Modification of mine (with softmax)
#==================================


#----------------------------------
# KL-divergence Pruning 
#----------------------------------


# KL distance between two vectors X and Y:
# 
def KLD_pq(X, Y):
    # return stats.entropy(p, q)  # default: base=e
    p = softmax(X)
    q = softmax(Y)
    ans = KLD(p, q)
    del p, q
    gc.collect()
    return ans


def J(U):
    ans = 0.;   u = len(U)
    for i in range(u - 1):
        for j in range(i + 1, u):
            ans += KLD_pq(U[i], U[j])
    return ans


def KL_find_next(yt, P):
    P = np.array(P)
    not_in_p = np.where(P == False)[0]
    #
    yt = np.array(yt)
    ansJ = []
    for i in not_in_p:
        ansP = deepcopy(P)  # 啊，发现问题所在了，还是深拷贝的问题
        ansP[i] = True
        ansU = yt[ansP == True].tolist()
        ansJ.append( J(ansU) )
        del ansP, ansU
    idx = ansJ.index( np.max(ansJ) )
    del ansJ, yt, P
    # 
    gc.collect()
    return not_in_p[idx]  # Notice the position of P



# 
def KL_divergence_Pruning_modify(yt, nb_cls, nb_pru):
    # P = [False] * nb_cls;   P[0] = True
    P = np.zeros(nb_cls, dtype=DTY_BOL).tolist()
    P[0] = True
    while np.sum(P) < nb_pru:
        idx = KL_find_next(yt, P)
        P[idx] = True
    yo = np.array(yt)[np.array(P) == True].tolist()
    return deepcopy(yo), deepcopy(P)



#----------------------------------
#
#----------------------------------




#==================================
# \citep{indyk2014composable, aghamolaei2015diversity, abbassi2013diversity}
# 
# Composable Core-sets for Diversity and Coverge Maximization
# Diversity Maximization via Composable Coresets
# Diversity Maximization Under Matroid Constraints
#==================================
#
# works for [multi-class classification]


# def pruning_methods(name_func, *params_func):
#     return name_func(*params_func)
# 
# Remark:
#     specially for dt.DIST in DDisMI
#     using Kappa statistic: K, theta1, theta2 = KappaMulti(ha, hb, y)
#         K = 0, different;   K = 1, completely the same



#----------------------------------
# GMM(S, k)
#----------------------------------
# 
# Input:    S, a set of points; k, size of the subset
# Output:   S', a subset of S of size k
#   1.  S' <-- An arbitrary point p
#   2.  for i = 2,...,k do
#   3.      find p \in S\S' which maximizes min_{q\in S'} dist(p,q)
#   4.      S' <-- S' \cup {p}


def GMM_Kappa_sum(p, S, y):
    # ans = [KappaMulti(p, q, y)[0] for q in S]
    ans = []
    for q in S:
        tem, _, _ = KappaMulti(p, q, y)
        ans.append(tem)
    tem = np.sum(ans)
    return tem


def GMM_Algorithm(yt, y, nb_cls, nb_pru):
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    randseed = int(time.time() * GAP_MID % GAP_INF)
    prng = np.random.RandomState(randseed)
    idx = prng.randint(nb_cls)
    P[idx] = True
    #
    for i in range(1, nb_pru):
        # find_max_p
        all_q_in_S = np.array(yt)[P].tolist()
        idx_p_not_S = np.where(P == False)[0]
        if len(idx_p_not_S) == 0:
            idx = -1
        else:
            ans = [GMM_Kappa_sum(yt[j], all_q_in_S, y) for j in idx_p_not_S]
            idx_p = ans.index( np.max(ans) )
            idx = idx_p_not_S[idx_p]
            del ans, idx_p
        del all_q_in_S, idx_p_not_S
        # fine_max_p
        if idx > -1:
            P[idx] = True
    #   #   #
    P = P.tolist()
    del randseed, prng, idx
    yo = np.array(yt)[P].tolist()
    gc.collect()
    return deepcopy(yo), deepcopy(P)



#----------------------------------
# Local Search Algorithm
#----------------------------------
#
# Input:    S, a set of points; k, size of the subset
# Output:   S', a subset of S of size k
#   1.  S' <-- An arbitrary set of k points which contains the two farthest points
#   2.  while there exists p\in S\S' and p'\in S' such that div(S'\{p'} \cup {p}) >= div(S')(1+\epsilon/n) do
#   3.      S' <-- S' \ {p'} \cup {p}


def LocalSearch_kappa_sum(S, y):
    n = len(S)
    Kij = np.zeros(shape=(n, n))
    for i in range(n - 1):
        for j in range(i + 1, n):
            ans, _, _ = KappaMulti(S[i], S[j], y)
            Kij[i, j] = ans
    # upper triangular matrix
    ans = np.sum(Kij) / (n * (n - 1.) / 2.)
    del n, Kij
    gc.collect()
    return ans


def Local_Search(yt, y, nb_cls, nb_pru, epsilon):
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    # an abritrary set of k points which contains the two farthest points
    # 
    Kij = np.zeros(shape=(nb_cls, nb_cls))
    Kij += sys.maxsize
    for i in range(nb_cls - 1):
        for j in range(i + 1, nb_cls):
            ans, _, _ = KappaMulti(yt[i], yt[j], y)
            Kij[i, j] = ans
    # upper triangular matrix
    idx1 = np.where(Kij == np.min(Kij))
    row = idx1[0][0];   col = idx1[1][0]
    P[row] = True;      P[col] = True
    del Kij, i,j, ans, idx1  # row, col
    # 
    randseed = int(time.time() * GAP_MID % GAP_INF)
    prng = np.random.RandomState(randseed)
    idx2 = np.arange(nb_cls)
    prng.shuffle(idx2)
    idx3 = idx2[: nb_pru]
    if (row in idx3) and (col in idx3):
        idx4 = idx3  # idx2[: nb_pru]
    elif (row in idx3) and (col not in idx3):
        idx4 = idx3[: -1]
    elif (row not in idx3) and (col in idx3):
        idx4 = idx3[: -1]
    elif (row not in idx3) and (col not in idx3):
        idx4 = idx3[: -2]
    else:
        pass
    for i in idx4:
        P[i] = True
    del randseed, prng, idx2,idx3,idx4, row,col  # i,
    # 
    # while there exists p\in S\S'
    nb_count = np.sum(P) * (len(P) - np.sum(P))  # nb_cls = len(P)
    yt = np.array(yt)
    S_within  = np.where(P ==  True)[0].tolist()
    S_without = np.where(P == False)[0].tolist()
    while nb_count >= 0:
        flag = False  # whether exists (p, q)?
        div_b4 = LocalSearch_kappa_sum(yt[S_within].tolist(), y)
        for p in S_within:
            idx_p = S_within.index(p)
            for q in S_without:
                tem_q = deepcopy(S_within)
                tem_q[idx_p] = q
                div_af = LocalSearch_kappa_sum(yt[tem_q].tolist(), y)
                if div_af > div_b4 * (1. + epsilon / nb_cls):
                    flag = True
                    S_within = deepcopy(tem_q)
                    tem_p = deepcopy(S_without)
                    idx_q = S_without.index(q)
                    tem_p[idx_q] = p
                    S_without = deepcopy(tem_p)
                    del tem_p, tem_q
                    break
            #   #   #
            if flag == True:
                break
        #   #   #
        if flag == False:
            break
    #   #   #
    # end while
    del nb_count, S_without  # S_within, 
    #
    PP = np.zeros(nb_cls, dtype=DTY_BOL)
    PP[S_within] = True
    yo = np.array(yt)[PP].tolist() 
    PP = PP.tolist()
    del S_within
    gc.collect()
    return deepcopy(yo), deepcopy(PP)



#----------------------------------
#
#----------------------------------



#==================================
# \citep{li2012diversity}
#
# Diversity Regularized Ensemble Pruning (DREP)  [Binary, Bagging]
#==================================
# 
# H = \{ h_i(\mathbf{x}) \}, i=1,\dots,n,  h_i:\mathcal{X}\mapsto\{-1,+1\}
# S = \{ (\mathbf{x}_k, y_k) \}, k=1,\dots,m
# Note that:  y \in {-1, +1},  transform!



#----------------------------------
# DREP
#----------------------------------


# $f(\mathbf{x};H) = \frac{1}{n} \sum_{1\leqslant i\leqslant n} h_i(\mathbf{x})$
#
def DREP_fxH(yt):
    yt = np.array(yt)
    if yt.ndim == 1:
        # bug!  # yt = np.array(np.mat(yt).T).tolist()
        yt = np.array([yt])
    fens = np.mean(yt, axis=0)  # .tolist()
    del yt
    #
    fens = np.sign(fens)
    randseed = int(time.time() * GAP_MID % GAP_INF)
    prng = np.random.RandomState(randseed)
    # fens = [prng.randint(2)*2-1 if i==0. else i for i in fens]
    #
    tie = [np.sum(fens == i) for i in [0, 1, -1]]
    if tie[1] > tie[2]:
        fens[fens == 0] = 1
    elif tie[1] < tie[2]:
        fens[fens == 0] = -1
    else:
        fens[fens == 0] = prng.randint(2) * 2 - 1
    fens = fens.tolist()
    #
    gc.collect()
    return deepcopy(fens)


# $\diff(h_i, h_j) = \frac{1}{m} \sum_{1\leqslant k\leqslant m} h_i(\mathbf{x}_k) h_j(\mathbf{x}_k)$
# 
def DREP_diff(hi, hj):
    tem = np.array(hi) == np.array(hj)
    tem = np.array(tem, dtype=DTY_FLT) * 2 - 1
    return np.mean(tem)
    # works for multi-class now



# \rho \in (0, 1)
# 
def DREP_Pruning(yt, y, nb_cls, rho):
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    accpl = [np.mean(np.array(h) != np.array(y)) for h in yt]
    idx = accpl.index( np.min(accpl) )
    P[idx] = True
    #
    flag = True  # whether the error of H* on S can be reduced
    nb_count = int(np.ceil(rho * nb_cls))  # = nb_pru
    #
    uY = np.unique(y).tolist()
    if len(uY) == 2 and 0 in uY and 1 in uY:
        tr_y = (np.array(y) * 2 - 1).tolist()
        tr_yt = (np.array(yt) * 2 - 1).tolist()
    elif len(uY) == 2 and -1 in uY and 1 in uY:
        tr_y = deepcopy(y)
        tr_yt = deepcopy(yt)
    else:  # len(uY) > 2
        tr_y = (np.array(y) - len(uY) // 2).tolist()
        tr_yt = (np.array(yt) - len(uY) // 2).tolist()
    del uY
    #
    while nb_count > 0:  # >= 
        hstar = np.array(tr_yt)[P].tolist()
        hstar = DREP_fxH(hstar)
        all_q_in_S = np.where(P == False)[0]
        dhstar = [ DREP_diff(tr_yt[q], hstar)  for q in all_q_in_S]
        dhidx = np.argsort(dhstar).tolist()  # sort in the ascending order
        tradeoff = int(np.ceil(rho * len(all_q_in_S)))
        gamma = dhidx[: tradeoff]  # index in Gamma
        gamma = [ all_q_in_S[q]  for q in gamma]
        # 
        errHstar = np.mean(np.array(hstar) != np.array(tr_y))
        idx = np.where(P == True)[0].tolist()
        errNew = [ np.mean(np.array( DREP_fxH(np.array(tr_yt)[idx+[p]].tolist()) ) != np.array(tr_y))  for p in gamma]
        errIdx = errNew.index( np.min(errNew) )
        if errNew[errIdx] <= errHstar:
            P[gamma[errIdx]] = True
            flag = True
        else:
            flag = False
        #
        del hstar,all_q_in_S,dhstar,dhidx,tradeoff, errHstar,errNew,errIdx
        nb_count -= 1
        if flag == False:
            break
    #   #   #
    yo = np.array(yt)[P].tolist()
    P = P.tolist()
    del idx,flag,accpl, tr_y,tr_yt
    gc.collect()
    return deepcopy(yo), deepcopy(P)



#----------------------------------
#
#----------------------------------



#==================================
# \citep{qian2015pareto}
#
# Pareto Ensemble Pruning (PEP)  (AAAI-15)  [Binary/ multi-class classification, Bagging]
#==================================
#



#----------------------------------
# assume
#----------------------------------
#
# $D = \{(\mathbf{x}_i, y_i)\} _{i=1}^m$        data set, with $m$ samples
# $H = \{h_j\} _{j=1}^n$                        set of trained individual classifiers, with $n$ ones
# $H_{\mathbf{s}}$ with $\mathbf{s} = \{0, 1\}^n$   pruned sub-ensemble, with a selector vector
#       $|\mathbf{s}| = \sum_{j=1}^n s_j$           minimize the size of $H_{\mathbf{s}}$
#
# ${\arg\min}_{ \mathbf{s} \in \{0, 1\}^n } \bigg( f(H_{\mathbf{s}}), |\mathbf{s| \bigg)$ 
#                                                   bi-objective ensemble pruning problem
# Note that: y\in {-1, +1},  transform!



# a pruned ensemble $H_{\mathbf{s}}$ is composited as 
# multi-class classification
# 
def PEP_Hs_x(y, yt, s):
    vY = np.unique(np.vstack((y, yt)))
    dY = len(vY)
    yt = np.array(yt)
    s = np.array(np.mat(s).T)
    #
    vote = [np.sum(s*(yt==i), axis=0).tolist() for i in vY]
    loca = np.array(vote).argmax(axis=0)
    Hsx = [vY[i] for i in loca]
    #
    del vY,dY, yt,s, vote,loca
    gc.collect()
    return deepcopy(Hsx)


# define the difference of two classifiers as
# [works for multi-class, after modification] 
#
def PEP_diff_hihj(hi, hj):
    tem = np.array(hi) == np.array(hj)
    tem = np.array(tem, dtype=DTY_FLT) * 2 - 1
    return np.mean( (1. - tem) / 2. )

# and the error of one classifier as 
# [works for multi-class now]
#
def PEP_err_hi(y, hi):
    tem = np.array(hi) == np.array(y)
    tem = np.array(tem, dtype=DTY_FLT) * 2 - 1
    return np.mean( (1. - tem) / 2. )

# both of them (i.e., \diff(hi,hj), \err(hi)) belong to [0, 1]. 
# If \diff(hi,hj) = 0 (or 1), hi and hj always make the same (or opposite) prediction; 
# If \err(hi) = 0 (or 1), hi always make the right (or wrong) prediction. 
# We hope: (1) \diff larger, (2) \err small


# the validation error is calculated as 
# binary  [I think multi-class classification]
# 
def PEP_f_Hs(y, yt, s):
    Hsx = PEP_Hs_x(y, yt, s)
    ans = np.mean(np.array(Hsx) != np.array(y))
    return ans, deepcopy(Hsx)



#----------------------------------
# performance objective $f$
#----------------------------------


# def PEP_objective_performance():
#     pass



#----------------------------------
# evaluation criterion $eval$
#----------------------------------



#----------------------------------
# OEP, SEP
#----------------------------------


# generate $\mathbf{s'}$ by flipping each bit of $\mathbf{s}$ with prob.$\frac{1}{n}$
# 
def PEP_flipping_uniformly(s):
    n = len(s)
    pr = np.random.uniform(size=n)  # \in [0, 1]
    pr = (pr < 1. / n)  # <= 
    # s', sprime
    sp = [1-s[i] if pr[i] else s[i] for i in range(n)]
    del n, pr
    gc.collect()
    return deepcopy(sp)



#----------------------------------
# Domination
#----------------------------------


# bi-objective
#
def PEP_bi_objective(y, yt, s):
    fHs, Hsx = PEP_f_Hs(y, yt, s)
    s_ab = np.sum(s)  # absolute
    ans = (fHs, s_ab)
    del fHs, Hsx, s_ab
    gc.collect()
    return deepcopy(ans)


#' ''
# the objective vector: 
# $\mathbf{g}(\mathbf{s}) = (g_1, g_2)$
# $\mathbf{g}: \mathcal{S} \to \mathbb{R}^2$
# 
# for two solutions 
# $\mathbf{s}, \mathbf{s'} \in \mathcal{S}$
#' ''

# (1) s weakly dominate s'  if g1(s)<=g1(s') and g2(s)<=g2(s')
#
def PEP_weakly_dominate(g_s1, g_s2):
    if (g_s1[0] <= g_s2[0]) and (g_s1[1] <= g_s2[1]):
        return True
    return False

# (2) s dominate s'  if s \succeq_{g} s'  and either g1(s)<g1(s') or g2(s)<g2(s')
#
def PEP_dominate(g_s1, g_s2):
    if PEP_weakly_dominate(g_s1, g_s2) == True:
        if g_s1[0] < g_s2[0]:
            return True
        elif g_s1[1] < g_s2[1]:
            return True
        else:
            return False
    return False



#----------------------------------
# OEP, SEP
#----------------------------------


# Simple Ensemble Pruning
#
def PEP_SEP(yt, y, nb_cls, rho):
    # nb_cls = len(yt)  # n
    # s = np.random.randint(2, size=nb_cls).tolist()
    tem_s = np.random.uniform(size=nb_cls)  # \in [0, 1]
    tem_i = (tem_s <= rho)
    if np.sum(tem_i) == 0:
        tem_i[ np.random.randint(nb_cls) ] = True
    s = np.zeros(nb_cls, dtype=DTY_INT)
    s[tem_i] = 1
    s = s.tolist()
    del tem_s, tem_i
    # 万一 tem_i = [] 呢？  solved
    #
    nb_pru = int(np.ceil(rho * nb_cls))
    nb_count = nb_pru
    while nb_count >= 0:
        sp = PEP_flipping_uniformly(s)
        f1, _ = PEP_f_Hs(y, yt, sp)
        f2, _ = PEP_f_Hs(y, yt, s)
        if f1 <= f2:
            s = deepcopy(sp)
        #   #
        nb_count = nb_count - 1  # nb_count -= 1
        del sp, f1, f2
        if np.sum(s) > nb_pru:
            break
    #   #   #
    yo = np.array(yt)[np.array(s) == 1].tolist()
    P = np.array(s, dtype=DTY_BOL).tolist()
    del nb_count, s
    gc.collect()
    return deepcopy(yo), deepcopy(P)


# Ordering-based Ensemble Pruning
#
def PEP_OEP(yt, y, nb_cls):
    # nb_cls = len(yt)
    Hs = np.zeros(nb_cls, dtype=DTY_BOL).tolist()
    ordered_idx = []
    while np.sum(Hs) < nb_cls:
        Hu_idx = np.where(np.array(Hs) == False)[0].tolist()
        obj_f = []
        for h in Hu_idx:
            tem_s = deepcopy(Hs)
            tem_s[h] = True  # tem_idx
            tem_ans, _ = PEP_f_Hs(y, yt, tem_s)
            obj_f.append(tem_ans)
            del tem_s, tem_ans
        idx_f = obj_f.index( np.min(obj_f) )
        idx_f = Hu_idx[idx_f]
        ordered_idx.append( idx_f )
        Hs[idx_f] = True
        del Hu_idx, obj_f, idx_f
    del Hs
    #
    obj_eval = []
    for h in range(1, nb_cls + 1):
        tem_s = np.zeros(nb_cls, dtype=DTY_INT)
        tem_s[ordered_idx[: h]] = 1
        tem_ans, _ = PEP_f_Hs(y, yt, tem_s.tolist())
        obj_eval.append(tem_ans)
        del tem_s, tem_ans
    idx_k = obj_eval.index( np.min(obj_eval) )
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    P[ ordered_idx[: (idx_k + 1)] ] = True  # Notice!!
    del obj_eval, idx_k  #, nb_cls
    #
    yo = np.array(yt)[P].tolist()
    P = P.tolist()
    gc.collect()
    return deepcopy(yo), deepcopy(P)



#----------------------------------
# VDS, PEP
#----------------------------------


# VDS Subroutine
#
def PEP_VDS(y, yt, nb_cls, s):
    # nb_cls = len(s)
    QL = np.zeros(nb_cls, dtype=DTY_BOL)
    sp = deepcopy(s)  # $\mathbf{s}$
    # initial
    Q = [];     L = []
    # Let N(.) denote the set of neighbor solutions of a binary vector with Hamming distance 1. 
    # Ns = [deepcopy(sp) for i in range(nb_cls)]
    # for i in range(nb_cls):
    #     Ns[i][i] = 1 - Ns[i][i]
    # Ns = np.array(Ns)
    while np.sum(QL) < nb_cls:
        # Let N(.) denote the set of neighbor solutions of a binary vector with Hamming distance 1.
        Ns = [deepcopy(sp) for i in range(nb_cls)]
        for i in range(nb_cls):
            Ns[i][i] = 1 - Ns[i][i]
        Ns = np.array(Ns)
        #
        idx_Vs = np.where(QL == False)[0]
        Vs = Ns[idx_Vs].tolist()
        # an objective $f: 2^H \mapsto \mathbb{R}$
        obj_f = [PEP_f_Hs(y, yt, i)[0] for i in Vs]  # obj_f = [PEP_f_Hs(y, yt, s)[0] for i in Vs] 
        idx_f = obj_f.index( np.min(obj_f) )
        yp = Vs[idx_f]  # $\mathbf{y}$
        Q.append( deepcopy(yp) )
        L.append( idx_Vs[idx_f] )
        QL[ idx_Vs[idx_f] ] = True
        sp = deepcopy(yp)
        #
        del Ns, idx_Vs, Vs, obj_f, idx_f, yp
    del QL  #, nb_cls
    gc.collect()
    return deepcopy(Q), deepcopy(L)


# Pareto Ensemble Pruning
# 
def PEP_PEP(yt, y, nb_cls, rho):
    # nb_cls = len(yt)
    s = np.random.randint(2, size=nb_cls).tolist()
    P = [];     P.append( deepcopy(s) )
    #
    nb_count = int(np.ceil(rho * nb_cls))  # = nb_pru
    while nb_count > 0:
        idx = np.random.randint( len(P) )
        s0 = P[idx]
        sp = PEP_flipping_uniformly(s0)
        g_sp = PEP_bi_objective(y, yt, sp)
        #
        flag1 = False
        for z1 in P:
            g_z1 = PEP_bi_objective(y, yt, z1)
            if PEP_dominate(g_z1, g_sp) == True:
                flag1 = True
                del g_z1
                break
            else:
                del g_z1
        del z1
        #
        if flag1 == False:
            idx1 = []
            for i in range(len(P)):
                g_z2 = PEP_bi_objective(y, yt, P[i])
                if PEP_weakly_dominate(g_sp, g_z2) == True:
                    idx1.append(i)
                del g_z2
            for i in idx1[::-1]:
                del P[i]
            P.append( deepcopy(sp) )
            del i, idx1
            #
            Q, _ = PEP_VDS(y, yt, nb_cls, sp)
            for q in Q:
                g_q = PEP_bi_objective(y, yt, q)
                flag3 = False
                for z3 in P:
                    g_z3 = PEP_bi_objective(y, yt, z3)
                    if PEP_dominate(g_z3, g_q) == True:
                        flag3 = True
                        del g_z3
                        break
                    else:
                        del g_z3
                del z3
                #
                if flag3 == False:
                    idx3 = []
                    for j in range(len(P)):
                        g_z4 = PEP_bi_objective(y, yt, P[j])
                        if PEP_weakly_dominate(g_q, g_z4) == True:
                            idx3.append(j)
                        del g_z4
                    for j in idx3[::-1]:
                        del P[j] 
                    P.append( deepcopy(q) )
                    del j, idx3
                #   #
                del flag3, g_q
            del q, Q
        del flag1, g_sp, sp, s0, idx
        nb_count = nb_count - 1
        # end of this iteration
    del nb_count, s
    #
    obj_eval = [PEP_f_Hs(y, yt, t)[0] for t in P]
    idx_eval = obj_eval.index( np.min(obj_eval) )
    s = P[idx_eval]
    del P, obj_eval, idx_eval
    P = np.array(s, dtype=DTY_BOL)
    if np.sum(P) == 0:
        P[ np.random.randint(nb_cls) ] = True
    P = P.tolist()
    yo = np.array(yt)[P].tolist()
    del s  #, nb_cls
    gc.collect()
    return deepcopy(yo), deepcopy(P)



#----------------------------------
# Modification of mine (to speed up)
#----------------------------------


def PEP_PEP_modify(yt, y, nb_cls, rho):
    # nb_cls = len(yt)
    nb_pru = int(np.ceil(rho * nb_cls))
    s = np.random.randint(2, size=nb_cls).tolist()
    P = [ deepcopy(s) ]
    del s
    # 
    nb_count = nb_pru
    while nb_count > 0:
        idx = np.random.randint( len(P) )
        s = P[idx]
        sp = PEP_flipping_uniformly(s)
        g_sp = PEP_bi_objective(y, yt, sp)
        #
        flag1 = False
        for z1 in P:
            g_z1 = PEP_bi_objective(y, yt, z1)
            if PEP_dominate(g_z1, g_sp) == True:
                flag1 = True
                break
        if flag1 == False:
            idx1 = []
            for i in range(len(P)):
                g_z1 = PEP_bi_objective(y, yt, P[i])
                if PEP_weakly_dominate(g_sp, g_z1) == True:
                    idx1.append(i)
            for i in idx1[::-1]:
                del P[i]
            P.append( deepcopy(sp) )
            del i, idx1
            #
            Q, _ = PEP_VDS(y, yt, nb_cls, sp)
            for q in Q:
                g_q = PEP_bi_objective(y, yt, q)
                flag3 = False
                for z3 in P:
                    g_z3 = PEP_bi_objective(y, yt, z3)
                    if PEP_dominate(g_z3, g_q) == True:
                        flag3 = True
                        break
                if flag3 == False:
                    idx3 = []
                    for j in range(len(P)):
                        g_z3 = PEP_bi_objective(y, yt, P[j])
                        if PEP_weakly_dominate(g_q, g_z3) == True:
                            idx3.append(j)
                    for j in idx3[::-1]:
                        del P[j]
                    P.append( deepcopy(q) )
                    del j, idx3
                #   #
                del g_z3, z3, flag3, g_q
            del q, Q
        del g_z1, z1, flag1, g_sp, sp, s, idx
        #
        nb_count = nb_count - 1
        # end of this iteration
        # 
        obj_eval = [ PEP_f_Hs(y, yt, s)[0]  for s in P]
        idx_eval = obj_eval.index( np.min(obj_eval) )
        s_ef = P[idx_eval]  # se/sf, s_eventually, s_finally
        del obj_eval, idx_eval
        if (np.sum(s_ef) <= nb_pru) and (np.sum(s_ef) > 0):
            break
    #   #
    P_ef = np.array(s_ef, dtype=DTY_BOL)
    if np.sum(P_ef) == 0:
        P_ef[ np.random.randint(nb_cls) ] = True
    yo = np.array(yt)[P_ef].tolist()
    P_ef = P_ef.tolist()
    del s_ef, nb_count, P, nb_pru  #, nb_cls
    gc.collect()
    return deepcopy(yo), deepcopy(P_ef)



#----------------------------------
#
#----------------------------------



#==================================
# Overall interface
#==================================


#----------------------------------
# Overall interface
#----------------------------------


# def comparative_compared_pruning_method(name_pru, yt, y, nb_cls, nb_pru, rho=None, epsilon=1e-3):
# def contrastive_pruning_according_validation(name_pru, yt, y, nb_cls, nb_pru, rho=None, epsilon=1e-3):
def existing_contrastive_pruning_method(name_pru, yt, y, nb_cls, nb_pru, rho=None, epsilon=1e-3):
    #@ params:  name_prune
    # self._name_pru_set = ['ES','KL','KL+','KP','OO','RE', 'GMM','LCS', 'DREP','SEP','OEP','PEP','PEP+']
    rho = nb_pru / nb_cls if not rho else rho
    # epsilon = 1e-6
    # print("rho     = {}\nepsilon = {}".format(rho, epsilon))
    # 
    if name_pru == 'ES':
        yo, P       = Early_Stopping(              yt,    nb_cls, nb_pru)
    elif name_pru == 'KL':
        yo, P       = KL_divergence_Pruning(       yt,    nb_cls, nb_pru)
    elif name_pru == 'KL+':
        yo, P       = KL_divergence_Pruning_modify(yt,    nb_cls, nb_pru)
    elif name_pru == 'KP':
        yo, P       = Kappa_Pruning(               yt, y, nb_cls, nb_pru)
    elif name_pru == 'OO':
        yo, P, flag = Orientation_Ordering_Pruning(yt, y)
    elif name_pru == 'RE':
        yo, P       = Reduce_Error_Pruning(        yt, y, nb_cls, nb_pru)
    elif name_pru == 'GMM':  # 'GMM_Algorithm'
        yo, P       = GMM_Algorithm(               yt, y, nb_cls, nb_pru)
    elif name_pru == 'LCS':  # 'Local_Search':
        yo, P       = Local_Search(                yt, y, nb_cls, nb_pru, epsilon)
    elif name_pru == 'DREP':
        yo, P       = DREP_Pruning(                yt, y, nb_cls,         rho)
    elif name_pru == 'SEP':
        yo, P       = PEP_SEP(                     yt, y, nb_cls,         rho)
    elif name_pru == 'OEP':
        yo, P       = PEP_OEP(                     yt, y, nb_cls)
    elif name_pru == 'PEP':
        yo, P       = PEP_PEP(                     yt, y, nb_cls,         rho)
    elif name_pru == 'PEP+':
        yo, P       = PEP_PEP_modify(              yt, y, nb_cls,         rho)
    else:
        raise UserWarning("LookupError! Check the `name_prune`.")
    #
    if name_pru != 'OO':
        flag = None
    P = np.where(np.array(P) == True)[0].tolist()
    return deepcopy(yo), deepcopy(P), flag



#----------------------------------
#
#----------------------------------




#==================================
#
#==================================


#----------------------------------
#
#----------------------------------



