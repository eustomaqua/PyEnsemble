# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
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
"""


