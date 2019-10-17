# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
import gc
gc.enable()

import numpy as np

from pyensemble.utils_const import check_zero



#==================================
# \citep{martine2006pruning}
#
# Pruning in Ordered Bagging Ensembles (ICML-06)
# [multi-class classification, Bagging]
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
# the signature vector of the classifier $h_t$, for the dataset $L_{tr}$ composed
# of $N_{tr}$ examples
#
def signature_vector(ht, y):
    y = np.array(y);    ht = np.array(ht)
    ct = 2. * (y == ht) - 1.
    ans = ct.tolist()
    del y, ht, ct
    gc.collect()
    return deepcopy(ans)


# $c_{ti}$ is equal to +1 if $h_t$ (the t-th unit in the ensemble) correctly classifies
# the i-th example of $L_{tr}$
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



# The classifiers are ordered by increasing values of the angle between the signature
# vectors of the individual classifiers and the reference vector $\mathbf{c}_{ref}$
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


