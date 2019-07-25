# coding: utf8
# Aim to: some calculation of entropy, (existing methods)
# Available for:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from copy import deepcopy
# garbage collector
import gc
gc.enable()
import time

import numpy as np 
from scipy import stats
# import multiprocessing as mp 
from pathos import multiprocessing as pp 
# from pympler.asizeof import asizeof


from utils_constant import DTY_FLT
from utils_constant import DTY_INT
from utils_constant import DTY_BOL

from utils_constant import CONST_ZERO
from utils_constant import check_zero

from utils_constant import GAP_INF
from utils_constant import GAP_MID
from utils_constant import GAP_NAN



#==========================================
# previously on ``distributed.py``
#==========================================



#----------  Convert data  -----------


# minimum description length
#
def binsMDL(data, nb_bin=5):  # bins5MDL
    # Let `U' be a set of size `d' of labelled instances
    # accompanied by a large set of features `N' with cardinality `n',
    # represented in a `dxn' matrix. 
    #
    data = np.array(data, dtype=DTY_FLT)
    d = data.shape[0]  # number of samples
    n = data.shape[1]  # number of features
    #
    for j in range(n):  # By Feature
        fmin = np.min(data[:,j]);   fmax = np.max(data[:,j])
        fgap = (fmax - fmin) / nb_bin
        trans = data[:, j]
        #
        idx = (data[:,j] == fmin);  trans[idx] = 0
        pleft = fmin;   pright = fmin + fgap
        #
        for i in range(nb_bin):
            idx = ((data[:,j] > pleft) & (data[:,j] <= pright))
            trans[idx] = i
            pleft += fgap;  pright += fgap
        #   #
        data[:, j] = deepcopy(trans)
    #   #
    data = np.array(data, dtype=DTY_INT).tolist()
    del d,n, i,j,fmin,fmax,fgap, trans,idx,pleft,pright
    gc.collect()
    return deepcopy(data)  # list



#----------  Probability of Discrete Variable  -----------


# probability of one vector
#
def prob(X):
    X = np.array(X)
    vX = np.unique(X).tolist(); dX = len(vX)
    px = np.zeros(dX)
    for i in range(dX):
        px[i] = np.mean(X == vX[i])
    px = px.tolist()
    del i, X, dX
    gc.collect()
    return deepcopy(px), deepcopy(vX)  # list


# joint probability of two vectors
#
def jointProb(X, Y):
    X = np.array(X);    Y = np.array(Y)
    vX = np.unique(X).tolist(); vY = np.unique(Y).tolist()
    dX = len(vX);   dY = len(vY)
    pxy = np.zeros((dX, dY))
    for i in range(dX):
        for j in range(dY):
            # pxy[i, j] = np.mean((X == vX[i]) & (Y == vY[j]))
            #
            # print("jointProb -- X=vX[i]: ", X==vX[i])
            # print("jointProb -- Y=vY[j]: ", Y==vY[j])
            # print("jointProb -- X & Y  : ", i, j, (X==vX[i])&(Y==vY[j]) )
            # print("jointProb -- X & Y :", i, j, X, Y)
            tem = (X == vX[i]) & (Y == vY[j])
            # print("jointProb -- X & Y  : ", i, j, tem, end="")
            tem = np.mean(tem)
            # print("\tjointProb -- X & Y  : ", i, j, tem )
            pxy[i, j] = tem
    pxy = pxy.tolist()
    del dX, dY, i,j,X,Y
    gc.collect()
    return deepcopy(pxy), deepcopy(vX), deepcopy(vY)  # list



#----------  Shannon Entropy  -----------
#
# calculate values of entropy
# H(.) is the entropy function and p(.,.) is the joint probability


# for a scalar value
#
def H(p):
    # if p == 0.:
    #     return 0. 
    # return (-1.) * p * np.log2(p)
    return -1. * p * np.log2(p) if p != 0. else 0. 


# H(X), H(Y):  for one vector
#
def H1(X):
    px, _ = prob(X)
    # calc 1,3
    ans = 0. 
    for i in px:
        ans += H(i)
    #   #
    # i = None #-1
    del px  #, i
    gc.collect()
    return ans


# H(X, Y):  for two vectors
#
def H2(X, Y):
    pxy, _, _ = jointProb(X, Y)
    # calc 2
    ans = 0. 
    for i in pxy:
        for j in i:
            ans += H(j)
    #   #   #
    # i = j = None #-1
    del pxy  #, i,j
    gc.collect()
    return ans



#----------    -----------
#----------    -----------



#------------------------------------------
# 
#------------------------------------------



#==========================================
# Inspired by zadeh2017diversity
#==========================================



# I(.;.) is the mutual information function
# I(X; Y)
#
def I(X, Y):
    px, _ = prob(X);    py, _ = prob(Y)
    pxy, _, _ = jointProb(X, Y)
    #
    # calc 1
    ans = 0. 
    for i in range(len(px)):
        for j in range(len(py)):
            #
            tem = pxy[i][j] / check_zero( px[i] * py[j] )
            tem = pxy[i][j] * np.log2( check_zero(tem) )
            ans += tem
    #   #   #
    # i = j = None
    del px, py, pxy  #, i, j, tem
    gc.collect()
    return ans


# MI(X, Y): The normalized mutual information of two discrete random variables X and Y
#
def MI(X, Y):
    tem = np.sqrt(H1(X) * H1(Y))
    return I(X, Y) / check_zero(tem)

# VI(X, Y): The normalized variation of information of two discrete random variables X and Y
#
def VI(X, Y):
    tem = H2(X, Y)
    return 1. - I(X, Y) / check_zero(tem)


# For two feature vectors like p and q, and the class label vector L, define DIST(p,q) as follows:
#
def DIST(X, Y, L, lam):  # lambda
    if X == Y:  # list
        return 0. 
    tem = MI(X, L) + MI(Y, L)
    return lam * VI(X, Y) + (1. - lam) * tem / 2.


# S \subset or \subseteq N, N is the set of all features and |S|=k. 
# We want to maximize the following objective function (as the objective of 
# diversity maximization problem) for `S` \subset `N` and |S|=k
#

def DIV1(S, L, lam):
    S = np.array(S);    k = S.shape[1]
    ans = [[ DIST(S[:,i].tolist(), S[:,j].tolist(), L, lam)  for j in range(k)] for i in range(k)]
    ans = np.sum(ans) / 2.
    del S, k
    gc.collect()
    return ans

def DIV2(S, L, lam):
    S = np.array(S);    k = S.shape[1]
    ans1 = [[ VI(S[:,i].tolist(), S[:,j].tolist())  for j in range(k)] for i in range(k)]
    ans1 = np.sum(ans1)
    ans2 = [ MI(S[:,i].tolist(), L)  for i in range(k)]
    ans2 = np.sum(ans2)
    ans = ans1 * lam / 2. + ans2 * (1. - lam) * (k - 1.) / 2. 
    del S, k, ans1, ans2
    gc.collect()
    return ans



#----------  Algorithm (DDisMI ->) Greedy  -----------


def dist_sum(p, S, L, lam):
    S = np.array(S);    n = S.shape[1]
    # calc 1
    ans = 0. 
    for i in range(n):
        tem = DIST(p, S[:,i].tolist(), L, lam)
        ans += tem
    del S, n, i
    gc.collect()
    return ans


# T is the set of points/features; S = [True, False] represents 
# this one is in S or not, and S is selected features. 
#
def arg_max_p(T, S, L, lam):
    T = np.array(T);    S = np.array(S)
    # 
    # calc 3
    all_q_in_S = T[:, S].tolist()
    idx_p_not_S = np.where(S == False)[0]
    if len(idx_p_not_S) == 0:
        del T,S, all_q_in_S, idx_p_not_S
        return -1  # idx = -1
    #   #
    ans = [ dist_sum(T[:,i].tolist(), all_q_in_S, L, lam)  for i in idx_p_not_S]
    idx_p = ans.index( np.max(ans) )
    idx = idx_p_not_S[idx_p]
    #
    del T,S, all_q_in_S, idx_p_not_S, idx_p, ans
    gc.collect()
    return idx



#----------  Algorithm Greedy  -----------


# T:    set of points / features
# k:    number of selected features
#
def Greedy(T, k, L, lam):
    T = np.array(T);    n = T.shape[1]
    S = np.zeros(n, dtype=DTY_BOL)
    p = np.random.randint(0, n)
    S[p] = True
    for i in range(1, k):
        idx = arg_max_p(T, S, L, lam)
        if idx > -1:
            S[idx] = True  # 1
    S = S.tolist()
    del T, n, p  #, i, idx
    gc.collect()
    return deepcopy(S)  # list



#----------  Algorithm (DDisMI ->) DDisMI  -----------
#
# N:    set of features (d samples, n features, dxn matrix)
# k:    number of selected features
# m:    number of machines



def choose_proper_platform(nb, pr):
    m = int(np.round(np.sqrt(1. / pr)))
    k = np.max([int(np.round(nb * pr)), 1])
    while k * m >= nb:
        m = np.max([m - 1, 1])
        if m == 1:
            break
    # m = np.max([m, 2])
    return k, m


def randomly_partition(n, m):
    randseed = int(time.time() * GAP_MID % GAP_INF)
    prng = np.random.RandomState(randseed)
    tem = np.arange(n)
    prng.shuffle(tem)
    idx = np.zeros(n, dtype=DTY_INT) - 1  # init  # np.ones(n) - 2
    #
    if n % m != 0:
        # floors and ceilings
        floors = int(np.floor(n / float(m)))
        ceilings = int(np.ceil(n / float(m)))
        # modulus and mumble 
        modulus = n - m * floors
        mumble = m * ceilings - n
        # mod: n % m
        # 
        for k in range(modulus):
            ij = tem[k * ceilings : (k + 1) * ceilings]
            idx[ij] = k
        ijt = ceilings * modulus  # range(modulus, modulus + mumble):  # m = modulus + mumble
        for k in range(mumble):
            ij = tem[k * floors + ijt : (k + 1) * floors + ijt]
            idx[ij] = k + modulus
        #
        del floors, ceilings,  modulus, mumble,  k, ij, ijt
    else:
        ijt = int(n / m)
        for k in range(m):
            ij = tem[k * ijt : (k + 1) * ijt]
            idx[ij] = k
        del ijt, ij, k
    # 
    idx = idx.tolist()
    gc.collect()
    return deepcopy(idx)



#----------  Algorithm DDisMI  -----------


# Group / Machine i-th
#
def find_idx_in_sub(i, Tl, N,k,L,lam):
    sub_idx_in_N = np.where(Tl == i)[0]  # or np.argwhere(Tl == i).T[0]
    sub_idx_greedy = Greedy(N[:, (Tl == i)].tolist(), k, L, lam)
    sub_idx_greedy = np.where( np.array(sub_idx_greedy) == True)[0]
    ans = sub_idx_in_N[sub_idx_greedy]
    del sub_idx_in_N, sub_idx_greedy
    gc.collect()
    return deepcopy(ans)  # np.array (ans, Tl)


# def DDisMI_MultiPool(N, k, m, L, lam):
#     pass
#
def DDisMI(N, k, m, L, lam):
    N = np.array(N);    n = N.shape[1]
    # S = np.zeros(n, dtype=np.bool)
    Tl = randomly_partition(n=n, m=m); Tl = np.array(Tl)
    Sl = np.zeros(n, dtype=DTY_INT) - 1  # init
    #
    # define lambda function
    # 
    # concurrent selection
    pool = pp.ProcessingPool(nodes = m)
    sub_idx = pool.map(find_idx_in_sub, list(range(m)), [Tl]*m, [N]*m, [k]*m, [L]*m, [lam]*m)
    del pool, Tl
    #
    for i in range(m):
        Sl[ sub_idx[i] ] = i
    del sub_idx
    sub_all_in_N = np.where(Sl != -1)[0]
    sub_all_greedy = Greedy(N[:, (Sl != -1)].tolist(), k, L, lam)
    sub_all_greedy = np.where( np.array(sub_all_greedy) == True)[0]
    #
    final_S = np.zeros(n, dtype=DTY_BOL)
    final_S[ sub_all_in_N[ sub_all_greedy ] ] = 1
    del sub_all_in_N, sub_all_greedy
    # 
    div_temS = DIV1(N[:, final_S].tolist(), L, lam)
    div_Sl = [ DIV1(N[:, (Sl == i)].tolist(), L, lam)  for i in range(m)]
    if np.sum(np.array(div_Sl) > div_temS) >= 1:
        tem_argmax_l = div_Sl.index( np.max(div_Sl) )
        final_S = (Sl == tem_argmax_l)
        del tem_argmax_l
    # 
    del div_temS, div_Sl, N,n,m,Sl
    final_S = final_S.tolist()
    gc.collect()
    return deepcopy(final_S)



# if you want to do ``Serial Execution``, just to do:
# S = Greedy(N, k, L, lam)



#----------    -----------
#----------    -----------



#------------------------------------------
#
#------------------------------------------




#==========================================
#
#==========================================


#------------------------------------------
#
#------------------------------------------


