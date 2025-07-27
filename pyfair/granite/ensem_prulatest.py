# coding: utf-8
#
# Target:
#   Existing pruning methods in ensemble methods,
#   with the pruned order
#
# Including:
#   1) Ranking based / Ordering based
#   2) Clustering based
#   3) Optimization based
#   4) Other
#   5) Ameliorate of mine (my own version)
#   6) Composable Core-sets, Diversity Maximization
#


from copy import deepcopy
import gc
import time
import numpy as np
from pympler.asizeof import asizeof

from pyfair.facil.utils_const import check_zero, DTY_INT, DTY_BOL
from pyfair.facil.ensem_voting import plurality_voting
from pyfair.marble.data_entropy import I as EntropyI
gc.enable()


# =========================================
# \citep{zhang2017ranking}
# \citep{zhang2020selective}
# \citep{zhang2019pruning}
# =========================================


# =========================================
# \citep{zhang2017ranking}
# Title : A ranking-based strategy to prune variable
#         selection ensembles
# Method: PST2E
# =========================================


# ----------------------------------
# Brief introduction of stochastic stepwise ensembles (ST2E)
# Algorithm 1:
#   The stochastic stepwise (ST2) algorithm for variable se-
#   lection
#

# Input : \mathbf{y}  nx1 vector
#         \mathbf{X}  nxp design matrix
#         \lambda     a parameter to control group size (i.e.,
#                     the number of variables in a group)
#         \kappa      a parameter to control number of groups
#                     to assess
# Output: Subset \mathcal{S}  a set consists of the selected
#                             variables

# Main steps of ST2
#
# 1. Initialize \mathcal{S}=\emptyset
#    the candidate set \mathcal{C} = {X1, X2, ..., Xp}
#    the AIC vlaue initial_aic = 10^5, `improve` = 1
# 2. While `improve` > 0
#   2a. Implement `forward selection` step:
#       [add_aic, add_var] = ForwardSelect(
#                               y, X, S, C, lambda, kappa)
#   2b. Let S=S\cup{add_var}, C=C\setminus{add_var},
#       `improve` = initial_aic - add_aic, initial_aic=add_aic
#   2c. If S\neq\emptyset, implement `backward selection` step:
#       [del_aic, del_var] = BackwardSelect(
#                               y, X, S, lambda, kappa)
#   2d. Let S=S\setminus{del_var}, C=C\cup{del_var},
#       `improve` = initial_aic - del_aic, initial_aic=del_aic
# 3. EndWhile


# ----------------------------------
# PST2E: a ranking-based strategy to prune ST2E
# Algorithm 2:
#   The ST2E algorithm for variable selection
#

# Input : All inputs of ST2 algorithm (\ie, y, X, lambda and
#                                           kappa)
#         B,  ensemble size.
# Output: A matrix \mathbf{E},  a matrix of order Bxp storing
#                               the results of each individual.

# Main steps of ST2E
#
# 1. Initialize a matrix \mathbf{E} of order Bxp with all ele-
#    ments being 0.
# 2. For b = 1,2,...,B
#   2a. Implement ST2 algorithm with y and X:
#       S = ST2(y, X, lambda, kappa)
#   2b. Let
#       \mathbf{E}(b,j) = { 1,  if X_j\in S ;    j=1,2,...,p
#                         { 0,  if X_j\notin S,
# 3. EndFor


# ----------------------------------
# PST2E:
# Algorithm 3:
#   The novel algorithm PST2E for variable ranking and selection
#

# Input : All inputs of ST2E algorithm (i.e.,
#                                   y,X, lambda, kappa, and B)
#         U,  size of pruned subensemble
# Output: A matrix \mathbf{E}',  a matrix of order Uxp storing
#                                the results of each individual
#                                in the pruned ensemble

# Main steps of PST2E
#
# 1. Initialize a matrix \mathbf{E} of order Bxp with all elements
#    being 0.
# 2. Execute ST2E algorithm and store the results in \mathbf{E}:
#       \mathbf{E} = ST2E(y, X, lambda, kappa, B)
# 3. For the b-th (b = 1,2,...,B) ensemble member,
#       use the selected variables to build a linear regression and
#       compute its prediction error Err(b) as
#           Err(b) = \frac{1}{n} \sum_{i=1}^n (
#                           y_i - \hat{y}_i(b)
#                   )^2 , t=1,2,...,B
#       where \hat{y}_i(b) stands for the estimated value for the
#       i-th instance predicted by the t-th model.
# 4. Sort all the ensemble memebers in line with Err(b) (b = 1,2,...,B)
#    in descending order.
# 5. Select the U members sorted on top of the ranked list to construct
#    a subensemble. In other words, arrange the U rows of \mathbf{E}
#    ranked ahead into a new matrix \mathbf{E}'.


# ----------------------------------


# =========================================
# \citep{zhang2020selective}
# Title : On selective learning in stochastic stepwise ensemble
# Method: Pruned-ST2E
# =========================================


# ----------------------------------
# Brief introduction of stochastic stepwise ensembles (ST2Es)
#


# ----------------------------------
# Pruned-ST2E: novel technique to prune ST2E
# Algorithm 1:
#   The proposed Pruned-ST2E algorithm
#

# Input :  \mathbf{y}  nx1 vector
#          \mathbf{X}  nxp design matrix
#          \lambda     a parameter in ST2 to control group size
#                      in a step
#          \kappa      a parameter in ST2 to control number of
#                      groups to assess
#          B           size of full ensemble
#          U           size of pruned subensemble
# Output:  Average importance measure computed as
#               R(j) = \frac{1}{U} \sum_{r=1}^U \mathbf{E}'(r,j),
#                                 j=1,2,...,p.

# Main steps of Pruned-ST2E
#
# 1. Initialize a matrix \mathbf{E} of order Bxp with all elements
#    being 0.
# 2. For b = 1,2,...,B
#   2a. Provide \mathbf{y} and \mathbf{X} as the input of the ST2
#       algorithm to perform variable selection, i.e.,
#                    \mathcal{S}_b = ST2(X,y,lambda,kappa)
#   2b. Let \mathbf{E}(b,j) = { 1, if X_j \in S_b  ;  j=1,2,...,p.
#                             { 0, if X_j\notin S_b,
# 3. EndFor
# 4. Sort all the ensemble members in ascending order according to
#    the number of their selected variables, i.e.,
#           N_b = \sum_{j=1}^p \mathbf{E}(b,j) ,  b=1,2,...,B.
# 5. Select the U members sorted on top of the ranked list, i.e.,
#    arrange the U rows of \mathbf{E} ranked ahead into a new
#    matrix \mathbf{E}'.


# ----------------------------------


# =========================================
# \citep{zhang2019pruning}
# Title : Pruning variable selection ensembles
# Method: Ensemble Pruning of Stability Selection
# =========================================


# ----------------------------------


# =========================================
# \citep{xia2018maximum}
# =========================================


# ----------------------------------
# \citep{xia2018maximum}
# Maximum relevancy maximum complementary based ordered
# aggregation for ensemble pruning
#               [now works for mult-class]
# ----------------------------------
# It has been proven that the problem of ensemble pruning
# is an NP-complete problem [4,5].

# binary classification. Y \in {-1,+1}.


# Given a set of classifiers $C = \{C_1, C_2, ..., C_m\}$,
# the `relevance score` of the base classifier $C_i$ is:
#       Rel_{C_i} = 1 - \frac{ P(C \setminus C_i) }{ P(C) }     (1)
# where P(C) denotes the accuracy of ensemble with all the base
# classifiers on the training set, and P(C\C_i) denotes the accuracy
# of ensemble without base classifier C_i. Both of them are between
# 0 and 1.
# Rel_{C_i} reflects the efficacy of the corresponding classifier
# C_i, and could be positive [Ci possesses positive relevancy to the
# class, and removing Ci from the ensemble will reduce the generali-
# zation ability of final ensemble] or negative [Ci possesses nega-
# tive impact, removing it will increase the generalization].
# The higher score of relevancy is attributed to the more importance
# of the C_i.

def _relevancy_score(y, yt):
    nb_cls = len(yt)
    # fens = plurality_voting(y, yt)  # weighted_voting(y,yt,coef)
    fens = plurality_voting(yt)
    PC = np.mean(np.equal(fens, y))  # fens==y
    #   #
    idx = np.ones(shape=(nb_cls, nb_cls), dtype=DTY_BOL)
    for i in range(nb_cls):
        idx[i][i] = False
    yt = np.array(yt)
    fidx = [yt[i].tolist() for i in idx]
    # fidx = [yt[i] for i in idx]  # shape=[[nb_cls-1,] nb_cls]
    # fidx = [plurality_voting(y, i) for i in fidx]  # [nb_y,nb_cls]
    fidx = [plurality_voting(i) for i in fidx]
    del yt  # fidx ?? check required  # checked
    #   #
    PCCi = [np.mean(np.equal(i, y)) for i in fidx]  # i==y
    PC = check_zero(PC)  # RuntimeWarning: divide by zero encountered
    Rel_Ci = [1. - i / PC for i in PCCi]  # shape: [nb_cls,]
    return deepcopy(Rel_Ci), float(PC), deepcopy(PCCi)


# In order to derive such a complementary score, we map the original
# dataset $D$ and the ensemble of classifiers $C$ to a new dataset $F$
# within each feature value $F_ij$ matches with the classification
# owed by base classifier $C_j$ on the instance $(x_i, y_j)$ in $D$,
# and the class label in $F$ is the same as the one in dataset $D$.
#
# Given an already feature set $S$, the complementary of base classifier
# $C_i$ can be obtained as:
#       Com_{C_i} = \frac{ P(S\cup C_i) }{ P(S) } - 1       (7)
# where P(S\cup Ci) is the accuracy of the ensemble with all the classi-
# fiers in S and the classifier Ci, P(S) is the accuracy of the ensemble
# with all the classifiers in $S$. $Com_{C_i}$ reflects how much $C_i$
# contributes to existing sub-ensemble.
# e.g., Com_{C_i}=0.1 implies that the performance of ensemble can be
#       increased 10% by adding classifier Ci into the ensemble.

# Input : $S$ is an empty set, which denotes the set of selected features.
# Output: $F = \{f_1, f_2, ..., f_n\}$ is a set which contains all the
#         candidate features.

def _complementary_score(y, yt, h):
    fens = plurality_voting(yt)        # y,
    hens = plurality_voting(yt + [h])  # y,
    PS = np.mean(np.equal(fens, y))    # fens==y
    PSCi = np.mean(np.equal(hens, y))  # hens==y
    Com_Ci = PSCi / check_zero(PS) - 1.
    return float(Com_Ci), float(PS), float(PSCi)


# Considering a binary classification problem with class label $CL={-1,+1}$,
# where the feature value in $F$ would either be -1 or +1, a probabilistic
# density function derived from such dataset will not provide essential
# information.
# Given two base classifiers Ci and Cj, the $MI$ is defined as follows:
#       MI(C_i,C_j) = \frac{1}{N}\sum_{k=1}^N I(C_i(x_k) = C_j(x_k))    (8)
# where N denotes the number of samples in the training set. Ci(xk) is the
# classification result assigned by classifier Ci to sample \mathbf{x}_k,
# and I(.) is the indicator function (I(true)=1, and I(false)=0).

def _MRMR_MI_binary(Ci, Cj):
    # return np.mean(np.equal(Ci, Cj))
    Ci = np.array(Ci, dtype=DTY_INT)
    Cj = np.array(Cj, dtype=DTY_INT)
    return np.mean(Ci == Cj)


def _MRMR_MI_multiclass(Ci, Cj, y):  # CL
    Ci_xk = np.equal(Ci, y)
    Cj_xk = np.equal(Cj, y)
    return np.mean(np.logical_and(Ci_xk, Cj_xk))


def _MRMR_MI(Ci, Cj):  # multi-class
    ans = np.mean(np.equal(Ci, Cj))
    return float(ans)


# Fig.2  The procedure of MRMR
#       maximal relevant minimal redundant (MRMR) method
#
# Input : S, is an empty set, which denotes the set of selected features.
#         F, ={f1,f2,...,fn}, is a set which contains all the candidate
#         features.
# Output:
#
# 1. Select a feature $f_i$ from $F$ which has a maximum mutual informa-
#    tion between itself and the target class $CL$, then update $S$ and
#    $F$ as follows:
#       S = S \cup \{f\}                (4)    Correct: S\cup{f_i}
#       F = F \setminus \{f_i\}         (5)
# 2. Select feature $f_i$ from $F$ which satisfies the following condition:
#       f_i = \max_{f_i\in F} \{
#                   MI(f_i;CL) - \frac{1}{|S|}\sum_{f_j\in S} MI(f_i,f_j)
#             \}
# 3. Update $S$ and $F$ according to Eq.(4) and Eq.(5), and repeat step (2)
#    until desired number of features is selected.
#


# (2) Select feature f_j from F which satisfies the following condition
#       f_i = \max_{f_i \in F}\{
#                   MI(f_i; CL) -
#                   \frac{1}{|S|} \sum_{f_j \in S} MI(f_i, f_j)
#             \}
#
def _MRMR_subroute(y, yt, fj_in_S, k):
    MI_fj_CL = _MRMR_MI(yt[k], y)
    MI_fi_fj = [_MRMR_MI(yt[k], yt[j]) for j in fj_in_S]
    return MI_fj_CL - np.mean(MI_fi_fj)


def procedure_MRMR(y, yt, nb_cls, nb_pru):
    S = np.zeros(nb_cls, dtype=DTY_BOL)  # init, \emptyset
    MI_fi = [_MRMR_MI(fi, y) for fi in yt]
    # idx = np.argsort(MI_fi)[-1]  # with a maximum mutual information
    idx = MI_fi.index(np.max(MI_fi))  # between itself and target class
    S[idx] = True
    seq = [idx]  # seq = [int(idx)]
    #   #
    for _ in range(1, nb_pru):
        fi_in_F = np.where(np.logical_not(S))[0]
        fj_in_S = np.where(S)[0]  # is True
        objective = [_MRMR_subroute(y, yt, fj_in_S, i) for i in fi_in_F]
        # idx = np.argsort(objective)[-1]
        idx = objective.index(np.max(objective))
        idx = fi_in_F[idx]
        S[idx] = True
        seq.append(idx)  # seq.append(int(idx))
    #   #
    # P = np.where(S)[0]
    # return S.tolist(), P.tolist(), deepcopy(seq)
    return S.tolist(), deepcopy(seq)  # list


# ----------------------------------
# ----------------------------------


# After the relevance and complementary scores of each base classifier
# are obtained, we can rank the base classifier according to their
# score, the relevance-complementary score can be computed as follows:
#       RC_{C_i} = Rel_{C_i} + Com_{C_i}        (9)
#
# The complementary measure can reduce the chance of selecting highly
# related base classifiers. e.g., given Ci and Cj, which provide
# identical classification results on the training set, their relevance
# scores can be expressed as `Rel_{C_i} = Rel_{C_j}`, but according to
# the greedy searching process in Fig.2, the complementary score of C_j
# would be zero. The RC score for C_i will be higher than C_j.

def _relevance_complementary_score(Rel_Ci, Com_Ci):
    # Notice that: there is elementary, i.e., float (input)
    return Rel_Ci + Com_Ci  # i.e., RC_Ci


# Fig.3  The procedure of our proposed ensemble pruning
#       Maximum relevancy maximum complementary based ordered
#       aggregation for ensemble pruning
#
# Input : D, the training set.
# Output:
#
# 1. Use standard Bagging method to generate an ensemble of $t$ classifiers
#    $C = \{ C_1, C_2, ..., C_t \}$ from training set $D$.
# 2. Generate a new dataset $T$ according to the classification resulted by
#    each base classifier on the dataset $D$.
# 3. Calculate the relevance score of all features $f_i$ in $T$ using Eq.(1).
# 4. Calculate the complementary score of all the features $f_i$ in $T$ using
#    procedure described in Figure 2 and the Eq.(7).
# 5. Calculate the relevance-complementary score $RC$ of all the features us
#    -ing (9). Rank all the features in $T$ based on their $RC$ in descending
#    order.
# 6. Choosing the $M$ first ordered base classifiers which can maximize the
#    evaluation function to form the final ensemble.
#

def procedure_MRMC_ordered_EP(y, yt, nb_cls, nb_pru, S=None):
    # 3. calculate the relevance score using Eq.(1)
    Rel_Ci, _, _ = _relevancy_score(y, yt)
    # 4. calculate the complementary score using Eq.(7) and Fig.2
    #    method 2 (might be totally the same as method 1)
    # Com_Ci = [complementary_score(y, yt, h)[0] for h in yt]
    # ' ''
    #    method 1
    if S is None:
        S, _ = procedure_MRMR(y, yt, nb_cls, nb_pru)
    S = np.array(yt)[S].tolist()
    Com_Ci = [_complementary_score(y, S, h)[0] for h in yt]
    # ' ''
    #   #
    # 5. calculate the relevance-complementary score RC using Eq.(9)
    RC_Ci = [i + j for i, j in zip(Rel_Ci, Com_Ci)]
    # 5. rank all the features in T based on their RC in descending order.
    sorted_RC_index = np.argsort(RC_Ci)[:: -1]
    # 6. choose the M first ordered based classifiers which can maximize
    #    the evaluation function to form the final ensemble.
    seq = sorted_RC_index[: nb_pru].tolist()
    S = np.zeros(nb_cls, dtype=DTY_BOL)
    S[seq] = True
    return S.tolist(), seq  # list


def procedure_MRMC_EP_with_original_ensemble(y, yt, nb_cls, nb_pru):
    Rel_Ci, _, _ = _relevancy_score(y, yt)
    Com_Ci = [_complementary_score(y, yt, h)[0] for h in yt]
    # Using the original ensemble as the `S`, not specific in the paper
    RC_Ci = [i + j for i, j in zip(Rel_Ci, Com_Ci)]
    sorted_RC_index = np.argsort(RC_Ci)[:: -1]
    seq = sorted_RC_index[: nb_pru].tolist()
    S = np.zeros(nb_cls, dtype=DTY_BOL)
    S[seq] = True
    return S.tolist(), deepcopy(seq)  # list


# ----------------------------------
# ----------------------------------


# =========================================
# \citep{li2018mrmr}
# MRMR-based ensemble pruning for facial expression recognition
#               [multi-class]
# =========================================
#
# D :  the training data set, =\{(x_1,y_1),...,(x_n,y_n)\}
# n :  number of instances in training set D
# m :  the generated classifier pool size
# PM:  Prediction Matrix (PM),  PM=[C_ij], i\in[n], j\in[m]
#       where C_ij is the predictive label made by the j-th
#       classifier for samle x_i.


# ----------------------------------
# Training set data sampling
#
# Eq.(2) calculate the percentage of the number of classifiers that
# correctly classified a certain data sample over the size of
# generated classifier pool.
# If the percentage was greater than the predefined threshold \beta,
# then the data sample will be retained; otherwise, it will be
# removed:
#
# accept = { 1, if \frac{ \sum_{i=1}^m \mathbb{I}(c_i = y) }{m} > \beta     (2)
#          { 0, if \frac{ \sum_{i=1}^m \mathbb{I}(c_i = y) }{m} < \beta


# ----------------------------------
# Inspired by MRMR [42], MRMREP aims to select base classifiers which
# own high ability while being minimally redundant.


# ----------------------------------
# Measuring the capability of base classifier
#
# S:  a classifier subset with k classifiers, where k<m
# L:  the target label


def _judge_double_check_pruned_index(S):
    if np.array(S).dtype == DTY_BOL:
        return np.where(S)[0].tolist()
    # else:
    #     S = sorted(S)
    return sorted(S)


# the mutual information of the whole subset
# Aim to:
#       \max D(S_k, L),     D = I(S_k; L)                     (3)
# Replaced:
#       I(S_k;L) = \frac{1}{|S|} \sum_{c_i\in S} I(c_i;L)     (5)
#
def _MRMREP_mutual_information_of_whole_subset(y, yt, S):
    # I_Sk_L = [EntropyI(ci, y) for ci in yt]
    # return np.mean(I_Sk_L)
    #
    # if np.array(S).dtype == DTY_BOL:
    #     I_Sk_L = [EntropyI(ci, y) for ci, j in zip(yt, S) if j]
    # else:
    #     I_Sk_L = [EntropyI(yt[j], y) for j in S]
    S = _judge_double_check_pruned_index(S)
    I_Sk_L = [EntropyI(yt[j], y) for j in S]
    return np.mean(I_Sk_L)  # return nan if S==\emptyset


# Just using mutual information to measure classifier ability is
# not comprehensive enough. The F-statistic is mainly used to
# measure a feature's discrimination ability in the problem of
# multi-class classification. Its selected features can increase
# the separability among classes. The brief form of the F-statistic
# can be presented as the ratio of the Between-group variation and
# the Within-group variation:
#       F = \frac{ V_{between-group} }{ V_{within-group} }
# More specifically, it is:
#       F_{c_i} = \frac{
#                     \sum_{c=1}^C N_c |\mu_c - \mu| / (C - 1)
#                 }{
#                     \sum_{c=1}^C \sum_{k=1}^{N_c} (x_k - \mu_c)^2 / (N - C)
#                 }
# where N is the total number of samples, C is the number of classes,
# and N_c is the number of samples in the C-th class. \mu and \mu_c
# describe the mean value of the predicted values of classifier c_i
# in all samples and in the C-th class, respectively.
# x_k is the predicted value of the k-th sample by classifier c_i.
#
# For selected base classifiers, a larger F-statistic means stronger
# capability of separating classes. Furthermore, appropriate selection
# of classifiers can enlarge distances between different classes and
# narrow distances within classes.
#
def _MRMREP_F_statistic(y, Ci):
    N = len(y)  # number of instances
    C = len(np.unique(y))  # number of classes
    vY = np.unique(y).tolist()
    N_c = [np.sum(np.equal(y, i)) for i in vY]
    #   #
    mu = np.mean(Ci)
    y, Ci = np.array(y), np.array(Ci)
    mu_c = [np.mean(Ci[y == i]) for i in vY]
    #   #
    # numerator = np.sum([N_c[i] * abs(mu_c[i] - mu) for i in range(C)]) / (C - 1.)
    numerator = [N_c[i] * np.abs(mu_c[i] - mu) for i in range(C)]
    numerator = np.sum(numerator) / (C - 1.)
    denominator = np.zeros(N)
    # method 1:
    for i in range(C):
        denominator[y == vY[i]] = mu_c[i]
    denominator = [(Ci[i] - denominator[i])**2 for i in range(N)]
    # '''
    # # method 2:
    # for i in range(C):
    #     idx = y == vY[i]
    #     denominator[idx] = Ci[idx] - mu_c[i]
    # # method 3:
    # for i in range(N):
    #     for j in range(C):
    #         if vY[j] == y[i]:
    #             denominator[i] = Ci[i] - mu_c[j]
    # # method 2&3:
    # denominator = [i ** 2 for i in denominator]
    # '''
    denominator = np.sum(denominator) / (N - C)
    return numerator / check_zero(denominator)


# Normalzation:
# 1. min-max / 0-1 / normalization / 线性函数归一化 / 离散标准化
# 2. z-score 标准化 / zero-mean normalization
#
# refs:
#   https://blog.csdn.net/pipisorry/article/details/52247379
#   https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
#
# from sklearn import preprocessing as prep


# prep.minmax_scale()
def _normalization_min_max(x):
    xmin = np.min(x)
    xmax = np.max(x) - xmin
    # return [(i - xmin)/check_zero(xmax) for i in x]
    xmax = check_zero(xmax)
    return [(i - xmin) / xmax for i in x]


# prep.scale(x)
def _normalization_z_score(x):
    xmean = np.mean(x)
    x_std = np.std(x)  # , ddof=1)
    # return [(i - xmean) / check_zero(x_std) for i in x]
    x_std = check_zero(x_std)
    return [(i - xmean) / x_std for i in x]


# In summary, the capability of classifiers could be validated by Eq.(8).
# In Eq.(8), I(.) and F(.) are normalized between 0 to 1 seperately.
# The larger D(.), the stronger the ability of the selected subset S:
#       max D(I+F,L)
#           D = \frac{1}{|S|} \sum_{c_i\in S} (
#                   (1 - \alpha) * I(c_i;L) + \alpha * F(c_i;L)
#               )                                                   (8)
#
def _MRMREP_capability_of_classifiers(y, yt, S, alpha=0.5):
    # normalize='minmax'):
    # Might raise ValueError:
    #        zero-size array to reduction operation minimum
    #        which has no identity
    #    when S = []
    #
    # if np.array(S).dtype == DTY_BOL:
    #     S = np.where(S)[0].tolist()
    S = _judge_double_check_pruned_index(S)
    if len(S) == 0:
        # return np.sum([]) / check_zero(0.)
        return 0.0  # for robustness
    #   #   #
    I_ci_L = [EntropyI(yt[i], y) for i in S]
    F_ci_L = [_MRMREP_F_statistic(y, yt[i]) for i in S]
    I_ci_L = _normalization_min_max(I_ci_L)
    F_ci_L = _normalization_min_max(F_ci_L)
    D_IF_L = [(1. - alpha) * i + alpha * j for i, j in zip(I_ci_L, F_ci_L)]
    return np.mean(D_IF_L)


# ----------------------------------
# Pruning redundant classifiers


# The 2nd stage of EP is to wipe out the redundant and irrelevant
# classifiers in the classifier pool. Accomplished by measuring the
# similarity between classifiers, for which the Hamming distance is
# adopted.
# The Hamming distance is used in data transmission error control
# encoding, and it can be expressed as follows:
#       d(c_i,c_j) = \sum_{k=1}^K c_i(k) \bigoplus c_j(k)       (9)
# where c_i,c_j represent two different classifiers, c_i(k) is an
# attribute of classifier c_i, and c_i(k)\bigoplus c_j(k) indicates
# whether the attribute k of c_j is the same as c_i.
#
def _MRMREP_Hamming_distance(Ci, Cj):
    d_ci_cj = [i != j for i, j in zip(Ci, Cj)]
    return int(np.sum(d_ci_cj))  # int(np.int32)


# Furthermore, the distance of each pair of classifiers can be
# expressed as follows:
#       WD = [ 0    d_12 ... d_1k ]      (10)
#            [ d_21 0    ... d_2m ]
#            [ ...  ...  ... ...  ]
#            [ d_k1 d_k2 ... 0    ]
#
# Correct: WD = [d_ij],  i=[m], j=[m], instead of k
#          satisfying:  d_ij = d_ji
#
# It could be:
#   MRMREP_distance_of_each_pair(yt) or
#   MRMREP_distance_of_each_pair(ys) where ys=yt[S]
#
def _MRMREP_distance_of_each_pair(ys):
    nb_cls = len(ys)
    WD = np.zeros((nb_cls, nb_cls), dtype=DTY_INT)
    for i in range(nb_cls - 1):
        for j in range(i + 1, nb_cls):
            dist = _MRMREP_Hamming_distance(ys[i], ys[j])
            WD[i, j] = dist
            WD[j, i] = dist
    return WD.tolist()


# """
# # Lastly, the target function of the second stage can be expressed
# # as follows:
# #       min R(S),
# #   R = \frac{1}{|S|^2} \sum{c_i,c_j\in S, i\neq j} d(c_i,c_j)  (11)
# #
# def MRMREP_target_function_of_second_stage(yt, S, hyperparam='1'):
#     # if np.array(S).dtype == DTY_BOL:
#     #     S = np.where(S)[0].tolist()
#     S = judge_double_check_pruned_index(S)
#     if hyperparam == '1':
#         WD = MRMREP_distance_of_each_pair(yt)
#         WD = np.array(WD, dtype=DTY_FLT)
#         # WD = WD[S]
#         # WD = WD[:, S]
#         WD = WD[S][:, S].tolist()
#     elif hyperparam == '2':
#         yt = np.array(yt)[S].tolist()
#         WD = MRMREP_distance_of_each_pair(yt)
#     number_S = len(S)
#     return np.sum(WD) / number_S ** 2
#
#
# #
# # dist = WD[i,j]     \in [0, nb_y]
# # half of np.sum(WD) \in [0, nb_y] * m*(m-1)/2
# # np.sum(WD)         \in [0, nb_y] * (m*m -m)
# # np.sum(WD)/|S|**2  \in [0, nb_y] * (m-1)/m < [0, nb_y]
# #
# def MRMREP_second_objective_based_distance_WD(WD, S):
#     # if np.array(S).dtype == DTY_BOL:
#     #     S = np.where(S)[0].tolist()
#     S = judge_double_check_pruned_index(S)
#     WD = np.array(WD, dtype=DTY_FLT)
#     # WD = WD[:, S]
#     # WD = WD[S]
#     WD = WD[:, S][S].tolist()
#     number_S = len(WD)
#     return np.sum(WD) / number_S ** 2
# """


def _MRMREP_target_function_of_second_stage(yt, S):
    S = _judge_double_check_pruned_index(S)
    ys = np.array(yt)[S].tolist()
    WD = _MRMREP_distance_of_each_pair(ys)
    number_S = len(S)
    return np.sum(WD) / check_zero(number_S**2)


# According to the MRMR, the final target function is as follows:
#       max \phi(D, R),  \phi = D - R
#
# """
# def MRMREP_final_target_function(y, yt, S, WD=None, alpha=0.5, hyperparam='2'):
#     D = MRMREP_capability_of_classifiers(y, yt, S, alpha)
#     if WD is None:
#         R = MRMREP_target_function_of_second_stage(yt, S, hyperparam)
#     else:
#         R = MRMREP_second_objective_based_distance_WD(WD, S)
#     return D - R
# """

# D \in [0, 1] since I,F are normalized into [0, 1]
# R \in [0, nb_y * nb_S * (nb_S - 1)] / nb_S**2
#   i.e., [0, nb_y * (nb_S - 1) / nb_S]
#   belongs to [0, 1) * nb_y = [0, nb_y)
# therefore, \phi = D-R \in (-nb_y, 1]  # NOT (-1, 1]
#
# I, F are normalized into [0, 1]
# (1-\alpha)*I + \alpha*F \in [0, 1-alpha] + [0, \alpha] = [0, 1]
# therefore, D \in [0, 1]


def _MRMREP_final_target_function(y, yt, S, alpha=0.5):
    D = _MRMREP_capability_of_classifiers(y, yt, S, alpha)
    R = _MRMREP_target_function_of_second_stage(yt, S)
    return float(D - R)


# ----------------------------------
# Searching for the optimal subset


# ----------------------------------
# Classifier fusion
# ----------------------------------
#
# S = [c_1,...,c_k] (k<m),  the classifier subset from the
#                           above steps,
# m     the size of the classifer pool
# Plurality voting is used to fuse the labels predicted by
#                  each selected classifier


# Table 2. MRMREP
#
# Input : the prediction matrix PM, the training labels Y,
#         forward search step L, backward search step R (L > R)
# Output: selected subset S
#
#  1. Compute the F-statistic value of each classifier c_i using
#     Eq.(7)
#  2. Compute the mutual information between classifier c_i and
#     target label Y of each sample, I_{c_i} = I(c_i;Y)
#  3. Compute the Hamming distance of each pair of classifiers
#     and construct the distance matrix WD.
#  4. The selection procedure starts with an empty set of classi-
#     fiers (S=\emptyset).
#     Select the best model into the set according to Eq.(8).
#  5. Repeat L times:
#  6.    Find a classifier in the non-selected subset to maximize
#        Eq.(12) and include it in S.
#  7.    c_i^+ = \argmax_{c_i \notin S_k} \phi(S_k + c_i)
#  8.    S_{k+1} = S_k + c_i^+
#  9. End Repeat
# 10. Repeat R times:
# 11.    Find a classifier in S that can maximize the following
#        formula, and then remove it from the selected subset.
# 12.    c_i^- = \argmax_{c_i \in S_k} \phi(S_k - c_i)
# 13.    S_{k+1} = S_k - c_i^-
# 14. End Repeat
# 10. Repeat steps (5) [lines 5--9] and (6) [lines 10--14] until
#     all classifiers are included in S in a certain order, and
#     then output the ranked list of each classifier.
#


def _subroute_MRMREP_init(y, yt, alpha):
    # 1. Compute the F-statistic value of each c_i in yt
    F_ci = [_MRMREP_F_statistic(y, ci) for ci in yt]
    # 2. Compute the mutual information between ci and target
    #    label Y of each sample.
    I_ci = [EntropyI(ci, y) for ci in yt]
    # 3. Compute the Hamming distance of each pair of classifiers
    WD = _MRMREP_distance_of_each_pair(yt)  # the distance matrix
    # 4. starts with an empty set of S, choose the best model
    #    based on Eq.(8)
    F_ci = _normalization_min_max(F_ci)
    I_ci = _normalization_min_max(I_ci)
    D_ci = [(1. - alpha) * i + alpha * j for i, j in zip(I_ci, F_ci)]
    idx = D_ci.index(np.max(D_ci))
    # D_ci = [i + j for i, j in zip(F_ci, I_ci)]  # D_IF_L
    # idx = np.argsort(D_ci)[-1]  # max D(I + F, L)
    return idx


def _subroute_MRMREP_cover(y, yt, alpha, Sk):
    # 1) Find a classifier in the non-selected subset to maximize
    #    formula (12) and include it in S.
    idx_not_in_S = np.where(np.logical_not(Sk))[0]
    idx_not_in_S = idx_not_in_S.tolist()
    if len(idx_not_in_S) == 0:
        return -1
    # 2) c_i^+ = \argmax_{c_i \notin S_k} \phi(S_k + c_i)
    L_temS = [deepcopy(Sk) for _ in idx_not_in_S]
    for i, j in enumerate(idx_not_in_S):
        L_temS[i][j] = True
    L_tem_phi = [_MRMREP_final_target_function(
        y, yt, j, alpha) for j in L_temS]
    # # c_i_plus = np.argsort(L_tem_phi)[-1]
    c_i_plus = L_tem_phi.index(np.max(L_tem_phi))
    # 3) S_{k+1} = S_k + c_i^+
    return idx_not_in_S[c_i_plus]


def _subroute_MRMREP_untie(y, yt, alpha, Sk):
    # Might raise ValueError:
    #     zero-size array to reduction operation minimum which
    #     has no identity
    # when len(Sk) == 1
    #
    # 1) Find a classifier in S that can maximize the following
    #    formula, and then remove it from the selected subset.
    idx_in_S = np.where(Sk)[0].tolist()
    if len(idx_in_S) == 0:
        return -1
    # 2) c_i^- = \argmax_{c_i \in S_k} \phi(S_k - c_i)
    L_temS = [deepcopy(Sk) for _ in idx_in_S]
    for i, j in enumerate(idx_in_S):
        L_temS[i][j] = False
    L_tem_phi = [_MRMREP_final_target_function(
        y, yt, j, alpha) for j in L_temS]
    # # c_i_minus = np.argsort(L_tem_phi)[-1]
    c_i_minus = L_tem_phi.index(np.max(L_tem_phi))
    # 3) S_{k+1} = S_k - c_i^-
    return idx_in_S[c_i_minus]


# encode
# decode  # remove
# (leading/tending)
# Using `list.index(np.max(list))` prefers the former individuals
# Using `np.argsort(list)[-1]` tends to choose the latter individual


def _MRMREP_selected_subset(y, yt, nb_cls, L, R, alpha=0.5):
    # nb_cls == m,  number of individual classifiers
    # nb_pru == k,  number of the pruned sub-ensemble
    # PM = np.array(yt).transpose()  # [n,m], n instances
    #
    idx = _subroute_MRMREP_init(y, yt, alpha)
    S = np.zeros(nb_cls, dtype=DTY_BOL).tolist()
    S[idx] = True
    seq = [idx]
    # 7. repeat steps 5&6 until all classifiers are included into
    #    S in a certain order, and then output the ranked list of
    #    each classifier.
    while np.sum(S) < nb_cls:  # k
        # 5. repeat L times
        for _ in range(L):
            c_i_plus = _subroute_MRMREP_cover(y, yt, alpha, S)
            if c_i_plus == -1:
                break
            S[c_i_plus] = True
            seq.append(c_i_plus)
        if np.sum(S) == nb_cls:
            break
        # 6. repeat R times
        for _ in range(R):
            c_i_minus = _subroute_MRMREP_untie(y, yt, alpha, S)
            if c_i_minus == -1:
                break
            S[c_i_minus] = False
            seq.remove(c_i_minus)
        #   #
        # ' ''
        # Basically, like: S = [....]
        #   L times returns: c1, c2, c3, ..., c_L
        #   R times returns: C_L, ..., c3, c2, c1
        # That happens for Y \in {-1,+1} or {0,1}, i.e., binary class
        # But things change when Y \in {0,1,...,nc-1}
        #
        # Besides, if L==R, then sum(S) wouldn't change during while
        # and it becomes a dead loop. Thus, L>=R+1
        # ' ''
        #   #
    # 7. repeat steps 5 and 6 until all classifiers are included
    #    into S in a certain order, and then
    return deepcopy(seq)


def MRMREP_Pruning(y, yt, nb_cls, nb_pru, L=4, R=3, alpha=0.5):
    # ! L, R = {3, 2}??  the values are randomly chosen.
    ranked_list = _MRMREP_selected_subset(y, yt, nb_cls, L, R, alpha)
    seq = ranked_list[: nb_pru]
    S = np.zeros(nb_cls, dtype=DTY_BOL)
    S[seq] = True
    return S.tolist(), seq, ranked_list  # list


# ----------------------------------


# =========================================
# \citep{ali2019classification}
# =========================================


# ----------------------------------
# \citep{ali2019classification}
# ----------------------------------


# Algorithm 1 [31]: DCA for solving problem (14)
#
# Input :
# Output:
#
# 1. Initialize (x^0, y^0) \in \Omega, k <-- 0
# 2. repeat
# 3.    Compute \bar{x}^k \in \partial h(x^k)
#                                         \forall i=1,...,n
#               \bar{y_i}^k \in -\rho \partial (-r)(y_i^k)
# 4.    Compute x^{k+1} \in \argmin_{x\in C}\{
#                               g(x)- <\bar{x}^k,x> + <\bar{y}^k,|x|>
#                           \}
#               \bar{y_i}^{k+1} = |x_i^{k+1}|
#                                         \forall i=1,...,n.
# 5.    k <-- k+1
# 6. until Stopping criterion
#


# Algorithm 2: Ensemble Pruning by DCA
#
# Input : E:= Ensemble of Classifiers, (\rho, \theta), Threshold
# Output: Model Performance
# Notice: \rho  \in {0.001, 0.1, 10, 10^2, 10^3, 10^4, 10^5} of 7 values
#         \theta\in {0.001, 0.01, 0.5, 1, 10, 100, 500}      of 7 values
#
#  1. Calculate Accuracy-Diversity Matrix $D$ by Eq.(8) on TRAINING data
#  2. For all (\rho, \theta) do
#  3.    Solve the problem (14) by Algorithm 1 on TRAINING data
#  4.    Find the indices $\mathbf{I}$:= (x(:) >= Threshold)
#  5.    Select the classifiers corresponding to those indices $\mathbf{I}$
#        and construct subensemble $S_{(\rho,\theta)} \subset E$ by using
#        $\mathbf{I}$
#  6.    Predict on VALIDATION set with the models in the subensemble
#        $S_{(\rho,\theta)}$
#  7.    Aggregate predictions of all classifiers in the ensemble
#        $S_{(\rho,\theta)}$ by VOTING
#  8.    Calculate Error Matrix $Err_{(\rho, \theta)}$
#  9. End For
# 10. Select the best $(\rho^*, \theta^*)$ with the least error from
#     $Err_{(\rho,\theta)}$ and select the corresponding subset
#     $S^*_{(\rho^*,\theta^*)} \subset E$
# 11. Prediction on TEST set with the best subensemble
#     $S^*_{(\rho^*,\theta^*)}$
# 12. Aggregate predictions of all classifiers in the subensemble
#     $S^*_{^*,^*}$ by VOTING
# 13. Compute the error, Area Under Curve, Sensitivity
#


# ----------------------------------
# \citep{cao2018optimizing}
# Optimizing multi-sensor deployment via ensemble pruning for
# wearable activity recognition
#               [multi-class]
# ----------------------------------
#
# training set, compose of N labeled instances
#       \mathbf{X} = {(\mathbf{x}_i, y_i)}_{i=1}^N
# component classifiers, generated by ensemble approaches
#       H = {h_t} (t=1,2,...T)
# using majority voting [correct, plurality voting]
#
# Let S_u and L_u denote the current selected classifier set by pru-
# ning and current left classifier set in H at the u-th iteration,
# respectively.
# Obviously, we have H = S_u \cup L_u, and u = 1,2,...,T.


# Reduce-Error pruning (RE)

# Complementarity measure
# \citep{martinez2004aggregation, martinez2009analysis}


# mRMR and disc pruning
#   minimal redundancy and maximal relevance (mRMR) algorithm
#
# mRMR ensemble_pruning
# 1. starts with the individual classifier s_1 whose training
#    accuracy is the highest
# 2. according to mRMR approach, the s_u selected for
#    sub-ensemble S_u can be proposed as:
#       s_u = \argmax_k[
#                   I(h_k;Y) -
#                   \frac{1}{u-1} \sum_{h_i \in S_{u-1}} I(h_k;h_i)
#             ]
#    in which I(m;n) is the mutation information of variable m
#    and n. Y is target class; the index k \in L_{u-1} and
#    S_u = S_{u-1} \cup {s_u}
#
def mRMR_ensemble_pruning(y, yt, nb_cls, nb_pru):
    # starts with the individual classifier (best accuracy)
    acc = [np.mean(np.equal(y, i)) for i in yt]
    idx = acc.index(np.max(acc))  # idx = np.argsort(acc)[-1]
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    P[idx] = True
    seq = [idx]  # seq.append(int(idx))
    #   #
    # mRMR to choose s_u
    for _ in range(2, nb_pru + 1):  # for u in range(2, nb_pru+1):
        S_u = np.where(P)[0]  # pruned sub- # selected individuals
        L_u = np.where(np.logical_not(P))[0]  # left individuals
        term_left = [EntropyI(yt[k], y) for k in L_u]
        term_right = []
        for k in L_u:
            temp = [EntropyI(yt[k], yt[i]) for i in S_u]
            # BUG! temp = np.sum(temp) / (u - 1.)
            temp = np.mean(temp)
            term_right.append(temp)
        term = [i - j for i, j in zip(term_left, term_right)]
        # idx = np.argsort(term); idx = L_u[idx[-1]]
        idx = term.index(np.max(term))
        idx = L_u[idx]
        P[idx] = True
        seq.append(idx)
    return P.tolist(), deepcopy(seq)  # list


# Disc ensemble pruning
#
# discriminative pruning criterion:
#   s_u = \argmax_k[
#               I(h_k^{mis}; Y^{mis}) +
#               \frac{1}{u-1} \sum_{h_i \in S_{u-1}}
#                           I(h_k^{cor}; h_i^{cor})
#         ]
# in which h_k^{mis} and h_k^{cor} is misclassified instance part
# and correctly classified instance part of h_k by S_{u-1} respec-
# tively; Y^{mis} is the misclassified instance part of target
# class; the index k\in L_{u-1} and S_u=S_{u-1}\cup{s_u}.
#
# # def Disc_EP_subroute(hk, hi_y, y, hparam='mis'):
# #     # input : np.ndarry, [different from others, all list]
# #     if hparam == 'mis':
# #         idx = hk != y
# #     elif hparam == 'cor':
# #         idx = hk == y
# #     return
#   #
def Disc_ensemble_pruning(y, yt, nb_cls, nb_pru):
    # starts with the individual classifier (best accuracy)
    acc = [np.mean(np.equal(y, i)) for i in yt]
    idx = acc.index(np.max(acc))  # idx = np.argsort(acc)[-1]
    P = np.zeros(nb_cls, dtype=DTY_BOL)
    P[idx] = True
    seq = [idx]  # seq.append(int(idx))
    #   #
    # Disc to choose
    y, yt = np.array(y), np.array(yt)  # DTY_FLT
    for _ in range(2, nb_pru + 1):  # for u in
        S_u = np.where(P)[0]  # pruned sub- # selected individuals
        L_u = np.where(np.logical_not(P))[0]  # left individuals
        term_left, term_right = [], []
        for k in L_u:
            hk = yt[k]  # np.array(yt[k])
            cor_idx = hk == y  # np.equal(hk, y)
            # mis_idx = hk != y  # = ~cor_idx
            #   #
            # Entropy_I(h_k^{mis}; Y^{mis})
            temp_left = EntropyI(hk[~cor_idx].tolist(),
                                 y[~cor_idx].tolist())
            term_left.append(temp_left)
            # Entropy_I(h_k^{cor}; h_i^{cor})
            temp_righ = [hi[cor_idx].tolist() for hi in yt[S_u]]
            temp_righ = [EntropyI(hk[cor_idx].tolist(),
                                  hi) for hi in temp_righ]
            # BUG! temp_righ = np.sum(temp_righ) / (u - 1.)
            temp_righ = np.mean(temp_righ)
            term_right.append(temp_righ)
        #   #   #
        term = [i + j for i, j in zip(term_left, term_right)]
        # idx = np.argsort(term); idx = L_u[idx[-1]]
        idx = term.index(np.max(term))
        idx = L_u[idx]
        P[idx] = True
        seq.append(idx)
    return P.tolist(), deepcopy(seq)

# Notice:
# Disc_ensemble_pruning might:
#   UnboundLocalError: local variable 'i' referenced before assignment
#   core/data_entropy.py: line 175 px,_=prob(X); line 93, del X,dX,i
# Reasons could be:
#   np.sum(np.equal(y, fens)) == 0
#   or some fx in yt, np.sum(np.equal(fx, y)) == 0


# ----------------------------------


# =========================================
# \citep{zhang2019two}
# Two-Stage Bagging Pruning for Reducing the Ensemble Size and
# Improving the Classification Performance
#               [multi-class]
# =========================================


# ----------------------------------
# ----------------------------------


def _subroute_TwoStage_OBi(y_trn, indices):
    nb_y = set(range(len(y_trn)))
    OB_i = [list(nb_y - set(i)) for i in indices]
    return OB_i


# def subroute_TwoStage_checkTaTd(ta, td):
#     return ta, td


def _subroute_TwoStage_checkAC(ta):
    if not (0 <= ta <= 9):
        ta = min(max(ta, 0), 9)
    # ta \in {0,1,...,9}
    return ta


def _subroute_TwoStage_checkDIS(td):
    if not (1 <= td <= 10):
        td = min(max(td, 1), 10)
    # td \in {1,2,...,10}
    return td


# ----------------------------------
# ----------------------------------


# ----------------------------------
# 3.1 Accuracy-Based Pruning Method (AP)
#
# Algorithm 2: Accuracy based pruning for bagging
# Input : D- training set,
#         {D_i}- bootstrap subsets from D,
#         ES- number of base models or subsets,
#         {m_i}- a set of base models.
# Output: RM- a reduced set of base models,
#         PB- a pruned bagging ensemble
#
# 1. Initialize RM = \emptyset.
# 2. Collect the subsets of out-of-bag samples as
#                       OB_i = D - D_i, i=1,2,...,ES
# 3. Calculate the accuracy AC_i for each base model m_i
#    tested on the OB_i, i=[ES]
# 4. Given a parameter ta\in{0,1,2,...,9}, compute the
#    threshold T, which is the ta-th decile value of the
#    set {AC_i | i=1,2,...,ES}
# 5. For i\in {1,2,...,ES} do:
# 6.    if AC_i >= T:
# 7.        RM = RM \cup {m_i}
# 8. The outcome PB(x) of a test sample x predicted by the
#    pruned ensemble PB is given as follows:
#           PB(x) = majority class in {bm(x) | bm \in RM}
#

# For a given parameter `ta`, it is easy to know that
# `floor( (ta/10)*ES )` base model will be excluded out of
# the original ensemble and the size of reduced classifier
# set is equal to `ES - floor( (ta/10)*ES )`.
#
# rho = nb_pru / nb_cls  &  nb_pru = ceil(nb_cls * rho)
# Then remove would be floor(nb_cls * (1 - rho))
#      ta = (1 - rho) * 10
#      ta = np.round((1-rho)*10)  --> int
#      ta = max(ta, 0) and min(ta, 9)
# Notice that:
#   1/nb_cls < 1-rho < 1   (1-rho)*10 \in (10/nb_cls, 10)


def _subroute_TwoStage_AccuracyBased(y_trn, y_insp, OB_i):
    y, yt = np.array(y_trn), np.array(y_insp)
    if all([len(i) > 0 for i in OB_i]):
        # y, yt = np.array(y_trn), np.array(y_insp)
        AC_i = [np.mean(yt[i][idx] == y[idx]) for i, idx in enumerate(OB_i)]
        return AC_i
    # else:
    #     # y_trn, y_insp = np.array(y_trn), np.array(y_insp)
    #     # y_val, y_cast = np.array(y_val), np.array(y_cast)
    #   #   #
    AC_i, nb_trn = [], len(y_trn)
    for i, idx in enumerate(OB_i):
        # ' ''
        # tem = np.random.randint(nb_trn, size=nb_trn).tolist() if not idx else idx
        # tem = list(set(range(nb_trn)) - set(tem)) if not idx else idx
        # tem = np.mean(y_insp[i] == y_trn) if not tem else np.mean(y_insp[i][tem] == y_trn[tem])
        # ' ''
        tem = idx
        if not tem:
            tem = np.random.randint(nb_trn, size=nb_trn).tolist()
            tem = list(set(range(nb_trn)) - set(tem))
        if tem:
            tem = np.mean(yt[i][tem] == y[tem])
        else:  # tem == []  # not tem
            tem = np.mean(yt[i] == y)
            # or: tem = 0.0
            # or: tem = np.mean(y_cast[i] == y) using np.ndarray
        AC_i.append(float(tem))
    #   #   #
    del nb_trn, y, yt
    return deepcopy(AC_i)


def TwoStagePruning_AccuracyBasedPruning(
        y_trn, y_insp, nb_cls, indices, ta=3):
    # 1. Initialize RM = \emptyset.
    '''
    P = np.zeros(nb_cls, dtype=DTY_BOL)  # RM
    seq = []
    '''
    # 2. Collect the subsets of out-of-bag samples as
    #    OB_i = D - D_i, i=1,2,...,ES
    OB_i = _subroute_TwoStage_OBi(y_trn, indices)
    #
    # 3. Calculate the accuracy AC_i for each base model
    #    m_i tested on the OB_i, i=1,2,...,ES.
    AC_i = _subroute_TwoStage_AccuracyBased(y_trn, y_insp, OB_i)
    #
    # 4. Given a parameter ta\in{0,1,2,...,9}, compute the
    #    threshold T, which is the $ta$-th decile value of
    #    the set {AC_i | i=1,2,...,ES}
    if not (0 <= ta <= 9):
        # ta = max(ta, 0); ta = min(ta, 9)
        ta = min(max(ta, 0), 9)
    T = np.percentile(AC_i, ta * 10)  # threshold
    # Note: ta\in {0,1,2,..,9} i.e., ta-th decide value of set AC_i
    #
    # 5. for i\in {1,2,...,ES} do:
    # 6.     if AC_i >= T:
    # 7.         RM = RM \cup {m_i}
    # '''
    # for i in range(nb_cls):
    #     if AC_i[i] >= T:
    #         P[i] = True
    #         seq.append(i)
    # '''
    #   #
    # 8. The outcome PB(x) of a test sample x predicted by
    #    the pruned ensemble PB is given as follows:
    #        PB(x) = majority class in {bm(x) | bm\in RM}
    # '''
    # return P.tolist(), deepcopy(seq)  # list
    # '''
    #   #
    P = np.array(AC_i) >= T
    seq = np.where(P)[0].tolist()
    return P.tolist(), seq  # list


# ----------------------------------
# 3.2 Distance-Based Pruning Method (DP)
#
# Algorith 3: Distance based pruning for bagging algorithm.
# Input : D- training set,
#         {D_i}- subsets sampled from D,
#         {m_i}- a set of base models,
#         ES- number of base models or subsets,
#         x- feature vector representing a test sample
# Output: RM- a reduced set of base models,
#         PB- a pruned bagging ensemble
#
# 1. Collect the subsets of out-of-bag samples as
#                               OB_i = D - D_i, i=1,2,...,maxiter.
# 2. Calculate the center of each OB_i, as C_i, i=1,2,...,maxiter
# 3. Calculate the Euclidean distance d_i = ||x - C_i|| from
#    the test sample x to each center of OB_i, i=1,2,...,ES.
# 4. Given a parameter td\in{1,2,...,10}, compute the threshold
#    T, which is the $td$-th decile value of the set
#    DIS = {d_i | i=1,2,...,ES}.
# 5. Initialize RM = \emptyset.
# 6. For i \in {1,2,...,ES} do:
# 7.    if d_i <= T:
# 8.        RM = RM \cup {m_i}
# 9. The outcome PB(x) of a test sample x predicted by the pruned
#    ensemble PB is given as follows:
#           PB(x) = majority class in {bm(x) | bm \in RM}
#

# Briefly, we first compute the center of an out-of-bag sample
# set OB_i as follows:
#       C_i = \frac{1}{n_i} \sum_{p \in OB_i} p,
#                                        i = 1,2,...,ES        (4)
# where n_i = |OB_i| is the size of the out-of_bag sample set OB_i.

# For any new test sample x, the Euclidean distance \mathbf{d}_i
# from the test sample x to each center of OB_i was calculated as
#       \mathbf{d}_i = ||x - C_i||                             (5)

# Similarly as the AP procedure, the selection of base models was
# executed according to a decile parameter 𝑡𝑑 ∈
# {1,2, 3, 4, 5, 6,7, 8, 9,10}. If 𝑑𝑖 is larger than a threshold 𝑇,
# which is calculated as the 𝑡𝑑-th decile in the set 𝐷𝐼𝑆 = {𝑑𝑖 |
# 𝑖 = 1, 2, .., 𝐸𝑆}, the basemodel 𝑚𝑖 will be excluded out of the
# original bagging ensemble; otherwise, it will be retained.
#
# Removed: (10-td)/10 * 100%
# Saved  : rho = 1 - (10-td)/10  =  1 - (1-td/10) = td/10
# Then  td = np.round( (nb_pru / nb_cls) * 10 )  --> 10
#       td = max(td, 1) and min(td, 10)
#
# `floor( (1-td/10) *ES )` base model will be excluded out of the
# original ensemble and the size of reduced classifier set is equal
# to `ES - floor( (1-td/10) *ES )`.


def _subroute_TwoStage_DistanceBased_inst(X_trn, X_val, OB_i):
    X_trn = np.array(X_trn)  # DTY_FLT
    X_val = np.array(X_val)  # DTY_FLT
    if all([len(i) > 0 for i in OB_i]):
        # C_i = [np.mean(X_trn[i], axis=0) for i in OB_i]
        C_i = [np.mean(X_trn[idx], axis=0) for idx in OB_i]
    else:
        C_i, nb_trn = [], len(X_trn)
        for i, idx in enumerate(OB_i):
            # ' ''
            # tem = np.random.randint(nb_trn, size=nb_trn).tolist() if not idx else idx
            # tem = list(set(range(nb_trn)) - set(tem)) if not idx else idx
            # tem = np.mean(X_trn, axis=0) if not tem else np.mean(X_trn[tem], axis=0)
            # ' ''
            tem = idx
            if not tem:
                tem = np.random.randint(
                    nb_trn, size=nb_trn).tolist()
                tem = list(set(range(nb_trn)) - set(tem))
            if tem:
                tem = np.mean(X_trn[tem], axis=0)
            else:
                tem = np.mean(X_trn, axis=0)
            C_i.append(deepcopy(tem))
    #   #   #
    # C_i: shape= [(nb_feat,) nb_cls] = (nb_cls, nb_feat)
    d_i = [np.sqrt(np.sum((X_val - i)**2, axis=1)) for i in C_i]
    d_i = [np.mean(i) for i in d_i]  # all test instances --> one
    #   #   #
    del X_trn, X_val
    C_i = [i.tolist() for i in C_i]  # or np.array(C_i).tolist()
    return C_i, d_i  # list


def _subroute_TwoStage_DistanceBased(y_trn, y_insp, OB_i, y_cast):
    y, yt = np.array(y_trn), np.array(y_insp)
    if all([len(i) > 0 for i in OB_i]):
        C_i = [yt[:, idx].mean(axis=1).tolist() for idx in OB_i]
    else:
        C_i, nb_trn = [], len(y_trn)
        for i, idx in enumerate(OB_i):
            tem = idx
            if not tem:
                tem = np.random.randint(nb_trn, size=nb_trn).tolist()
                tem = list(set(range(nb_trn)) - set(tem))
            if tem:
                tem = yt[:, tem].mean(axis=1).tolist()
            else:
                tem = yt.mean(axis=1).tolist()
                # or: tem = [0.0 for _ in range(nb_cls)]
                # or: tem = y_cast.mean(axis=1).tolist() by np.ndarray
            C_i.append(deepcopy(tem))
    # C_i: shape= (nb_cls, nb_cls)
    #   #   #
    ys = np.array(y_cast)  # DTY_FLT
    d_i = []
    for Ci_val in C_i:  # shape=(nb_cls,)
        ai_val = np.array([Ci_val]).T  # shape=(nb_cls,1)
        di_val = np.sqrt(np.sum((ys - ai_val)**2, axis=0))
        # # di_val = np.mean(di_val, axis=1)  # shape=(nb_cls,)
        # '''
        # di_val = (ys - ai_val)**2  # shape=(nb_cls, nb_val)
        # di_val = np.sum(di_val, axis=0)  # shape=(nb_val,)
        # di_val = np.sqrt(di_val)  # shape=(nb_val,)
        # '''
        di_val = np.mean(di_val)  # scalar
        d_i.append(di_val)  # np.float64
    # d_i: shape= (nb_cls,)
    #   #   #
    del ai_val, di_val, y, yt, ys
    return deepcopy(C_i), deepcopy(d_i)


def TwoStagePruning_DistanceBasedPruning(y_trn, y_insp, nb_cls, indices, td,
                                         y_cast):  # X_trn, X_val):
    # 1. Collect the subsets of base models, PB- a pruned
    #    bagging ensemble
    '''
    P = np.zeros(nb_cls, dtype=DTY_BOL)  # RM
    seq = []
    '''
    # 2. Calculate the center of each OB_i as C_i,
    #    i=1,2,...,maxiter
    OB_i = _subroute_TwoStage_OBi(y_trn, indices)
    #
    # 3. Calculate the Euclidean distance d_i=\|x-C_i\| from
    #    the test sample x to each center of OB_i, i=1,2,..,ES.
    #   #
    C_i, d_i = _subroute_TwoStage_DistanceBased(y_trn, y_insp, OB_i, y_cast)
    # 4. Given a parameter td\in {1,2,...,10}, compute the
    #    threshold T, which is the $td$-th decile value of the
    #    set DIS={d_i | i=1,2,...,ES}
    if not (1 <= td <= 10):
        # td = max(td, 1); td = min(td, 10)
        td = min(max(td, 1), 10)
    T = np.percentile(d_i, td * 10)  # threshold
    # Note that: td \in {1,2,...,10}
    #
    # 5. Initialize RM = \emptyset.
    # 6. for i \in {1,2,...,ES} do:
    # 7.     if d_i <= T:
    # 8.         RM = RM \cup {m_i}
    # '''
    # for i in range(nb_cls):
    #     if d_i[i] <= T:
    #         P[i] = True
    #         seq.append(i)
    # '''
    #   #   #
    # 9. The outcome PB(x) of a test sample x predicted by the
    #    pruned ensemble PB is given as follows:
    #        PB(x) = majority class in {bm(x) | bm \in RM}
    # '''
    # return P.tolist(), deepcopy(seq)
    # '''
    #   #   #
    P = np.array(d_i) <= T
    seq = np.where(P)[0].tolist()
    return P.tolist(), seq  # list


# ----------------------------------
# 3.3 Two-Stage Pruning on the Bagging Algorithm [TSP]


# ----------------------------------
# ----------------------------------


# ----------------------------------
#
# Algorithm 4: Two-stage pruning for "AP+DP"
# Input: 𝑅𝑀- a reduced set of base models generated by the
#             AP procedure,
#        𝑃- number of base models in 𝑅𝑀,
#        {𝑖1, 𝑖2, . . . , 𝑖𝑃}- index set of base models in 𝑅𝑀,
#        𝑥- feature vector of a test sample
# Output: 𝑅𝑀2- a reduced set of base models,
#         𝑃𝐵2- a pruned bagging ensemble
#
# 1 Given a parameter 𝑡𝑑 ∈ {1, 2, . . . , 10}, compute the
#   threshold 𝑇, which is the td-th decile value of the set
#               𝐷𝐼𝑆 = {𝑑𝑘 | 𝑘 ∈ {𝑖1, 𝑖2, . . . , 𝑖𝑃}}.
# 2 Initialize 𝑅𝑀2 = 0.
# 3 for 𝑘 ∈ {𝑖1, 𝑖2, . . . , 𝑖𝑃} do:
# 4     if 𝑑𝑘 ≤ 𝑇:
# 5         𝑅𝑀2 = 𝑅𝑀2 ∪{𝑚 𝑘}
# 6 The outcome 𝑃𝐵2(𝑥) of a test sample 𝑥 predicted by the
#   pruned ensemble 𝑃𝐵2 is given as follows:
#       𝑃𝐵2(𝑥) = majority class in {𝑏𝑚(𝑥) | 𝑏𝑚 ∈ 𝑅𝑀2}
#

def TwoStagePruning_APplusDP(y_trn, y_insp, nb_cls, indices,
                             ta, td, y_cast):
    OB_i = _subroute_TwoStage_OBi(y_trn, indices)
    #   #
    # 1. Accuracy-based
    AC_i = _subroute_TwoStage_AccuracyBased(y_trn, y_insp, OB_i)
    ta = _subroute_TwoStage_checkAC(ta)
    T = np.percentile(AC_i, ta * 10)  # threshold
    PA = np.array(AC_i) >= T
    # 2. Distance-based
    y_insp = np.array(y_insp)[PA].tolist()
    y_cast = np.array(y_cast)[PA].tolist()
    OB_i = [v for k, v in zip(PA, OB_i) if k]  # OB_j
    _, DIS_i = _subroute_TwoStage_DistanceBased(y_trn, y_insp, OB_i, y_cast)
    td = _subroute_TwoStage_checkDIS(td)
    T = np.percentile(DIS_i, td * 10)  # threshold
    PD = np.array(DIS_i) <= T
    #   #
    # 3. Gathering
    PA = np.where(PA)[0].tolist()
    PD = np.where(PD)[0].tolist()
    PD = [PA[i] for i in PD]
    SH = np.zeros(nb_cls, dtype=DTY_BOL)
    SH[PD] = True
    return SH.tolist(), deepcopy(PA), deepcopy(PD)


# ----------------------------------
#
# Algorithm 5: Two-stage pruning for "DP+AP"
# Input: 𝑅𝑀- a reduced set of base models generated by DP
#             procedure for a specific test sample 𝑥,
#        𝑃- number of base models in 𝑅𝑀,
#        {𝑖1, 𝑖2, . . . , 𝑖𝑃}- index set of base models in 𝑅𝑀
# Output: 𝑅𝑀2- a reduced set of base models,
#         𝑃𝐵2- a pruned bagging ensemble
#
# 1 Given a parameter ta∈ {0, 1, 2, . . . , 9}, compute the
#   threshold 𝑇, which is the ta-th decile value of the set
#               {𝐴𝐶𝑘 | 𝑘 ∈ {𝑖1, 𝑖2, . . . , 𝑖𝑃}}.
# 2 Initialize 𝑅𝑀2 = 0.
# 3 for 𝑘 ∈ {𝑖1, 𝑖2, . . . , 𝑖𝑃} do:
# 4     if 𝐴𝐶𝑘 ≥ 𝑇:
# 5         𝑅𝑀2 = 𝑅𝑀2 ∪{𝑚 𝑘}
# 6 The outcome 𝑃𝐵2(𝑥) of a test sample 𝑥 predicted by the
#   pruned ensemble 𝑃𝐵2 is given as follows:
#       𝑃𝐵2(𝑥) = majority class in {𝑏𝑚(𝑥) | 𝑏𝑚 ∈ 𝑅𝑀2}
#

def TwoStagePruning_DPplusAP(y_trn, y_insp, nb_cls, indices,
                             ta, td, y_cast):
    OB_i = _subroute_TwoStage_OBi(y_trn, indices)
    #   #
    # 1. Distance-based
    _, DIS_i = _subroute_TwoStage_DistanceBased(y_trn, y_insp, OB_i, y_cast)
    td = _subroute_TwoStage_checkDIS(td)
    T = np.percentile(DIS_i, td * 10)
    PD = np.array(DIS_i) <= T
    # 2. Accuracy-based
    y_insp = np.array(y_insp)[PD].tolist()
    OB_i = [v for k, v in zip(PD, OB_i) if k]  # OB_j
    AC_i = _subroute_TwoStage_AccuracyBased(y_trn, y_insp, OB_i)
    ta = _subroute_TwoStage_checkAC(ta)
    T = np.percentile(AC_i, ta * 10)
    PA = np.array(AC_i) >= T
    #   #
    # 3. Gathering
    PD = np.where(PD)[0].tolist()
    PA = np.where(PA)[0].tolist()
    PA = [PD[i] for i in PA]
    SH = np.zeros(nb_cls, dtype=DTY_BOL)
    SH[PA] = True
    return SH.tolist(), deepcopy(PD), deepcopy(PA)


# ----------------------------------
# ----------------------------------
# Previously
# Original Version in the paper


def TwoStagePrev_DistanceBasedPruning(y_trn, y_insp, nb_cls, indices,
                                      td, X_trn, X_val):
    OB_i = _subroute_TwoStage_OBi(y_trn, indices)
    _, DIS_i = _subroute_TwoStage_DistanceBased_inst(X_trn, X_val, OB_i)
    td = _subroute_TwoStage_checkDIS(td)
    T = np.percentile(DIS_i, td * 10)
    P = np.array(DIS_i) <= T
    seq = np.where(P)[0].tolist()
    return P.tolist(), seq


def TwoStagePreviously_AP_plus_DP(y_trn, y_insp, nb_cls, indices,
                                  ta, td, X_trn, X_val):
    # 1. Accuracy-based
    OB_i = _subroute_TwoStage_OBi(y_trn, indices)
    AC_i = _subroute_TwoStage_AccuracyBased(y_trn, y_insp, OB_i)
    ta = _subroute_TwoStage_checkAC(ta)
    T = np.percentile(AC_i, ta * 10)
    PA = np.array(AC_i) >= T
    # 2. Distance-based
    OB_i = [v for k, v in zip(PA, OB_i) if k]
    _, DIS_i = _subroute_TwoStage_DistanceBased_inst(X_trn, X_val, OB_i)
    td = _subroute_TwoStage_checkDIS(td)
    T = np.percentile(DIS_i, td * 10)
    PD = np.array(DIS_i) <= T
    # 3. Gathering
    PA = np.where(PA)[0].tolist()
    PD = [v for k, v in zip(PD, PA) if k]
    SH = np.zeros(nb_cls, dtype=DTY_BOL)
    SH[PD] = True
    return SH.tolist(), PA, PD


def TwoStagePreviously_DP_plus_AP(y_trn, y_insp, nb_cls, indices,
                                  ta, td, X_trn, X_val):
    # 1. Distance-based
    OB_i = _subroute_TwoStage_OBi(y_trn, indices)
    _, DIS_i = _subroute_TwoStage_DistanceBased_inst(X_trn, X_val, OB_i)
    td = _subroute_TwoStage_checkDIS(td)
    T = np.percentile(DIS_i, td * 10)
    PD = np.array(DIS_i) <= T
    # 2. Accuracy-based
    y_insp = np.array(y_insp)[PD].tolist()
    OB_i = [v for k, v in zip(PD, OB_i) if k]
    AC_i = _subroute_TwoStage_AccuracyBased(y_trn, y_insp, OB_i)
    ta = _subroute_TwoStage_checkAC(ta)
    T = np.percentile(AC_i, ta * 10)
    PA = np.array(AC_i) >= T
    # 3. Gathering
    PD = np.where(PD)[0].tolist()
    PA = [v for k, v in zip(PA, PD) if k]
    SH = np.zeros(nb_cls, dtype=DTY_BOL)
    SH[PA] = True
    return SH.tolist(), PD, PA


# ----------------------------------
# ----------------------------------

# =========================================
#  Existing EP Methods
#  Lately / Latest
# =========================================


# ----------------------------------
# contrastive
# ----------------------------------


def contrastive_pruning_lately(name_pru, nb_cls, nb_pru,
                               y_trn, y_val, y_insp, y_cast,
                               alpha=0.5, L=3, R=2, **kwargs):
    # L=3,R=2, ta=4,tb=6
    rho = nb_pru / nb_cls
    ta = int(np.round((1. - rho) * 10))
    td = int(np.round(rho * 10))
    flag = True if len(y_val) == 0 else False
    if flag:
        y_val, y_cast = deepcopy(y_trn), deepcopy(y_insp)

    # ordering-based, MRMC-ordered-aggregation pruning
    if name_pru == "MRMC-MRMR":
        P, seq = procedure_MRMR(y_val, y_cast, nb_cls, nb_pru)
    elif name_pru == "MRMC-MRMC":
        P, seq = procedure_MRMC_ordered_EP(y_val, y_cast, nb_cls, nb_pru)
    elif name_pru == "MRMC-ALL":
        P, seq = procedure_MRMC_EP_with_original_ensemble(
            y_val, y_cast, nb_cls, nb_pru)

    # optimization-based, \citep{li2018mrmr}
    elif name_pru == "MRMREP":
        P, seq, _ = MRMREP_Pruning(y_val, y_cast, nb_cls, nb_pru,
                                   L=L, R=R, alpha=alpha)

    # \citep{cao2018optimizing}
    elif name_pru == "mRMR-EP":
        P, seq = mRMR_ensemble_pruning(y_val, y_cast, nb_cls, nb_pru)
    elif name_pru == "Disc-EP":
        P, seq = Disc_ensemble_pruning(y_val, y_cast, nb_cls, nb_pru)

    # \citep{zhang2019two}
    elif name_pru == "TSP-AP":
        P, seq = TwoStagePruning_AccuracyBasedPruning(
            y_trn, y_insp, nb_cls, ta=ta, **kwargs)
        # NOTICE: might len(seq)==nb_cls
        #   when all individuals have zero error rate.
        # Only 'TSP-AP' would make mistake like that.
        #   #
        if sum(P) == len(seq) >= nb_cls:
            AC_i = np.mean(np.equal(y_cast, y_val), axis=1).tolist()
            T = np.percentile(AC_i, min(max(ta, 0), 9) * 10)
            P = np.array(AC_i) >= T
            seq = np.where(P)[0].tolist()
            del AC_i, T
        #   #   #   #
    elif name_pru == "TSP-DP":
        P, seq = TwoStagePruning_DistanceBasedPruning(
            y_trn, y_insp, nb_cls, td=td, y_cast=y_cast, **kwargs)
    elif name_pru == "TSP-AP+DP":
        P, _, seq = TwoStagePruning_APplusDP(
            y_trn, y_insp, nb_cls, ta=td, td=ta, y_cast=y_cast,
            **kwargs)
    elif name_pru == "TSP-DP+AP":
        P, _, seq = TwoStagePruning_DPplusAP(
            y_trn, y_insp, nb_cls, ta=td, td=ta, y_cast=y_cast,
            **kwargs)
    # # params:
    # "TSP-AP":  y_trn,y_insp, nb_cls, `indices`, ta
    # "TSP-DP":  y_trn,y_insp, nb_cls, `indices`, td, y_cast
    # "TSP-AP+DP", "TSP-DP+AP":
    #            y_trn,y_insp, nb_cls, `indices`, ta,td, y_cast

    # exactly in \citep{zhang2019two}
    elif name_pru == "TSPrev-DP":
        P, seq = TwoStagePrev_DistanceBasedPruning(
            y_trn, y_insp, nb_cls, td=td, **kwargs)
    elif name_pru == "TSPrev-AD":  # "TSPrevADP"
        P, _, seq = TwoStagePreviously_AP_plus_DP(
            y_trn, y_insp, nb_cls, ta=td, td=ta, **kwargs)
    elif name_pru == "TSPrev-DA":  # "TSPrevDAP"
        P, _, seq = TwoStagePreviously_DP_plus_AP(
            y_trn, y_insp, nb_cls, ta=td, td=ta, **kwargs)
    # # PARAMS:
    # "TSPrev-DP":  y_trn,y_insp, nb_cls, `indices`, td, X_trn,X_val
    # "TSPrev-AD", "TSPrev-DA":
    #               y_trn,y_insp, nb_cls, `indices`, ta,td, X_trn,X_val
    # # NOTICE:
    #     ``ta=td, td=ta`` is on purpose, to keep more indivi-
    #     duals, or it might keep only one after pruning.

    else:
        raise UserWarning("Error occurred in `contrastive_pruning_lately`.")

    if len(seq) == 0:
        idx = np.random.randint(nb_cls)
        # NOTICE: <class 'int'>
        P[idx] = True
        seq.append(idx)
    # for rebustness in `TSP-?` etc.

    ys_cast = np.array(y_cast)[P].tolist() if not flag else []
    ys_insp = np.array(y_insp)[P].tolist()
    return ys_insp, ys_cast, P, seq


def contrastive_pruning_lately_validate(
        name_pru, nb_cls, nb_pru, y_trn, y_val, y_insp, y_cast,
        y_pred, coef, clfs, alpha=0.5, L=4, R=3, **kwargs):
    since = time.time()
    ys_insp, ys_cast, P, seq = contrastive_pruning_lately(
        name_pru, nb_cls, nb_pru, y_trn, y_val, y_insp, y_cast,
        alpha=alpha, L=L, R=R, **kwargs)
    time_elapsed = time.time() - since

    ys_pred = np.array(y_pred)[P].tolist()
    opt_coef = np.array(coef)[P].tolist()
    opt_clfs = [cj for i, cj in zip(P, clfs) if i]
    space_cost__ = asizeof(opt_clfs) + asizeof(opt_coef)

    return opt_coef, opt_clfs, ys_insp, ys_cast, ys_pred, \
        time_elapsed, space_cost__, P, seq


# ----------------------------------
# ----------------------------------
