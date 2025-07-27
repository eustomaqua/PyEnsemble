# coding: utf-8
#
# Aim to provide:
#   Performance-based metrics
#   primarily for binary classification
#


import numpy as np
import numba
from pyfair.facil.utils_const import check_zero, non_negative


def comp_accuracy(y, hx):
    t = np.mean(np.equal(y, hx))
    return float(t)


def comp_error_rate(y, hx):
    t = np.mean(np.not_equal(y, hx))
    return float(t)


# -------------------------------------
# Performance metrics
#
# After having the confusion matrix,
# for one single classifier (binary)

# '''
# |                      |prediction is positive|predict negative|
# |true label is positive| true positive (TP)|false negative (FN)|
# |true label is negative|false positive (FP)| true negative (TN)|
# '''


# TP, FP, FN, TN
# --------------------------
# NB. some doesn't work for multi-class!


# 精度 /正确率
# Accuracy
# @numba.jit(nopython=True)
def calc_accuracy(tp, fp, fn, tn):
    n = float(tp + fp + fn + tn)
    tmp = (tp + tn) / check_zero(n)
    return float(tmp)


# 错误率
# Error rate = 1 - accuracy
def calc_error_rate(tp, fp, fn, tn):
    n = float(tp + fp + fn + tn)
    # return 1. - (tp + tn) / check_zero(n)
    tmp = (fp + fn) / check_zero(n)
    return float(tmp)


# 查准率 P
# Precision
def calc_precision(tp, fp, fn, tn):
    denominator = float(tp + fp)
    tmp = tp / check_zero(denominator)
    return float(tmp)


# 查全率 R
# Recall
def calc_recall(tp, fp, fn, tn):
    denominator = float(tp + fn)
    tmp = tp / check_zero(denominator)
    return float(tmp)


# F1 度量
# F1 measure
def calc_f1_score(tp, fp, fn, tn):
    n = float(tp + fp + fn + tn)
    denominator = n + tp - tn
    tmp = 2 * tp / check_zero(denominator)
    return float(tmp)


# F1 度量的一般形式
# F1
def calc_f_beta(p, r, beta=1):
    denominator = beta**2 * p + r
    numerator = (1. + beta**2) * p * r
    tmp = numerator / check_zero(denominator)
    return float(tmp)


# ROC curve, AUC
#
# def calc_auc_score(y, y_hat):
#     from sklearn import metrics
#     return metrics.roc_auc_score(y, y_hat)


def calc_macro_score(confusion, cv=5):
    tp, fp, fn, tn = zip(*confusion)
    p_score = list(map(calc_precision, tp, fp, fn, tn))
    r_score = list(map(calc_recall, tp, fp, fn, tn))
    # f1_score = list(map(calc_f1_score, tp, fp, fn, tn))
    macro_p = sum(p_score) / float(cv)
    macro_r = sum(r_score) / float(cv)

    numerator = 2. * macro_p * macro_r
    denominator = float(macro_p + macro_r)
    macro_f1 = numerator / check_zero(denominator)
    # return macro_p, macro_r, macro_f1
    return float(macro_p), float(macro_r), float(macro_f1)


def calc_micro_score(confusion, cv=5):
    tp, fp, fn, _ = zip(*confusion)  # ,tn
    tp_bar = sum(tp) / float(cv)
    fp_bar = sum(fp) / float(cv)
    fn_bar = sum(fn) / float(cv)
    # tn_bar = sum(tn) / cv

    micro_p = float(tp_bar + fp_bar)
    micro_p = tp_bar / check_zero(micro_p)
    micro_r = float(tp_bar + fn_bar)
    micro_r = tp_bar / check_zero(micro_r)

    numerator = 2. * micro_p * micro_r
    denominator = float(micro_p + micro_r)
    micro_f1 = numerator / check_zero(denominator)
    # return micro_p, micro_r, micro_f1
    return float(micro_p), float(micro_r), float(micro_f1)


# -------------------------------------
# Performance metrics
# 对单个分类器
#
# expect smaller fpr|fnr, larger tnr|tpr


# 真正率/召回率/命中率 hit rate
# Recall
# True Positive Rate, TPR
def calc_tpr(tp, fp, fn, tn):
    return calc_recall(tp, fp, fn, tn)


# 假正率/1-特异度, 误报/虚警/误检率 false alarm
# False Positive Rate, FPR
def calc_fpr(tp, fp, fn, tn):
    denominator = float(tn + fp)
    return fp / check_zero(denominator)


# 漏报率 miss rate, 也称为漏警率、漏检率
# 1-recall
def calc_fnr(tp, fp, fn, tn):
    denominator = float(tp + fn)
    tmp = fn / check_zero(denominator)
    return float(tmp)


# Recall
# True Positive Rate, TPR
def calc_sensitivity(tp, fp, fn, tn):
    return calc_recall(tp, fp, fn, tn)


# 特异度 = 1-假正率
# 1.-FPR
def calc_specificity(tp, fp, fn, tn):
    # denominator = float(fp + tn)
    denominator = float(tn + fp)
    tmp = tn / check_zero(denominator)
    return float(tmp)


def calc_tnr(tp, fp, fn, tn):
    return calc_specificity(tp, fp, fn, tn)


# -------------------------------------
# Data imbalance
#
# ref:
# https://support.sas.com/resources/papers/proceedings17/0942-2017.pdf
# g-mean = sensitivity * specificity**2


# def calc_geometric_mean(tp, fp, fn, tn):
#   sensitivity = calc_sensitivity(tp, fp, fn, tn)
#   specificity = calc_specificity(tp, fp, fn, tn)
#
@numba.jit(nopython=True)
def imba_geometric_mean(sen, spe):
    g_mean = np.sqrt(sen * spe)
    return float(g_mean)


# def calc_discriminant_power(tp, fp, fn, tn):
#   sensitivity = calc_sensitivity(tp, fp, fn, tn)
#   specificity = calc_specificity(tp, fp, fn, tn)
#
def imba_discriminant_power(sen, spe):
    """ The discriminant power assesses how well a classifier
    distinguishes between the positive and negative cases. The
    classifier is considered a poor classifier if DP < 1, limited
    if DP < 2, fair if DP < 3 and good in other cases.
    """
    X = sen / check_zero(1 - sen)
    Y = spe / check_zero(1 - spe)
    # numerator = np.log(X) + np.log(Y)
    numer_1 = check_zero(non_negative(X))
    numer_2 = check_zero(non_negative(Y))
    numerator = np.log(numer_1) + np.log(numer_2)
    denominator = np.sqrt(3) / np.pi
    dp = numerator * denominator
    return float(dp)


@numba.jit(nopython=True)
def imba_balanced_accuracy(sen, spe):
    return sen * spe / 2.


def imba_Matthew_s_cc(tp, fp, fn, tn):
    """ The value ranges from -1 to +1 with a value of +1
    representing a perfect prediction, 0 as no better than
    random prediction and -1 the worst possible prediction.
    """
    denom_1 = (tp + fp) * (tp + fn)
    denom_2 = (tn + fp) * (tn + fn)
    denominator = np.sqrt(denom_1 * denom_2)
    numerator = float(tp * tn - fp * fn)
    mcc = numerator / check_zero(denominator)
    return float(mcc)


def imba_Cohen_s_kappa(tp, fp, fn, tn):
    """ In a similar fashion to the MCC, kappa takes on
    values from -1 to +1, with a value of 0 meaning there
    is no agreement between the actual and classified
    classes. A value of 1 indicates perfect concordance
    of the model prediction and the actual classes and a
    value of −1 indicates total disagreement between
    prediction and the actual.
    """
    total_acc = calc_accuracy(tp, fp, fn, tn)

    numer_1 = (tn + fp) * (tn + fn)
    numer_2 = (fn + tp) * (fp + tp)
    numerator = numer_1 + numer_2
    denom = float(tp + tn + fp + fn) ** 2
    random_acc = numerator / check_zero(denom)

    numerator = total_acc - random_acc
    denom = 1. - random_acc
    kappa = numerator / check_zero(denom)
    return float(kappa), random_acc


@numba.jit(nopython=True)
def imba_Youden_s_index(sen, spe):
    """ A higher value of 𝛾 is an indication of a good
    performing classifier.
    """
    # sen, spe  \in [0,1]
    # sen+spe-1 \in [-1,1]
    gamma = sen - (1. - spe)
    return float(gamma)


def imba_likelihoods(sen, spe):
    """ Based on the definitions, a higher positive
    likelihood ratio and a lower negative likelihood
    ratio is an indication of a good performance on
    positive and negative classes respectively.
    """
    lr_pos = sen / check_zero(1 - spe)
    lr_neg = (1 - sen) / check_zero(spe)
    return lr_pos, lr_neg


# -------------------------------------
#


# metrics_perf.py, performance.py
# core/oracle_metric.py
#
# TARGET:
#   works for binary classification only

# # performance.py, confusion_mat.py
# hfm/metrics/excl_perf_bin.py
# from hfm.utils.verifiers import check_zero
# ---------------------
#
