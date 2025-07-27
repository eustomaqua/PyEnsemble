# coding: utf-8
#
# Aim to provide:
#   Confusion matrices
#


import numpy as np
import numba


# -------------------------------------
# Contingency table
#   for binary classification
#
# |  True label  |      Prediction     |
# |              | Positive | Negative |
# | Positive (1) |    TP    |    FN    |
# | Negative (0) |    FP    |    TN    |

# Contingency table (binary)
# |True label `y`| Prediction `f(x)`   |


@numba.jit(nopython=True)
def contingency_tab_bi(y, y_hat, pos=1):
    # For one single classifier
    tp = np.sum((y == pos) & (y_hat == pos))  # a
    fn = np.sum((y == pos) & (y_hat != pos))  # b
    fp = np.sum((y != pos) & (y_hat == pos))  # c
    tn = np.sum((y != pos) & (y_hat != pos))  # d
    return tp, fp, fn, tn
    # return tp, fn, fp, tn


# input: np.ndarray, not list
#
# P[f()=1 | y=0] = fp/(fp+tn) =g_Cm[2]/g_Cm[2+0]
# P[f()=1 | y=1] = tp/(tp+fn) =g_Cm[0]/g_Cm[0+2]


# # For multi-class classification
# '''
# 机器学习 周志华
# 二分类代价矩阵
# |True Label|      Prediction       |
# |          | Class 0   | Class 1   |
# | Class 0  |    0      | cost_{01} |
# | Class 1  | cost_{10} |    0      |
#
# contingency_table_multi
#   |             |hb= c_0 |hb= c_1 |hb= c_{n_c-1}|
#   |ha= c_0      | C_{00} | C_{01} | C_{0?}      |
#   |ha= c_1      | C_{10} | C_{11} | C_{1?}      |
#   |ha= c_{n_c-1}| C_{?0} | C_{?1} | C_{??}      |
# contingency_table_{?} when ?==2
#   |            |hb!=y, hb=-1|hb==y, hb=1|
#   |ha!=y, ha=-1|  d  /TN    |  c  /FP   |
#   |ha==y, ha==1|  b  /FN    |  a  /TP   | when ha is yt
#
# # 多分类有几种：
# # 第一种：用这个样本分对了还是分错了区分
# # 第二种：用一个确定的作为pos，其他类都是neg，转化成二分类再做
# # 第三种：变成 NxN 矩阵
#
# 分类结果混淆矩阵（多分类）
# '''


def contg_tab_mu_type3(y, y_hat, vY):
    # def contingency_tab_mu():
    dY = len(vY)
    Cij = np.zeros(shape=(dY, dY), dtype='int')  # DTY_INT)
    for i in range(dY):
        for j in range(dY):
            Cij[i, j] = np.sum((y == vY[i]) & (y_hat == vY[j]))
    return Cij  # Cij.copy(), np.ndarray


def contg_tab_mu_merge(Cij, vY, pos=1):
    k = vY.index(pos)  # idx
    tp = Cij[k][k]
    fn = np.sum(Cij[k]) - Cij[k, k]
    fp = np.sum(Cij[:, k]) - Cij[k, k]

    Kij = Cij.copy()
    Kij[k] = 0
    Kij[:, k] = 0
    tn = np.sum(Kij)
    return tp, fp, fn, tn


# @numba.jit(nopython=True)
# def contg_tab_multi_type3(h, hp, vY):
# def contg_tab_multi_merge(Cij, vY, pos=1):


# '''
# Machine learning, Zhi-Hua Zhou
# |          | hi = pos | hi = neg |
# | hj = pos |  a       |  c       |
# | hj = neg |  b       |  d       |
# i.e.,
# |          | hj = pos | hj = neg |
# | hi = pos |  a       |  b       |
# | hi = neg |  c       |  d       |
#
# contg_tab_multi_type3()
# contg_tab_multi_merge()
# 加起来就是 contingency_zh()
# '''


contg_tab_mu_type2 = contingency_tab_bi


# @numba.jit(nopython=True)
# def contg_tab_multi_type2(h, hp, pos=1):
#     # namely, def contingency_zh()


# '''
# Kuncheva2003measures
# |              |hk correct (1)|hk wrong (0)|
# |hi correct (1)|   N^{11}     |   N^{10}   |
# |hi  wrong  (0)|   N^{01}     |   N^{00}   |
# Total, N= N^{00}+N^{01}+N^{10}+N^{11}
#
# i.e.,
# |          |hk correct|hk wrong|
# |hi correct|   tp     |   fn   | or | a | b |
# |hi  wrong |   fp     |   tn   |    | c | d |
#
# McNemar test
# |             |Alg B correct|Alg B  wrong |
# |Alg A correct|    e_{00}   |    e_{10}   |
# |Alg A  wrong |    e_{01}   |    e_{11}   |
# '''


def contg_tab_mu_type1(y, ha, hb):
    za = np.array(y == ha, dtype='int')  # DTY_INT)
    zb = np.array(y == hb, dtype='int')  # DTY_INT)

    tp = np.sum((za == 1) & (zb == 1))  # N^{11} # e_{00}
    fn = np.sum((za == 1) & (zb != 1))  # N^{10} # e_{10}
    fp = np.sum((za != 1) & (zb == 1))  # N^{01} # e_{01}
    tn = np.sum((za != 1) & (zb != 1))  # N^{00} # e_{11}
    return tp, fp, fn, tn


# def contg_tab_multi_type1(h, ha, hb):
#     # namely, def contingency_ku()


# -------------------------------------
# Performance metrics (cont.)


# 在 n 个二分类混淆矩阵上综合考察

@numba.jit(nopython=True)
def calc_confusion(y, fx, cv=5, pos=1):
    # cross validation
    #  y.shape: (nb_inst,)
    # fx.shape: (cv, nb_inst)
    return list(map(contingency_tab_bi,
                    [y] * cv, fx, [pos] * cv))


# def calc_macro_score(confusion, cv=5):
# def calc_micro_score(confusion, cv=5):


# 分类结果混淆矩阵
# from hfm.metrics.contingency_mat import contingency_tab_bi
#
# @numba.jit(nopython=True)
# def calc_confusion(y, fx, cv=5, pos=1, neg=0):
#     confusion = list(map(contingency_tab_bi,
#                     [y] * cv, fx, [pos] * cv, [neg] * cv))
#     tp, fp, fn, tn = zip(*confusion)
#     return tp, fp, fn, tn
#     # return [contingency_table(y, t, pos, neg) for t in fx]
#     return list(map(contingency_table,
#                     [y] * cv, fx, [pos] * cv, [neg] * cv))


# # --------------------------
# # 分类结果混淆矩阵（二分类）
#
# '''
# 机器学习 周志华
# |True Label| Prediction          |
# |          | Positive | Negative |
# | Positive | TP       | FN       |
# | Negative | FP       | TN       |
#
# 集成学习 周志华
# |          | hi = pos | hi = neg |
# | hj = pos |  a       |  c       |
# | hj = neg |  b       |  d       |
# i.e.,
# |          | hj = pos | hj = neg |
# | hi = pos |  a       |  b       |
# | hi = neg |  c       |  d       |
# '''
#
# # metric_cont.py, metrics_contg.py
# hfm/metrics/contingency_mat.py
# from hfm.utils.verifiers import DTY_INT
#     # ''' if neg=0
#     # return tn, fn, fp, tp
#     # '''
#     # # return tp, fn, fp, tn
