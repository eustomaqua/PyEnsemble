# coding: utf-8
#
# TARGET:
#   Oracle bounds regarding fairness for weighted voting
#


from copy import deepcopy
import math

import numpy as np
# from scipy.special import perm, comb  # 计算排列/组合数
from scipy.special import comb
import scipy.stats as stats

from pyfair.facil.utils_const import check_zero  # DTY_PLT
# from pyfair.facil.metrics_cont import contingency_tab_multiclass
# from pyfair.metrics.normal_perf import contingency_tab_multiclass


# =====================================
# Matlab plot 比较检验


def _regulate_vals(vals, typ='acc'):
    if typ == 'acc':
        return [1. - i for i in vals]
    elif typ == 'err':
        return vals
    raise ValueError("No such test for `{}`".format(typ))
    # assert typ in ['acc', 'err'], "No such test for "+typ


def _regulate_sign(judge, content=r"$\mu$"):  # judgment
    mark = "Accept H0" if judge else "Reject H0"
    if not content:
        return mark  # that is, content==''
    return "{}: equal {}".format(mark, content)


def _avg_and_stdev(vals, k=None):
    if k is None:
        k = len(vals)
    # k: number of values in the designated list
    # aka. SD. def _avg_standard_deviation()
    mu = sum(vals) / k

    sigma2 = [(i - mu)**2 for i in vals]
    # sigma2 = sum(sigma2) / (k - 1.)
    sigma2 = sum(sigma2) / check_zero(k - 1.)

    sigma = math.sqrt(sigma2)
    return mu, sigma2, sigma


# -------------------------------------
# 2.4.1 假设检验
#   对关于单个学习器泛化性能的假设进行检验

# hypothetical_test
# H0: error rate ==epsilon_0


_qbinom_critical_value = {
    0.05: {},
    0.10: {}
}  # [alpha][k]

_qt_critical_value = {
    0.05: {2: 12.706, 5: 2.776, 10: 2.262, 20: 2.093, 30: 2.045},
    0.10: {2: 6.314, 5: 2.132, 10: 1.833, 20: 1.729, 30: 1.699},
}  # [alpha][k]


# 二项检验

def binomial_test(err, m, epsilon, alpha=.05):
    # H0: <= epsilon_0
    # m: number of instances in the test set

    # var_epsilon = max([err])  # not `max(errs)`

    tau_bin = 0.0
    start = math.ceil(epsilon * m + 1)
    for i in range(start, m + 1):
        tmp = epsilon**i * (1 - epsilon)**(m - i)
        tau_bin += comb(m, i) * tmp

    # threshold = 0.0  # ???? TODO
    # mark = _regulate_sign(tao_bin < threshold)
    mark = _regulate_sign(tau_bin < alpha)
    return mark, tau_bin


def t_test(errs, k, epsilon, alpha=.05):
    # k: number of cross-validation
    # k = len(errs)  # hat{epsilon}

    mu, sig2, sigma = _avg_and_stdev(errs, k)

    # sigma = check_zero(sigma)
    # tau_t = (mu - epsilon) * math.sqrt(k) / sigma
    # if abs(mu - epsilon) <= threshold:
    # mark = _regulate_sign(abs(tao_t) <= threshold)

    tau_t = math.sqrt(k) * (mu - epsilon)
    tau_t = tau_t / check_zero(sigma)
    threshold = _qt_critical_value[alpha][k]

    mark = _regulate_sign(tau_t < threshold)
    return mark, tau_t, mu, sig2


# 单样本t检验
#
# 应用场景：
#   对某个样本的均值进行检验，比较是否和总体的均值(自己定）是否存在差异。
#
# 原假设和备择假设：
#   H_0:  \bar{X}  = \mu  样本均值和总体均值相等
#   H_1:  \bar{X}\neq\mu  样本均值和总体均值不等
#
#   p值大于0.05，说明我们不能拒绝原假设(即认为样本均值和总体均值没有显著差异)

def scipy_ttest_for1(errs, epsilon, alpha=.05):
    # aka. def sci_t_test()
    clue, sig = stats.ttest_1samp(errs, epsilon)
    mark = _regulate_sign(sig > alpha)
    return mark, clue, sig


# -------------------------------------
# 2.4.2 交叉验证t检验
# 基本思想：
#   若两个学习器的性能相同，则它们使用相同的训练/测试集得到的
# 测试错误率应相同，即 \epsilon_i^A = \epsilon_i^B


def paired_t_tests(valA, valB, k, alpha=.05):
    # For cross validation, H0: epsilon A==B
    # aka. def CV_paired_t_test()

    delta = [A - B for A, B in zip(valA, valB)]

    mu, _, sigma = _avg_and_stdev(delta, k)  # ,sig2,
    sigma = check_zero(sigma)

    tau_t = math.sqrt(k) * mu
    tau_t = abs(tau_t / sigma)

    threshold = _qt_critical_value[alpha][k]
    mark = _regulate_sign(tau_t < threshold)

    # mu_A, _, s_A = _avg_and_stdev(valA, k)
    # mu_B, _, s_B = _avg_and_stdev(valB, k)
    # return mark, tau_t, (mu_A, mu_B), (s_A, s_B)
    return mark, tau_t


_qt_5x2cv_critical_value = {
    0.05: {5: 2.5706},
    0.10: {5: 2.0150},
}  # [alpha][k]


# def paired_5x2cv_test(valA, valB, alpha=.05):
def paired_5x2cv_test(valA, valB, k=5, alpha=.05):
    # valA/B: [(i_1,i_2)] i=1,..,5  # k=5,
    # val*: [CV1 *2, CV2 *2, CV3 *2, CV4 *2, CV5 *2]
    # k = 5

    idx_i1 = list(range(0, 9, 2))   # [0,2,4,6,8]
    idx_i2 = list(range(1, 10, 2))  # [1,3,5,7,9]

    delta_i1 = [valA[i] - valB[i] for i in idx_i1]
    delta_i2 = [valA[i] - valB[i] for i in idx_i2]

    mu = 0.5 * (delta_i1[0] + delta_i2[0])
    sigma2 = [0.] * k
    for i in range(k):
        LHS = delta_i1[i] - (delta_i1[i] + delta_i2[i]) / 2
        RHS = delta_i2[i] - (delta_i1[i] + delta_i2[i]) / 2
        sigma2[i] = LHS ** 2 + RHS ** 2  # tmp

    denominator = math.sqrt(sum(sigma2) * 0.2)
    tau_t = mu / check_zero(denominator)
    threshold = _qt_5x2cv_critical_value[alpha][k]
    mark = _regulate_sign(tau_t < threshold)
    return mark, tau_t


# 独立样本t检验(双样本t检验)
#
# 应用场景：
#   是针对两组不相关样本（各样本量可以相等也可以不相等），检验它们在均值之间的
# 差异。对于该检验方法而言，我们首先要确定两个总体的方差是否相等，如果不等，先
# 利用levene检验，检验两总体是否具有方差齐性。
#
# 原假设和备择假设：
#   H_0:  \mu_1  = \mu_2  两独立样本的均值相等
#   H_1:  \mu_1\neq\mu_2  两独立样本的均值不等

def scipy_ttest_for2(valA, valB, alpha=.05):
    mk_s2 = stats.levene(valA, valB).pvalue
    mk_s2 = mk_s2 > alpha
    clue, sig = stats.ttest_ind(valA, valB, equal_var=mk_s2)
    mark = _regulate_sign(sig > alpha)

    # mk_s2 = _regulate_sign(mk_s2) + ": sigma2"
    mk_s2 = _regulate_sign(mk_s2, r"$\sigma^2$")
    return mark, clue, sig, mk_s2


# 配对t检验
#
# 应用场景：
#   是针对同一组样本在不同场景下均值之间的差异。检验的是两配对样本差值的均
# 值是否等于0，如果等于0，则认为配对样本之间的均值没有差异，否则存在差异。
#
# 原假设和备择假设：
#   H_0:  \mu_1  = \mu_2  两配对样本的均值相等
#   H_1:  \mu_1\neq\mu_2  两配对样本的均值不等
#
# 与独立样本t检验相比，配对样本T检验要求样本是配对的，两个样本的样本量要相同。

def scipy_ttest_pair(valA, valB, alpha=.05, mode='pl'):
    if mode == 'pl':
        clue, sig = stats.ttest_rel(valA, valB)
    elif mode == 'sg':
        diff = [a - b for a, b in zip(valA, valB)]
        clue, sig = stats.ttest_1samp(diff, 0)

    else:
        raise ValueError("Incorrect mode for paired t_test.")
    mark = _regulate_sign(sig > alpha)
    return mark, clue, sig


# -------------------------------------
# 2.4.3 McNemar检验
#
#   对二分类问题，使用留出法不仅可估计出学习器A和B的测试错误率，还可
# 获得两学习器分类结果的差别，即两者都正确、都错误、一个正确另一个错误
# 的样本数，如“列联表”所示。
#   若我们所做的假设是两学习器性能相同，则应有……

_qchisq_critical_value = {
    0.05: {1: 3.8415},
    0.10: {1: 2.7055}
}  # [alpha][freedom]


# '''
# contingency_table_binary
#     |         | hj = +1 | hj = -1 |
#     | hi = +1 |    a    |    b    |
#     | hi = -1 |    c    |    d    |
#
# contingency_table_multiclass
#     |         | hb == y | hb != y |
#     | ha == y |    a    |    b    |
#     | ha != y |    c    |    d    |
# '''
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


def contingency_tab_multiclass(ha, hb, y):
    # Do NOT use this function to calcuate!
    a = np.sum(np.logical_and(
        np.equal(ha, y), np.equal(hb, y)))          # e_{00}, N^{11}
    c = np.sum(np.logical_and(
        np.not_equal(ha, y), np.equal(hb, y)))      # e_{01}, N^{01}
    b = np.sum(np.logical_and(
        np.equal(ha, y), np.not_equal(hb, y)))      # e_{10}, N^{10}
    d = np.sum(np.logical_and(
        np.not_equal(ha, y), np.not_equal(hb, y)))  # e_{11}, N^{11}
    return int(a), int(b), int(c), int(d)


def McNemar_test(ha, hb, y, k=2, alpha=.05):
    # H0: the performance of these two classifiers are the same
    # k=2 is because there are two algorithms being compared
    _, e10, e01, _ = contingency_tab_multiclass(ha, hb, y)
    # e00, e10, e01, e11 = contingency_tab_multiclass(ha, hb, y)

    numerator = abs(e01 - e10) - 1
    numerator = numerator ** 2
    denominator = e01 + e10
    # chi-square
    tau_chi2 = numerator / check_zero(denominator)

    threshold = _qchisq_critical_value[alpha][k - 1]
    mark = _regulate_sign(tau_chi2 < threshold)
    return mark, tau_chi2


# -------------------------------------
# 2.4.4 Friedman检验 与 Nemenyi后续检验
#
#   交叉验证t检验和McNemar检验都是在一个数据集上比较两个算法的
# 性能，而在很多时候，我们会在一组数据集上对多个算法进行比较。当
# 有多个算法参与比较时，一种做法是在每个数据集上分别列出两两比较的
# 结果，而在两两比较时可使用前述方法；另一种方法更为直接，即使用
# 基于算法排序的Friedman检验。


# 算法平均序值表，用于算法比较
#
#   descend: 从大到小排序，值大的排名高
#   ascend : 从小到大排序，值小的排名高

def _Friedman_sequential(U, mode='ascend', dim=2):
    # ordinal value / successive/sequential
    if isinstance(U, list):
        U = np.array(U, 'float')
    # U: np.ndarray, size=(datasets, algorithms)

    index = np.argsort(U, axis=dim - 1)
    if mode == 'descend':
        index = index[:, :: -1]
    row, col = np.shape(U)

    rank = np.zeros_like(U)
    for m in range(row):
        for n in range(col):
            rank[m, index[m, n]] = n
    rank += 1

    # return np.asarray(rank, 'int')
    return rank


def _Friedman_successive(U, rank):
    # average ordering/ranking
    row, col = np.shape(U)

    idx_bar = deepcopy(rank)
    # idx_bar = rank.copy().astype('float')
    # idx_bar = np.array(rank, dtype='float')

    for m in range(row):
        uniq = np.unique(U[m, :])
        if len(uniq) == col:
            continue

        tmp = deepcopy(rank[m])  # Notice!
        # for k in range(len(uniq)):
        for k, _ in enumerate(uniq):
            loca = U[m] == uniq[k]
            if np.sum(loca) > 1:
                tmp[loca] = np.mean(rank[m, loca])
            del loca
        idx_bar[m, :] = tmp

    return idx_bar


_qf_critical_value = {
    0.05: {
        4: [10.128, 5.143, 3.863, 3.259, 2.901, 2.661, 2.488, 2.355, 2.250],
        5: [7.709, 4.459, 3.490, 3.007, 2.711, 2.508, 2.359, 2.244, 2.153,
            2.077, 2.014, 1.960, 1.913, 1.873, 1.836, 1.804, 1.775, ],
        7: [5.987, 3.885, 3.160, 2.776, 2.534, 2.364, 2.237, 2.138, 2.059,
            1.993, 1.937, 1.889, 1.848, 1.811, 1.779, 1.750, 1.724],
        8: [5.591, 3.739, 3.072, 2.714, 2.485, 2.324, 2.203, 2.109, 2.032],
        # 9: [5.318, 3.634, 3.009, 2.668, 2.449, 2.295, 2.178, 2.087, 2.013],
        10: [5.117, 3.555, 2.960, 2.634, 2.422, 2.272, 2.159, 2.070, 1.998],
        15: [4.600, 3.340, 2.827, 2.537, 2.346, 2.209, 2.104, 2.022, 1.955],
        20: [4.381, 3.245, 2.766, 2.492, 2.310, 2.179, 2.079, 2.000, 1.935],

        9: [5.318, 3.634, 3.009, 2.668, 2.449, 2.295, 2.178, 2.087, 2.013,
            1.951, 1.899, 1.854, 1.815, 1.781, 1.750, 1.723, 1.698, 1.676, ],
        11: [4.965, 3.493, 2.922, 2.606, 2.400, 2.254, 2.143, 2.056, 1.986,
             1.927, 1.877, 1.834, 1.796, 1.763, 1.734, 1.707, 1.683, 1.661],
        25: [4.260, 3.191, 2.732, 2.466, 2.290, 2.162, 2.064, 1.987, 1.923,
             1.870, 1.825, 1.786, 1.752, 1.721, 1.694, 1.670, 1.648, 1.628],
        55: [4.020, 3.080, 2.660, 2.413, 2.247, 2.127, 2.034, 1.960, 1.899,
             1.848, 1.805, 1.767, 1.734, 1.705, 1.679, 1.655, 1.634, 1.614],
        165: [3.899, 3.023, 2.623, 2.386, 2.225, 2.108, 2.018, 1.945, 1.886,
              1.836, 1.794, 1.757, 1.725, 1.696, 1.670, 1.647, 1.626, 1.607],
        275: [3.876, 3.012, 2.616, 2.380, 2.221, 2.104, 2.014, 1.943, 1.884,
              1.834, 1.792, 1.755, 1.723, 1.694, 1.669, 1.646, 1.625, 1.606],
        825: [3.853, 3.001, 2.609, 2.375, 2.216, 2.100, 2.011, 1.940, 1.881,
              1.832, 1.790, 1.753, 1.721, 1.693, 1.667, 1.644, 1.624, 1.605],


        143: [3.908, 3.028, 2.626, 2.388, 2.227, 2.109, 2.019, 1.947, 1.887,
              1.837, 1.795, 1.758, 1.725, 1.697, 1.671, 1.648, 1.627, 1.608],
        429: [3.863, 3.006, 2.612, 2.377, 2.218, 2.102, 2.013, 1.941, 1.882,
              1.833, 1.791, 1.754, 1.722, 1.693, 1.668, 1.645, 1.624, 1.605],
    },
    0.10: {
        4: [5.538, 3.463, 2.813, 2.480, 2.273, 2.130, 2.023, 1.940, 1.874],
        5: [4.545, 3.113, 2.606, 2.333, 2.158, 2.035, 1.943, 1.870, 1.811,
            1.763, 1.721, 1.686, 1.655, 1.628, 1.603, 1.582, 1.562, ],
        7: [3.776, 2.807, 2.416, 2.195, 2.049, 1.945, 1.865, 1.802, 1.750,
            1.707, 1.670, 1.639, 1.611, 1.586, 1.564, 1.545, 1.527],
        8: [3.589, 2.726, 2.365, 2.157, 2.019, 1.919, 1.843, 1.782, 1.733],
        # 9: [3.458, 2.668, 2.327, 2.129, 1.997, 1.901, 1.827, 1.768, 1.720],
        10: [3.360, 2.624, 2.299, 2.108, 1.980, 1.886, 1.814, 1.757, 1.710],
        15: [3.102, 2.503, 2.219, 2.048, 1.931, 1.845, 1.779, 1.726, 1.682],
        20: [2.990, 2.448, 2.182, 2.020, 1.909, 1.826, 1.762, 1.711, 1.668],

        9: [3.458, 2.668, 2.327, 2.129, 1.997, 1.901, 1.827, 1.768, 1.720,
            1.680, 1.645, 1.615, 1.589, 1.566, 1.545, 1.526, 1.509, 1.494, ],
        11: [3.285, 2.589, 2.276, 2.091, 1.966, 1.875, 1.804, 1.748, 1.702,
             1.663, 1.630, 1.601, 1.576, 1.553, 1.533, 1.515, 1.499, 1.484],
        25: [2.927, 2.417, 2.161, 2.004, 1.896, 1.815, 1.753, 1.702, 1.661,
             1.625, 1.595, 1.569, 1.546, 1.525, 1.506, 1.490, 1.474, 1.461],
        55: [2.801, 2.352, 2.118, 1.971, 1.869, 1.792, 1.733, 1.684, 1.644,
             1.611, 1.581, 1.556, 1.534, 1.514, 1.496, 1.479, 1.465, 1.451],
        165: [2.736, 2.319, 2.095, 1.953, 1.854, 1.780, 1.722, 1.675, 1.636,
              1.603, 1.574, 1.549, 1.527, 1.508, 1.490, 1.474, 1.460, 1.446],
        275: [2.724, 2.312, 2.090, 1.950, 1.851, 1.778, 1.720, 1.673, 1.634,
              1.601, 1.573, 1.548, 1.526, 1.506, 1.489, 1.473, 1.459, 1.445],
        825: [2.712, 2.306, 2.086, 1.947, 1.849, 1.775, 1.718, 1.671, 1.632,
              1.599, 1.571, 1.546, 1.525, 1.505, 1.488, 1.472, 1.458, 1.444],

        143: [2.741, 2.321, 2.097, 1.955, 1.855, 1.781, 1.723, 1.676, 1.636,
              1.603, 1.575, 1.550, 1.528, 1.508, 1.490, 1.474, 1.460, 1.447],
        429: [2.717, 2.309, 2.088, 1.948, 1.850, 1.776, 1.719, 1.672, 1.633,
              1.600, 1.572, 1.547, 1.525, 1.506, 1.488, 1.472, 1.458, 1.445],
    },
}  # [alpha][N][k-2]  # np.nan in np.array()
# R command:  qf(1-alpha, k-1, (k-1)*(N-1))

_qtukey_critical_value = {
    0.05: [1.960, 2.344, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164,
           3.219, 3.268, 3.313, 3.354, 3.391, 3.426, 3.458, 3.489, 3.517,
           3.544, 3.569, 3.593, 3.616, 3.637, 3.658, 3.678, 3.696, 3.714,
           3.732, 3.749, 3.765, 3.780, 3.795, 3.810, ],

    0.10: [1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920,
           2.978, 3.030, 3.077, 3.120, 3.159, 3.196, 3.230, 3.261, 3.291,
           3.319, 3.346, 3.371, 3.394, 3.417, 3.439, 3.459, 3.479, 3.498,
           3.516, 3.533, 3.550, 3.567, 3.582, 3.597, ],

}  # [alpha][k-2]
# R command:  qtukey(1-alpha, k, Inf)/sqrt(2)


def Friedman_init(U, mode='descend'):
    rank = _Friedman_sequential(U, mode, dim=2)
    idx_bar = _Friedman_successive(U, rank)
    # rank = rank.astype('int')
    return rank, idx_bar


def Friedman_test(idx_bar, alpha=.05):
    # aka. def compare_algorithms_average_index()
    #               U, mode='descend', alpha=.05)

    N, k = np.shape(idx_bar)  # U)
    r_i = np.mean(idx_bar, axis=0)
    # mu = (k + 1.) / 2
    # sig2 = (k**2 - 1.) / 12

    # '''
    # tau_chi2 = [(i - (k + 1) / 2)**2 for i in r_i]
    # tau_chi2 = sum(tau_chi2) * (12 * N) * (k - 1)
    # tau_chi2 = tau_chi2 / k / check_zero(k**2 - 1)
    # '''

    tau_chi2 = sum([i**2 for i in r_i])
    RHS = k * (k + 1)**2 / 4
    LHS = 12 * N / (k * (k + 1))
    tau_chi2 = LHS * (tau_chi2 - RHS)

    numerator = (N - 1) * tau_chi2
    denominator = N * (k - 1) - tau_chi2
    tau_F = numerator / check_zero(denominator)

    threshold = _qf_critical_value[alpha][N][k - 2]
    mark = _regulate_sign(tau_F < threshold)
    # CD = Nememyi_posthoc_test(N, k, alpha)
    return mark, tau_F, tau_chi2


def Nememyi_posthoc_test(idx_bar, alpha=.05):
    N, k = np.shape(idx_bar)

    CD = k * (k + 1) / (6 * N)
    CD = math.sqrt(CD)
    threshold = _qtukey_critical_value[alpha][k - 2]
    CD = threshold * CD
    return CD, threshold


# =====================================
# Matlab plot 2.3 性能度量


# -------------------------------------
# 2.3.1 错误率与精度


# -------------------------------------
# 2.3.2 查准率、查全率与F1


# -------------------------------------
# 2.3.3 ROC与AUC


# -------------------------------------
# 2.3.4 代价敏感错误率与代价曲线


# =====================================
# Paired t-test


# -------------------------------------
# two-tailed paired $t$-test
# G_1, G_2: i.e., Goals = (avg, std)

_enc_rez = 2


def _decode_sign(mark):
    judge = mark.startswith("Accept H0")
    return "T" if judge else "F"  # "Tie/True"


def _encode_sign_math(avg, std, rez=2):
    # Resolution, res/dpi
    if rez == 2:
        sign = "${:.2f}\\!\\pm\\!{:.2f}$".format(avg, std)
    elif rez == 4:
        sign = "${:.4f}\\!\\pm\\!{:.4f}$".format(avg, std)
    elif rez == 1:
        sign = "${:.1f}\\!\\pm\\!{:.1f}$".format(avg, std)
    elif rez == 3:
        sign = "${:.3f}\\!\\pm\\!{:.3f}$".format(avg, std)
    else:  # rez==6
        sign = "${:f}\\!\\pm\\!{:f}$".format(avg, std)
    return sign


def _encode_sign_text(avg, std, rez=2):
    if rez == 2:
        sign = "{:.2f}\\topelement$\\pm${:.2f}".format(avg, std)
    elif rez == 4:
        sign = "{:.4f}\\topelement$\\pm${:.4f}".format(avg, std)
    elif rez == 1:
        sign = "{:.1f}\\topelement$\\pm${:.1f}".format(avg, std)
    elif rez == 3:
        sign = "{:.3f}\\topelement$\\pm${:.3f}".format(avg, std)
    else:  # rez==6
        sign = "{:f}\\topelement$\\pm${:f}".format(avg, std)
    # sign = sign.replace("\topelement{", "").replace("}", "")
    return sign


def _encode_sign(avg, std, rez=2, mode="text"):
    if mode == "text":
        sign = _encode_sign_text(avg, std, rez)
    elif mode == "math":
        sign = _encode_sign_math(avg, std, rez)
    return sign


def _cmp_avg_descend(G_A, G_B):
    avg_A, std_A = G_A
    avg_B, std_B = G_B
    if avg_A > avg_B:
        mark = "W"
    elif avg_A < avg_B:
        mark = "L"
    elif std_A < std_B:
        mark = "W"
    elif std_A > std_B:
        mark = "L"
    else:
        mark = "T"
    return mark


def _cmp_avg_ascend(G_A, G_B):
    avg_A, std_A = G_A
    avg_B, std_B = G_B
    if avg_A < avg_B:
        mark = "W"
    elif avg_A > avg_B:
        mark = "L"
    elif std_A < std_B:
        mark = "W"
    elif std_A > std_B:
        mark = "L"
    else:
        mark = "T"
    return mark


def cmp_paired_avg(G_A, G_B, mode="descend"):
    if mode == "descend":
        mark = _cmp_avg_descend(G_A, G_B)
    elif mode == "ascend":
        mark = _cmp_avg_ascend(G_A, G_B)
    return mark


def cmp_paired_wtl(G_A, G_B, mk_mu, mk_s2, mode="ascend"):
    std_A = G_A[1]  # avg_A, std_A = G_A
    std_B = G_B[1]  # avg_B, std_B = G_B
    if _decode_sign(mk_s2) == "T":
        if _decode_sign(mk_mu) == "T":
            mark = "T"  # tie
        else:
            mark = cmp_paired_avg(G_A, G_B, mode)
    elif _decode_sign(mk_s2) == "F":
        if _decode_sign(mk_mu) == "T":
            if std_A < std_B:
                mark = "W"
            elif std_A > std_B:
                mark = "L"
            else:
                mark = "T"
        elif _decode_sign(mk_mu) == "F":
            mark = cmp_paired_avg(G_A, G_B, mode)
    return mark


def comp_t_init(valA, valB):
    # def paired_t_init()
    k = len(valA)                 # error rate /accuracy
    assert k == len(valB), "cross_validation number error"
    avg_A, _, std_A = _avg_and_stdev(valA, k)
    avg_B, _, std_B = _avg_and_stdev(valB, k)
    G_A = (avg_A, std_A)
    G_B = (avg_B, std_B)
    sign_A = _encode_sign(avg_A, std_A, _enc_rez)
    sign_B = _encode_sign(avg_B, std_B, _enc_rez)
    return sign_A, sign_B, G_A, G_B  # , k


def comp_t_sing(val, k=None, rez=None):
    if k is None:
        k = len(val)
    avg_, _, std_ = _avg_and_stdev(val, k)
    Goal = (avg_, std_)
    if rez is None:
        rez = _enc_rez
    sign = _encode_sign(avg_, std_, rez)
    return sign, Goal


def comp_t_prep(valA, valB, alpha=.05, method="manual"):
    k = len(valA)           # error rate
    assert k == len(valB), "cross_validation number error"

    if method == "manual":  # customise
        mk_mu, _ = paired_t_tests(valA, valB, k, alpha)
    elif method == "scipy":
        mk_mu, _, _ = scipy_ttest_pair(valA, valB, alpha)

    mk_s2 = stats.levene(valA, valB)
    mk_s2 = mk_s2.pvalue > alpha
    mk_s2 = _regulate_sign(mk_s2, r"$\sigma^2$")
    return mk_mu, mk_s2


# 方差齐性检验


# -------------------------------------


# =====================================


# -------------------------------------
# 皮尔逊相关系数
# Pearson_correlation_coefficient
# https://blog.csdn.net/huangfei711/article/details/78456165
# https://www.cnblogs.com/BlogNetSpace/p/18265455


def Pearson_correlation(X, Y):
    # or np.corrcoef(X, Y)[1, 0]
    # or np.corrcoef(Y, X)[1, 0]

    # Covariance 协方差
    Xi_bar = np.array(X) - np.mean(X)
    Yi_bar = np.array(Y) - np.mean(Y)
    tmp = np.multiply(Xi_bar, Yi_bar)  # =Xi_bar*Yi_bar
    n = len(tmp)
    cov = float(np.sum(tmp)) / (n - 1.)

    numerator = float(np.sum(tmp))     # ↓ denominator
    denom_X = np.sum(Xi_bar * Xi_bar)  # sum(Xi_bar**2)
    denom_Y = np.sum(Yi_bar * Yi_bar)  # sum(Yi_bar**2)
    denominator = np.sqrt(denom_X) * np.sqrt(denom_Y)
    denominator = check_zero(float(denominator))
    return numerator / denominator, cov


# -------------------------------------
