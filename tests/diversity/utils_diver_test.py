# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from pyensemble.diversity.utils_diver import contingency_table
from pyensemble.diversity.utils_diver import multiclass_contingency_table
from pyensemble.diversity.utils_diver import number_individuals_correct



def case_contingency_table(m, L):
    # m:  number of instances / samples
    # L:  number of labels / classes
    if L > 0:
        hi = np.random.randint(2, size=m).tolist()
        hj = np.random.randint(2, size=m).tolist()
    else:
        hi = np.random.randint(2, size=m) * 2 - 1
        hj = np.random.randint(2, size=m) * 2 - 1
        hi, hj = hi.tolist(), hj.tolist()
    #   #
    a, b, c, d = contingency_table(hi, hj)
    #
    # print("hi", hi)
    # print("hj", hj)
    assert a + b + c + d == m


def test_contingency_table():
    case_contingency_table(m=11, L=2)
    case_contingency_table(m=21, L=-1)



def test_multiclass_contingency_table_part1():
    m = 31  # number of instances / samples
    L = 4   #  number of labels / classes
    hi = np.random.randint(L, size=m).tolist()
    hj = np.random.randint(L, size=m).tolist()
    y  = np.random.randint(L, size=m).tolist()
    #
    C = multiclass_contingency_table(hi, hj, y)
    assert np.sum(C) == m


def test_multiclass_contingency_table_part2():
    m = 31  # number of instances / samples
    L = 2   # number of labels / classes
    hi = np.random.randint(L, size=m).tolist()
    hj = np.random.randint(L, size=m).tolist()
    y  = np.random.randint(L, size=m).tolist()
    #
    Cij = multiclass_contingency_table(hi, hj, y)
    a, b, c, d = contingency_table(hi, hj)
    assert (a == Cij[0][0] and d == Cij[1][1]) or \
           (a == Cij[1][1] and d == Cij[0][0])
    assert (b == Cij[0][1] and c == Cij[1][0]) or \
           (b == Cij[1][0] and c == Cij[0][1])


def test_multiclass_contingency_table_part3():
    m = 31  # number of instances / samples
    L = 2   # number of labels / classes
    hi = (np.random.randint(L, size=m) *2-1).tolist()
    hj = (np.random.randint(L, size=m) *2-1).tolist()
    y  = (np.random.randint(L, size=m) *2-1).tolist()
    #
    Cij = multiclass_contingency_table(hi, hj, y)
    a, b, c, d = contingency_table(hi, hj)
    assert (a == Cij[0][0] and d == Cij[1][1]) or \
           (a == Cij[1][1] and d == Cij[0][0])
    assert (b == Cij[0][1] and c == Cij[1][0]) or \
           (b == Cij[1][0] and c == Cij[0][1])



def case_number_individuals_correct(m, L, T):
    # m:  number of instances / samples
    # L:  number of labels / classes
    # T:  number of individual classifiers
    #
    if L > 0:
        yt = np.random.randint(L, size=(T, m)).tolist()
        y = np.random.randint(L, size=m).tolist()
    else:
        yt = (np.random.randint(2, size=(T, m)) *2-1).tolist()
        y = (np.random.randint(2, size=m) *2-1).tolist()
    #   #
    rho_x = number_individuals_correct(yt, y)
    ans = [[yt[i][j] ^ y[j] for j in range(m)] for i in range(T)]
    ans = np.sum(np.array(ans) == 0, axis=0)
    assert ans == np.array(rho_x)


def test_number_individuals_correct():
    case_number_individuals_correct(m=41, L=2, T=21)
    case_number_individuals_correct(m=41, L=-1, T=21)
    case_number_individuals_correct(m=41, L=4, T=21)
    case_number_individuals_correct(m=41, L=7, T=21)