# coding: utf8

import unittest
import numpy as np

from pyensemble.diversity.utils_diver import contingency_table
from pyensemble.diversity.utils_diver import multiclass_contingency_table
from pyensemble.diversity.utils_diver import number_individuals_correctly



class TestCase(unittest.TestCase):
    """
    :param m:  number of instances / samples
    :param L:  number of labels / classes
    :param T:  number of individual classifiers
    """


    def case_continency_table(self, m, L):
        if L > 0:
            hi = np.random.randint(2, size=m).tolist()
            hj = np.random.randint(2, size=m).tolist()
        else:
            hi = np.random.randint(2, size=m) * 2 - 1
            hj = np.random.randint(2, size=m) * 2 - 1
            hi, hj = hi.tolist(), hj.tolist()
        #   #
        a, b, c, d = contingency_table(hi, hj)
        self.assertEqual(a + b + c + d, m, "Wrong shape! a + b + c + d != m")

    def test_contingency_table(self):
        self.case_continency_table(m=11, L=2)
        self.case_continency_table(m=21, L=-1)


    def test_multiclass_contingency_table_part1(self):
        m = 31  # number of instances / samples
        L = 4  # number of labels / classes
        hi = np.random.randint(L, size=m).tolist()
        hj = np.random.randint(L, size=m).tolist()
        y = np.random.randint(L, size=m).tolist()
        Cij = multiclass_contingency_table(hi, hj, y)
        self.assertEqual(np.sum(Cij), m, "Wrong shape! sum(Cij) != m")


    def test_multiclass_contingency_table_part2(self):
        m = 31  # number of instances / samples
        L = 2  # number of labels / classes
        hi = np.random.randint(L, size=m).tolist()
        hj = np.random.randint(L, size=m).tolist()
        y = np.random.randint(L, size=m).tolist()
        Cij = multiclass_contingency_table(hi, hj, y)
        a, b, c, d = contingency_table(hi, hj)
        #
        judge_ad = (a == Cij[0][0] and d == Cij[1][1]) or \
                   (a == Cij[1][1] and d == Cij[0][0])
        judge_bc = (b == Cij[0][1] and c == Cij[1][0]) or \
                   (b == Cij[1][0] and c == Cij[0][1])
        self.assertEqual(judge_ad, True, "Wrong values in shape.")
        self.assertEqual(judge_bc, True, "Wrong values in shape.")


    def test_multiclass_contingency_table_part3(self):
        m = 31  # number of instances / samples
        L = 2   # number of labels / classes
        hi = (np.random.randint(L, size=m) *2-1).tolist()
        hj = (np.random.randint(L, size=m) *2-1).tolist()
        y  = (np.random.randint(L, size=m) *2-1).tolist()
        Cij = multiclass_contingency_table(hi, hj, y)
        a, b, c, d = contingency_table(hi, hj)
        #
        judge_ad = (a == Cij[0][0] and d == Cij[1][1]) or \
                   (a == Cij[1][1] and d == Cij[0][0])
        judge_bc = (b == Cij[0][1] and c == Cij[1][0]) or \
                   (b == Cij[1][0] and c == Cij[0][1])
        self.assertEqual(judge_ad, True, "Wrong values in shape.")
        self.assertEqual(judge_bc, True, "Wrong values in shape.")


    def case_number_individuals_correct(self, m, L, T):
        if L > 0:
            yt = np.random.randint(L, size=(T, m)).tolist()
            y = np.random.randint(L, size=m).tolist()
        else:
            yt = (np.random.randint(2, size=(T, m)) * 2 - 1).tolist()
            y = (np.random.randint(2, size=m) * 2 - 1).tolist()
        #   #
        rho_x = number_individuals_correctly(yt, y)
        ans = [[yt[i][j] ^ y[j] for j in range(m)] for i in range(T)]
        ans = np.sum(np.array(ans) == 0, axis=0)
        self.assertEqual(all(ans == np.array(rho_x)), True, "Wrong values in $rho_x$")

    def test_number_individuals_correct(self):
        self.case_number_individuals_correct(m=41, L=2, T=21)
        self.case_number_individuals_correct(m=41, L=-1, T=21)
        self.case_number_individuals_correct(m=41, L=4, T=21)
        self.case_number_individuals_correct(m=41, L=7, T=21)



if __name__ == '__main__':
    unittest.main()
