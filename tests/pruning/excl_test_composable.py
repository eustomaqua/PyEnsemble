# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from tests.common import generate_simulated_data
from pyensemble.utils_const import check_equal  # CONST_ZERO

from pyensemble.pruning.composable import GMM_Algorithm as GMM
from pyensemble.pruning.composable import Local_Search_Alg as LCS
from pyensemble.diversity.pairwise import Kappa_Statistic_multiclass



class TestComposable(unittest.TestCase):

    def test_GMM_Algorithm(self):
        m, nb_cls = 30, 7
        y, yt = generate_simulated_data(m, 2, nb_cls)
        tem = GMM.GMM_Kappa_sum(yt[0], yt[1:], y)
        ans = [Kappa_Statistic_multiclass(yt[0], j, y, m)[0] for j in yt[1:]]
        ans = sum(ans)
        self.assertEqual(tem == ans, True)
        #
        nb_pru = 7
        yo, P = GMM.GMM_Algorithm(yt, y, nb_cls, nb_pru)
        self.assertEqual(sum(P) == nb_pru, True)
        self.assertEqual(len(yo) == nb_pru, True)
        #
        y, yt = generate_simulated_data(m, 4, nb_cls)
        tem = GMM.GMM_Kappa_sum(yt[0], yt[1:], y)
        ans = [Kappa_Statistic_multiclass(yt[0], j, y, m)[0] for j in yt[1:]]
        ans = sum(ans)
        # self.assertEqual(tem == ans, True)
        self.assertEqual(abs(tem - ans) <= 1e-6, True)
        yo, P = GMM.GMM_Algorithm(yt, y, nb_cls, nb_pru)
        self.assertEqual(sum(P) == nb_pru, True)

    def test_LCS_Algorithm(self):
        m, nb_cls = 30, 7
        y, yt = generate_simulated_data(m, 2, nb_cls)
        tem = LCS.LocalSearch_kappa_sum(yt, y)
        ans = 0.0
        for i in range(nb_cls - 1):
            for j in range(i + 1, nb_cls):
                ans += Kappa_Statistic_multiclass(yt[i], yt[j], y, m)[0]
        ans /= (nb_cls * (nb_cls - 1.) / 2.)
        # self.assertEqual(tem == ans, True)
        self.assertEqual(check_equal(tem, ans), True)
        nb_pru = 7
        yo, P = LCS.Local_Search(yt, y, nb_cls, nb_pru, 1e-6)
        self.assertEqual(sum(P) == nb_pru, True)
        self.assertEqual(len(yo) == nb_pru, True)
        #
        y, yt = generate_simulated_data(m, 4, nb_cls)
        tem = LCS.LocalSearch_kappa_sum(yt, y)
        ans = 0.0
        for i in range(nb_cls - 1):
            for j in range(i + 1, nb_cls):
                ans += Kappa_Statistic_multiclass(yt[i], yt[j], y, m)[0]
        ans /= (nb_cls * (nb_cls - 1.) / 2.)
        # self.assertEqual(tem == ans, True)
        self.assertEqual(abs(tem - ans) <= 1e-6, True)
        yo, P = LCS.Local_Search(yt, y, nb_cls, nb_pru, 1e-6)
        self.assertEqual(sum(P) == nb_pru, True)



if __name__ == '__main__':
    unittest.main()
