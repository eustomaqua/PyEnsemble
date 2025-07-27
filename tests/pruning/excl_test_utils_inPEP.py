# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np

from tests.common import generate_simulated_data
from pyensemble.pruning.utils_inPEP import PEP_Hs_x, PEP_diff_hihj, PEP_err_hi, PEP_f_Hs
from pyensemble.pruning.utils_inPEP import PEP_flipping_uniformly
from pyensemble.pruning.utils_inPEP import PEP_bi_objective
from pyensemble.pruning.utils_inPEP import PEP_weakly_dominate, PEP_dominate



class TestUtilsInPEP(unittest.TestCase):

    def test_assume(self):
        m, nb_cls = 100, 7
        y, yt = generate_simulated_data(m, 2, nb_cls)
        # weig = [0.1, 0.25, 0.2, 0.05, 0.15, 0.1, 0.15]
        weig = np.random.rand(nb_cls)
        weig /= np.sum(weig)
        weig = weig.tolist()
        #
        Hsx = PEP_Hs_x(y, yt, weig)
        from pyensemble.classify.voting import weighted_voting
        tem = weighted_voting(y, yt, weig)
        Hsx, tem = np.array(Hsx), np.array(tem)
        self.assertEqual(all(Hsx == tem), True)
        #
        diff = PEP_diff_hihj(yt[0], yt[1])
        self.assertEqual(0.0 <= diff <= 1.0, True)
        #
        err = PEP_err_hi(y, yt[0])
        self.assertEqual(0. <= err <= 1., True)
        err = PEP_err_hi(y, yt[1])
        self.assertEqual(0. <= err <= 1., True)
        #
        ans, _ = PEP_f_Hs(y, yt, weig)
        self.assertEqual(0. <= ans <= 1., True)


    def test_domination(self):
        m, nb_cls = 100, 7
        s = np.random.rand(nb_cls)
        sp = PEP_flipping_uniformly(s)
        ans = np.not_equal(sp, sp)
        ans = np.mean(ans) <= (1. / nb_cls)
        self.assertEqual(ans, True)
        #
        y, yt = generate_simulated_data(m, 2, nb_cls)
        ans = PEP_bi_objective(y, yt, s)
        fHs, s_ab = ans
        self.assertEqual(0. <= fHs <= 1., True)
        self.assertEqual(0 <= s_ab <= nb_cls, True)
        #
        g_s1, g_s2 = (0.1, 3), (0.2, 2)
        ans = PEP_weakly_dominate(g_s1, g_s2)
        self.assertEqual(ans, False)
        g_s1, g_s2 = (0.1, 3), (0.1, 4)
        ans = PEP_weakly_dominate(g_s1, g_s2)
        self.assertEqual(ans, True)
        g_s1, g_s2 = (0.1, 3), (0.3, 3)
        ans = PEP_weakly_dominate(g_s1, g_s2)
        self.assertEqual(ans, True)
        g_s1, g_s2 = (0.4, 3), (0.4, 3)
        ans = PEP_weakly_dominate(g_s1, g_s2)
        self.assertEqual(ans, True)
        #
        g_s1, g_s2 = (0.4, 3), (0.4, 3)
        ans = PEP_dominate(g_s1, g_s2)
        self.assertEqual(ans, False)
        g_s1, g_s2 = (0.2, 1), (0.4, 3)
        ans = PEP_dominate(g_s1, g_s2)
        self.assertEqual(ans, True)
        g_s1, g_s2 = (0.4, 2), (0.4, 3)
        ans = PEP_dominate(g_s1, g_s2)
        self.assertEqual(ans, True)
        g_s1, g_s2 = (0.3, 3), (0.4, 3)
        ans = PEP_dominate(g_s1, g_s2)
        self.assertEqual(ans, True)





if __name__ == '__main__':
    unittest.main()
