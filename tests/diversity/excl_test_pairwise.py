# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
from tests.common import generate_simulated_data
from tests.common import negative_generate_simulate
from pyensemble.utils_const import check_zero


from pyensemble.diversity.pairwise import Disagreement_Measure_multiclass
from pyensemble.diversity.pairwise import Disagreement_Measure_binary

from pyensemble.diversity.pairwise import Q_Statistic_binary
from pyensemble.diversity.pairwise import Correlation_Coefficient_binary

from pyensemble.diversity.pairwise import Kappa_Statistic_multiclass
from pyensemble.diversity.pairwise import Kappa_Statistic_binary

from pyensemble.diversity.pairwise import Double_Fault_Measure_multiclass
from pyensemble.diversity.pairwise import Double_Fault_Measure_binary



class TestPairwise(unittest.TestCase):


    def test_Disagreement_Measure(self):
        m = 100
        _, yt1, _, yt2 = negative_generate_simulate(m, 2)
        ha1, hb1 = yt1
        ha2, hb2 = yt2
        d1 = Disagreement_Measure_binary(ha1, hb1, m)
        d2 = Disagreement_Measure_binary(ha2, hb2, m)
        self.assertEqual(d1, d2)
        d3 = Disagreement_Measure_multiclass(ha1, hb1, m)
        d4 = Disagreement_Measure_multiclass(ha2, hb2, m)
        self.assertEqual(d3, d4)
        self.assertEqual(d1, d3)
        self.assertEqual(d2, d4)
        m = 100
        _, yt3 = generate_simulated_data(m, 7, 2)
        d3 = Disagreement_Measure_multiclass(yt3[0], yt3[1], m)
        self.assertEqual(0 <= d3 <= 1, True)

    def test_Q_Statistic(self):
        m = 100
        _, yt1, _, yt2 = negative_generate_simulate(m, 2)
        ha1, hb1 = yt1
        ha2, hb2 = yt2
        d1 = Q_Statistic_binary(ha1, hb1)
        d2 = Q_Statistic_binary(ha2, hb2)
        self.assertEqual(d1, d2)

    def test_Correlation_Coefficient(self):
        m = 100
        _, yt1, _, yt2 = negative_generate_simulate(m, 2)
        ha1, hb1 = yt1
        ha2, hb2 = yt2
        d1 = Correlation_Coefficient_binary(ha1, hb1)
        d2 = Correlation_Coefficient_binary(ha2, hb2)
        self.assertEqual(d1, d2)

    def test_Kappa_Statistic(self):
        m = 100
        y1, yt1, y2, yt2 = negative_generate_simulate(m, 2)
        ha1, hb1 = yt1
        ha2, hb2 = yt2
        d1 = Kappa_Statistic_binary(ha1, hb1, m)
        d2 = Kappa_Statistic_binary(ha2, hb2, m)
        self.assertEqual(d1, d2)
        d3 = Kappa_Statistic_multiclass(ha1, hb1, y1, m)
        d4 = Kappa_Statistic_multiclass(ha2, hb2, y2, m)
        self.assertEqual(all(np.array(d3) == np.array(d4)), True)
        self.assertEqual(d1, d3[0])
        self.assertEqual(d2, d4[0])
        y3, yt3 = generate_simulated_data(m, 7, 2)
        d3, t1, t2 = Kappa_Statistic_multiclass(yt3[0], yt3[1], y3, m)
        self.assertEqual((t1 - t2) / check_zero(1. - t2), d3)

    def test_Double_Fault_Measure(self):
        m = 100
        y1, yt1, y2, yt2 = negative_generate_simulate(m, 2)
        ha1, hb1 = yt1
        ha2, hb2 = yt2
        d1 = Double_Fault_Measure_binary(ha1, hb1, y1, m)
        d2 = Double_Fault_Measure_binary(ha2, hb2, y2, m)
        self.assertEqual(d1, d2)
        d3 = Double_Fault_Measure_multiclass(ha1, hb1, y1, m)
        d4 = Double_Fault_Measure_multiclass(ha2, hb2, y2, m)
        self.assertEqual(d3, d4)
        self.assertEqual(d1, d3)
        self.assertEqual(d2, d4)
        y3, yt3 = generate_simulated_data(m, 7, 2)
        d3 = Double_Fault_Measure_multiclass(yt3[0], yt3[1], y3, m)
        self.assertEqual(0 <= d3 <= 1.0, True)



if __name__ == '__main__':
    unittest.main()
