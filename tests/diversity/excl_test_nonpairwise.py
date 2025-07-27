# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from tests.common import generate_simulated_data
from tests.common import negative_generate_simulate

from pyensemble.diversity.nonpairwise import Kohavi_Wolpert_Variance_multiclass
from pyensemble.diversity.nonpairwise import Interrater_agreement_multiclass

from pyensemble.diversity.nonpairwise import Entropy_cc_multiclass
from pyensemble.diversity.nonpairwise import Entropy_sk_multiclass

from pyensemble.diversity.nonpairwise import Difficulty_multiclass
from pyensemble.diversity.nonpairwise import Generalized_Diversity_multiclass
from pyensemble.diversity.nonpairwise import Coincident_Failure_multiclass



class TestNonPairwise(unittest.TestCase):


    def test_KW_Variance(self):
        m, T = 100, 21
        y1, yt1, y2, yt2 = negative_generate_simulate(m, T)
        d1 = Kohavi_Wolpert_Variance_multiclass(yt1, y1, m, T)
        d2 = Kohavi_Wolpert_Variance_multiclass(yt2, y2, m, T)
        self.assertEqual(d1, d2)
        y3, yt3 = generate_simulated_data(m, 7, T)
        d3 = Kohavi_Wolpert_Variance_multiclass(yt3, y3, m, T)
        # self.assertEqual(d3 >= 3./8, True)  # only for binary
        # |-> Nope, all of them belong to [0, 1/2]
        self.assertEqual(d3 <= 1./2, True)
        self.assertEqual(d3 >= 0., True)
        self.assertEqual(0. < d1 <= 1./2, True)
        self.assertEqual(0. < d2 <= 1./2, True)


    def test_Interrater_agreement(self):
        m, T = 100, 21
        y1, yt1, y2, yt2 = negative_generate_simulate(m, T)
        d1 = Interrater_agreement_multiclass(yt1, y1, m, T)
        d2 = Interrater_agreement_multiclass(yt2, y2, m, T)
        self.assertEqual(d1, d2)
        y3, yt3 = generate_simulated_data(m, 7, T)
        d3 = Interrater_agreement_multiclass(yt3, y3, m, T)
        self.assertEqual(isinstance(d3, float), True)


    def test_Entropy_CC(self):
        m, T = 100, 21
        y1, yt1, y2, yt2 = negative_generate_simulate(m, T)
        d1 = Entropy_cc_multiclass(yt1, y1, m, T)
        d2 = Entropy_cc_multiclass(yt2, y2, m, T)
        self.assertEqual(d1, d2)
        y3, yt3 = generate_simulated_data(m, 7, T)
        d3 = Entropy_cc_multiclass(yt3, y3, m, T)
        self.assertEqual(d3 >= 0, True)

    def test_Entropy_SK(self):
        m, T = 100, 21
        y1, yt1, y2, yt2 = negative_generate_simulate(m, T)
        d1 = Entropy_sk_multiclass(yt1, y1, m, T)
        d2 = Entropy_sk_multiclass(yt2, y2, m, T)
        self.assertEqual(d1, d2)
        y3, yt3 = generate_simulated_data(m, 7, T)
        d3 = Entropy_sk_multiclass(yt3, y3, m, T)
        self.assertEqual(d3 >= 0, True)


    def test_Difficulty(self):
        m, T = 100, 21
        y1, yt1, y2, yt2 = negative_generate_simulate(m, T)
        d1 = Difficulty_multiclass(yt1, y1, T)
        d2 = Difficulty_multiclass(yt2, y2, T)
        self.assertEqual(d1, d2)
        y3, yt3 = generate_simulated_data(m, 7, T)
        d3 = Difficulty_multiclass(yt3, y3, T)
        self.assertEqual(d3 >= 0, True)


    def test_Generalized_Diversity(self):
        m, T = 100, 21
        y1, yt1, y2, yt2 = negative_generate_simulate(m, T)
        d1 = Generalized_Diversity_multiclass(yt1, y1, m, T)
        d2 = Generalized_Diversity_multiclass(yt2, y2, m, T)
        self.assertEqual(d1, d2)
        y3, yt3 = generate_simulated_data(m, 7, T)
        d3 = Generalized_Diversity_multiclass(yt3, y3, m, T)
        self.assertEqual(d3 >= 0, True)
        self.assertEqual(d3 <= 1, True)


    def test_Coincident_Failure(self):
        m, T = 100, 21
        y1, yt1, y2, yt2 = negative_generate_simulate(m, T)
        d1 = Coincident_Failure_multiclass(yt1, y1, m, T)
        d2 = Coincident_Failure_multiclass(yt2, y2, m, T)
        self.assertEqual(d1, d2)
        y3, yt3 = generate_simulated_data(m, 7, T)
        d3 = Coincident_Failure_multiclass(yt3, y3, m, T)
        self.assertEqual(d3 >= 0, True)



if __name__ == '__main__':
    unittest.main()
