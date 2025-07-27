# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from tests.common import negative_generate_simulate
from pyensemble.diversity import PAIRWISE, NONPAIRWISE, AVAILABLE_NAME_DIVER

from pyensemble.diversity.overall import pairwise_measure_for_item
from pyensemble.diversity.overall import pairwise_measure_overall_value
from pyensemble.diversity.overall import nonpairwise_measure_for_item
from pyensemble.diversity.overall import nonpairwise_measure_overall_value
from pyensemble.diversity.overall import calculate_item_in_diversity
from pyensemble.diversity.overall import calculate_overall_diversity



class TestOverall(unittest.TestCase):


    def test_pairwise_item(self):
        m = 100
        y1, yt1, y2, yt2 = negative_generate_simulate(m, 2)
        ha1, hb1 = yt1
        ha2, hb2 = yt2
        for name_div in PAIRWISE:
            d1 = pairwise_measure_for_item(name_div, ha1, hb1, y1, m)
            d2 = pairwise_measure_for_item(name_div, ha2, hb2, y2, m)
            self.assertEqual(d1, d2)

    def test_pairwise_ensem(self):
        m, T = 100, 21
        y1, yt1, y2, yt2 = negative_generate_simulate(m, T)
        for name_div in PAIRWISE:
            d1 = pairwise_measure_overall_value(name_div, yt1, y1, m, T)
            d2 = pairwise_measure_overall_value(name_div, yt2, y2, m, T)
            self.assertEqual(d1, d2)


    def test_nonpairwise_item(self):
        m = 100
        y1, yt1, y2, yt2 = negative_generate_simulate(m, 2)
        ha1, hb1 = yt1
        ha2, hb2 = yt2
        for name_div in NONPAIRWISE:
            d1 = nonpairwise_measure_for_item(name_div, ha1, hb1, y1, m)
            d2 = nonpairwise_measure_for_item(name_div, ha2, hb2, y2, m)
            self.assertEqual(d1, d2)

    def test_nonpairwise_ensem(self):
        m, T = 100, 21
        y1, yt1, y2, yt2 = negative_generate_simulate(m, T)
        for name_div in NONPAIRWISE:
            d1 = nonpairwise_measure_overall_value(name_div, yt1, y1, m, T)
            d2 = nonpairwise_measure_overall_value(name_div, yt2, y2, m, T)
            self.assertEqual(d1, d2)


    def test_inference(self):
        m = 100
        y1, (ha1, hb1), y2, (ha2, hb2) = negative_generate_simulate(m, 2)
        for name_div in AVAILABLE_NAME_DIVER:
            d1 = calculate_item_in_diversity(name_div, ha1, hb1, y1, m)
            d2 = calculate_item_in_diversity(name_div, ha2, hb2, y2, m)
            self.assertEqual(d1, d2)
        T = 21
        y1, yt1, y2, yt2 = negative_generate_simulate(m, T)
        for name_div in AVAILABLE_NAME_DIVER:
            d1 = calculate_overall_diversity(name_div, yt1, y1, m, T)
            d2 = calculate_overall_diversity(name_div, yt2, y2, m, T)
            self.assertEqual(d1, d2)




if __name__ == '__main__':
    unittest.main()
