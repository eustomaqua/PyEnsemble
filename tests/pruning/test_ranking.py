# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from tests.common import generate_simulated_data
from tests.common import negative_generate_simulate

from pyensemble.pruning.ranking_based import Early_Stopping



class TestRanking(unittest.TestCase):

    def test_ES_pruning(self):
        m, L, T, H = 100, 7, 31, 17
        _, yt = generate_simulated_data(m, L, T)
        yo, P = Early_Stopping(yt, T, H)
        self.assertEqual(sum(P), H)



if __name__ == '__main__':
    unittest.main()
