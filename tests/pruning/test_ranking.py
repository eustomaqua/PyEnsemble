# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from tests.common import generate_simulated_data

from pyensemble.pruning.ranking_based import Early_Stopping as ES
from pyensemble.pruning.ranking_based import KL_divergence_Pruning as KL
from pyensemble.pruning.ranking_based import KL_divergence_Pruning_modify as KLplus
from pyensemble.pruning.ranking_based import Kappa_Pruning as KP
from pyensemble.pruning.ranking_based import Orientation_Ordering_Pruning as OO
from pyensemble.pruning.ranking_based import Reduce_Error_Pruning as RE



class TestRanking(unittest.TestCase):

    def test_ES_pruning(self):
        m, L, T, H = 30, 4, 7, 3
        _, yt = generate_simulated_data(m, L, T)
        yo, P = ES.Early_Stopping(yt, T, H)
        self.assertEqual(sum(P), H)


    def test_KL_divergence_pruning(self):
        m, L, nb_cls, nb_pru = 30, 4, 7, 3
        y, yt = generate_simulated_data(m, L, nb_cls)
        yo, P = KL.KL_divergence_Pruning(yt, nb_cls, nb_pru)
        self.assertEqual(sum(P), nb_pru)


    def test_KL_divergence_modify(self):
        m, L, nb_cls, nb_pru = 30, 4, 7, 3
        y, yt = generate_simulated_data(m, L, nb_cls)
        yo, P = KLplus.KL_divergence_Pruning_modify(yt, nb_cls, nb_pru)
        self.assertEqual(sum(P), nb_pru)


    def test_Kappa_pruning(self):
        m, L, nb_cls, nb_pru = 30, 4, 7, 3
        y, yt = generate_simulated_data(m, L, nb_cls)
        yo, P = KP.Kappa_Pruning(yt, y, nb_cls, nb_pru)
        self.assertEqual(sum(P), nb_pru)


    def test_OO_pruning(self):
        m, L, nb_cls, nb_pru = 30, 4, 7, 3
        y, yt = generate_simulated_data(m, L, nb_cls)
        yo, P, flag = OO.Orientation_Ordering_Pruning(yt, y)
        self.assertEqual(sum(P) <= nb_cls, True)
        self.assertEqual(0. <= flag <= 180., True)


    def test_RE_pruning(self):
        m, L, nb_cls, nb_pru = 30, 4, 7, 3
        y, yt = generate_simulated_data(m, L, nb_cls)
        yo, P = RE.Reduce_Error_Pruning(yt, y, nb_cls, nb_pru)
        self.assertEqual(sum(P), nb_pru)



if __name__ == '__main__':
    unittest.main()
