# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
from tests.common import negative_generate_simulate

# from pyensemble.pruning import RANKING_BASED, OPTIMIZATION_BASED
# from pyensemble.pruning import COMPOSABLE_CORE_SETS
from pyensemble.pruning import AVAILABLE_NAME_PRUNE
from pyensemble.pruning.overall import existing_contrastive_pruning_method



class TestOverall(unittest.TestCase):

    def test_overall_pruning(self):
        m, L, nb_cls, nb_pru = 20, 2, 11, 5
        _, _, y, yt = negative_generate_simulate(m, nb_cls)
        for name_pru in AVAILABLE_NAME_PRUNE:
            yo, P, flag = existing_contrastive_pruning_method(name_pru, yt, y, nb_cls, nb_pru)
            if name_pru in ['OO', 'DREP', 'SEP', 'OEP', 'PEP', 'PEP+', 'LCS']:
                self.assertEqual(len(P) <= nb_cls, True)
            else:
                self.assertEqual(len(P), nb_pru)
            if name_pru == 'OO':
                self.assertEqual(0. <= flag <= 180., True)
            self.assertEqual(all(np.unique(yo) == np.array([-1, 1])), True)
        del L



if __name__ == '__main__':
    unittest.main()
