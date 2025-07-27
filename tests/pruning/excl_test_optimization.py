# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
from tests.common import generate_simulated_data
from tests.common import negative_generate_simulate

from pyensemble.pruning.optimization_based import DREP as DREP
from pyensemble.pruning.optimization_based import SEP_inPEP as SEP
from pyensemble.pruning.ranking_based import OEP_inPEP as OEP
from pyensemble.pruning.optimization_based import PEP_inPEP as PEP
from pyensemble.pruning.optimization_based import PEP_modify as PEPplus



class TestOptimization(unittest.TestCase):


    def test_DREP(self):
        m, nb_cls = 30, 7
        h, ht, y, yt = negative_generate_simulate(m, nb_cls)
        fens = DREP.DREP_fxH(yt)
        self.assertEqual(all(np.unique(fens) == np.array([-1, 1])), True)
        tem_h = DREP.DREP_diff(ht[0], ht[1])
        tem_y = DREP.DREP_diff(yt[0], yt[1])
        self.assertEqual(tem_h == tem_y, True)
        #
        yo, P = DREP.DREP_Pruning(yt, y, nb_cls, 0.4)
        self.assertEqual(len(yo) == sum(P), True)
        self.assertEqual(sum(P) <= nb_cls, True)
        h, ht = generate_simulated_data(m, 4, 2)
        tem_h = DREP.DREP_diff(ht[0], ht[1])
        self.assertEqual(-1. <= tem_h <= 1., True)
        #
        del h


    def test_SEP(self):
        m, nb_label, nb_cls = 30, 4, 7
        y, yt = generate_simulated_data(m, nb_label, nb_cls)
        nb_pru = 7
        yo, P = SEP.PEP_SEP(yt, y, nb_cls, nb_pru)
        self.assertEqual(sum(P) <= nb_cls, True)
        self.assertEqual(all(np.unique(yo) == np.unique(yt)), True)


    def test_OEP(self):
        m, nb_label, nb_cls = 30, 4, 7
        y, yt = generate_simulated_data(m, nb_label, nb_cls)
        yo, P = OEP.PEP_OEP(yt, y, nb_cls)
        self.assertEqual(sum(P) <= nb_cls, True)
        self.assertEqual(all(np.unique(yo) == np.unique(yt)), True)


    def test_PEP(self):
        m, nb_label, nb_cls = 30, 4, 7
        y, yt = generate_simulated_data(m, nb_label, nb_cls)
        s = np.random.randint(2, size=nb_cls).tolist()
        Q, L = PEP.PEP_VDS(y, yt, nb_cls, s)
        yo, P = PEP.PEP_PEP(yt, y, nb_cls, 0.4)
        self.assertEqual(sum(P) <= nb_cls, True)
        self.assertEqual(all(np.unique(yo) == np.unique(yt)), True)
        self.assertEqual(len(Q) == nb_cls, True)
        self.assertEqual(len(L) == nb_cls, True)


    def test_PEP_modify(self):
        m, nb_label, nb_cls = 30, 4, 7
        y, yt = generate_simulated_data(m, nb_label, nb_cls)
        yo, P = PEPplus.PEP_PEP_modify(yt, y, nb_cls, 0.4)
        self.assertEqual(sum(P) <= nb_cls, True)
        self.assertEqual(all(np.unique(yo) == np.unique(yt)), True)



if __name__ == '__main__':
    unittest.main()
