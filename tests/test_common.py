from __future__ import division
import unittest
import numpy as np
from tests.common import generate_simulated_data
from tests.common import negative_generate_simulate


class TestCommon(unittest.TestCase):

    def test_generate_simulated_data_negative(self):
        y, yt = generate_simulated_data(200, 1, 71)
        vY = np.unique(np.concatenate([[y], yt])).tolist()
        self.assertEqual(len(vY), 2)
        self.assertEqual(((-1 in vY) and (1 in vY)), True)
        y, yt = generate_simulated_data(200, -1, 71)
        vY = np.unique(np.concatenate([[y], yt])).tolist()
        self.assertEqual(len(vY), 2)
        self.assertEqual(((-1 in vY) and (1 in vY)), True)

    def case_generate_simulated_data_positive(self, m, L, T):
        y, yt = generate_simulated_data(m, L, T)
        vY = np.unique(np.concatenate([[y], yt])).tolist()
        self.assertEqual(len(vY), L)
        self.assertEqual(((min(vY) == 0) and (max(vY)==L-1)), True)

    def test_generate_simulated_data_positive(self):
        self.case_generate_simulated_data_positive(200, 2, 71)
        self.case_generate_simulated_data_positive(200, 7, 91)


    def test_negative_generate_simulated(self):
        y1, yt1, y2, yt2 = negative_generate_simulate(300, 81)
        h1 = np.unique(np.concatenate([[y1], yt1])).tolist()
        h2 = np.unique(np.concatenate([[y2], yt2])).tolist()
        self.assertEqual((len(h1) == 2) and (len(h2) == 2), True)
        self.assertEqual((0 in h1) and (1 in h1), True)
        self.assertEqual((-1 in h2) and (1 in h2), True)



if __name__ == '__main__':
    unittest.main()
