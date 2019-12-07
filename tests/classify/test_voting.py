# coding: utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
# from sklearn import tree            # DecisionTreeClassifier()
# from sklearn import naive_bayes     # GaussianNB, MultinomialNB, BernoulliNB
# from sklearn import svm             # SVC, NuSVC, LinearSVC
# from sklearn import neighbors       # KNeighborsClassifier(n_neighbors, weights='uniform' or 'distance')
# from sklearn import linear_model    # SGDClassifier(loss='hinge', penalty='l1' or 'l2')


from pyensemble.classify.voting import plurality_voting
from pyensemble.classify.voting import majority_voting
from pyensemble.classify.voting import weighted_voting
from pyensemble.classify.voting import individual, NAME_INDIVIDUALS
from pyensemble.classify import AVAILABLE_ABBR_CLS


class TestVoting(unittest.TestCase):

    def test_toy(self):
        # weig = [0.2, 0.3, 0.5]
        weig = [0.3, 0.6, 0.1]
        y  =  [0, 1, 0, 1, 0, 0, 1]
        yt = [[0, 1, 0, 0, 1, 0, 1],
              [1, 1, 1, 1, 0, 1, 0],
              [0, 0, 1, 0, 1, 1, 0]]
        #
        #     [0, 1, 1, 0, 1, 1, 0] plurality
        #     [0, 1, 1, 0, 1, 1, 0] majority
        #     [0, 0, 1, 0, 1, 1, 0] weighted
        fens = np.array(plurality_voting(y, yt))
        self.assertEqual(all(fens == np.array([0, 1, 1, 0, 1, 1, 0])), True)
        fens = np.array(majority_voting(y, yt))
        self.assertEqual(all(fens == np.array([0, 1, 1, 0, 1, 1, 0])), True)
        fens = np.array(weighted_voting(y, yt, weig))
        self.assertEqual(all(fens == np.array([1, 1, 1, 1, 0, 1, 0])), True)
        #
        # weig = [0.3, 0.1, 0.25, 0.35]
        weig = [0.3, 0.45, 0.15, 0.2]
        y  =  [0, 1, 2, 3, 3, 2, 1]
        yt = [[0, 2, 3, 2, 1, 2, 1],
              [1, 1, 1, 2, 1, 2, 3],
              [0, 2, 0, 3, 3, 1, 2],
              [1, 0, 3, 1, 3, 2, 0]]
        est = [0, 2, 3, 2, 1, 2, 0]
        fens = np.array(plurality_voting(y, yt))
        self.assertEqual(all(fens == np.array(est)), True)
        est = [0, 2, 3, 2, 1, 2, -1]
        fens = np.array(majority_voting(y, yt))
        self.assertEqual(all(fens == np.array(est)), True)
        est = [1, 1, 3, 2, 1, 2, 3]
        fens = np.array(weighted_voting(y, yt, weig))
        self.assertEqual(all(fens == np.array(est)), True)


    def test_individual(self):
        wX = np.random.rand(100, 4).tolist()  #       # list
        wy = np.random.randint(2, size=100).tolist()  # list
        for abbr_cls in AVAILABLE_ABBR_CLS:
            name_cls = NAME_INDIVIDUALS[abbr_cls]
            ht = individual(name_cls, wX, wy)  #      # np.ndarray
            judge1 = all(np.unique(ht) == np.unique(wy))
            judge2 = len(np.unique(ht)) == len(np.unique(wy))
            judge3 = len(np.unique(ht)) < len(np.unique(wy))
            self.assertEqual(judge1 or judge3, True)
            self.assertEqual(judge2 or judge3, True)




if __name__ == '__main__':
    unittest.main()
