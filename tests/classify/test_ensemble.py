# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import unittest
import numpy as np
from sklearn import tree

from pyensemble.utils_const import individual
from pyensemble.utils_const import FIXED_SEED


name_cls = tree.DecisionTreeClassifier()
nb_cls = 21



class TestIndividual(unittest.TestCase):
    def test_individual(self):
        prng = np.random.RandomState(FIXED_SEED)
        wX = prng.rand(100, 4)
        wy = prng.randint(4, size=100)
        # assert individual(name_cls, wX, wy)
        self.assertTrue(individual(name_cls, wX, wy))


class TestBagging(unittest.TestCase):

    def test_toy(self):
        prng = np.random.RandomState(FIXED_SEED + 12)
        X_trn = prng.rand(100, 4).tolist()
        y_trn = prng.randint(7, size=100).tolist()

        from pyensemble.classify.bagging import (
            BaggingSelectTraining,
            BaggingEnsembleAlgorithm)

        wX, wy = BaggingSelectTraining(X_trn, y_trn)
        self.assertNotEqual(wX, X_trn)
        self.assertNotEqual(wy, y_trn)

        coef, clfs = BaggingEnsembleAlgorithm(
            X_trn, y_trn, name_cls, nb_cls)
        self.assertNotEqual(id(clfs[0]), id(clfs[1]))


class TestAdaBoost(unittest.TestCase):

    def test_toy(self):
        prng = np.random.RandomState(FIXED_SEED + 34)
        X_trn = prng.rand(100, 4).tolist()
        y_trn = prng.randint(2, size=100).tolist()

        from pyensemble.classify.adaboost import (
            AdaBoostSelectTraining,
            AdaBoostEnsembleAlgorithm)

        weights = prng.rand(100)
        weights /= np.sum(weights)
        weights = weights.tolist()

        wX, wy = AdaBoostSelectTraining(X_trn, y_trn, weights)
        self.assertNotEqual(wX, X_trn)
        self.assertNotEqual(wy, y_trn)
        ans2, ans1 = np.unique(wy), np.unique(y_trn)
        self.assertTrue(np.all(np.equal(ans2, ans1)))

        coef, clfs = AdaBoostEnsembleAlgorithm(
            X_trn, y_trn, name_cls, nb_cls)
        self.assertNotEqual(id(clfs[0]), id(clfs[1]))
        self.assertNotEqual(coef[2], coef[3])


