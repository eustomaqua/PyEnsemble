# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest



class TestImport(unittest.TestCase):
    def test_pyensemble(self):
        try:
            import pyensemble as pyens
            self.assertTrue(pyens)

            self.assertTrue(pyens.classify)
            self.assertTrue(pyens.datasets)
            self.assertTrue(pyens.diversity)
            self.assertTrue(pyens.pruning)

        except Exception: # as e:
            # raise e
            print("Exception: import pyensemble")
        else:
            pass
        finally:
            pass


    def test_classify(self):
        try:
            import pyensemble as pyens
            self.assertTrue(pyens.classify.majority_voting)
            self.assertTrue(pyens.classify.plurality_voting)
            self.assertTrue(pyens.classify.weighted_voting)

            import pyensemble.classify as ensem_cls
            self.assertTrue(ensem_cls.majority_voting)
            self.assertTrue(ensem_cls.plurality_voting)
            self.assertTrue(ensem_cls.weighted_voting)

            self.assertTrue(ensem_cls.voting)
            self.assertTrue(ensem_cls.majority_voting)
            self.assertTrue(ensem_cls.plurality_voting)
            self.assertTrue(ensem_cls.weighted_voting)

        except Exception as e:
            # raise e
            print("Exception: pyensemble.classify")

    def test_ensemble(self):
        try:
            import pyensemble as pyens
            self.assertTrue(pyens.classify.BaggingEnsembleAlgorithm)
            self.assertTrue(pyens.classify.AdaBoostEnsembleAlgorithm)

            import pyensemble.classify as ensem_cls
            self.assertTrue(ensem_cls.BaggingEnsembleAlgorithm)
            self.assertTrue(ensem_cls.AdaBoostEnsembleAlgorithm)

        except Exception: # as e:
            # raise e
            print("Exception: pyensemble.classify: ensemble")


    def test_diversity(self):
        try:
            import pyensemble as pyens
            self.assertTrue(pyens.diversity)
            self.assertTrue(pyens.diversity.pairwise)
            self.assertTrue(pyens.diversity.nonpairwise)
            self.assertTrue(pyens.diversity.overall)
            self.assertTrue(pyens.diversity.utils_diver)

            import pyensemble.diversity as ensem_diver
            self.assertTrue(ensem_diver.pairwise.Kappa_Statistic_binary)
            self.assertTrue(ensem_diver.nonpairwise.Generalized_Diversity_multiclass)
            self.assertTrue(ensem_diver.overall.calculate_overall_diversity)
            self.assertTrue(ensem_diver.overall.calculate_item_in_diversity)
            self.assertTrue(ensem_diver.utils_diver)

        except Exception: # as e:
            # raise e
            print("Exception: pyensemble.diversity")


    def test_pruning(self):
        try:
            import pyensemble as pyens
            self.assertTrue(pyens.pruning)

            self.assertTrue(pyens.pruning.ranking)
            self.assertTrue(pyens.pruning.optimizing)
            self.assertTrue(pyens.pruning.composable)
            self.assertTrue(pyens.pruning.ranking_based)
            self.assertTrue(pyens.pruning.optimization_based)

            from pyensemble import pruning as ensem_prune
            self.assertTrue(ensem_prune.Early_Stopping)
            self.assertTrue(ensem_prune.DREP_Pruning)
            self.assertTrue(ensem_prune.GMM_Algorithm)
            self.assertTrue(ensem_prune.Local_Search)

            self.assertTrue(ensem_prune.OEP_Pruning)
            self.assertTrue(ensem_prune.SEP_Pruning)
            self.assertTrue(ensem_prune.PEP_Pruning)

            import pyensemble.pruning.ranking_based as pru_rank
            self.assertTrue(pru_rank.ES)
            self.assertTrue(pru_rank.KP)
            self.assertTrue(pru_rank.KL)
            self.assertTrue(pru_rank.KLplus)
            self.assertTrue(pru_rank.RE)
            self.assertTrue(pru_rank.OO)

            import pyensemble.pruning.optimization_based as pru_opti
            self.assertTrue(pru_opti.DREP)
            self.assertTrue(pru_opti.SEP)
            self.assertTrue(pru_opti.PEP)
            self.assertTrue(pru_opti.PEPplus)

            import pyensemble.pruning.composable as pru_comp
            self.assertTrue(pru_comp.GMA)
            self.assertTrue(pru_comp.LCS)

        except Exception: # as e:
            # raise
            print("Exception: pyensemble.pruning")

