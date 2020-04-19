# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



def test_pyensemble():
    try:
        import pyensemble as pyens
        assert pyens

        assert pyens.classify
        assert pyens.datasets
        assert pyens.diversity
        assert pyens.pruning

    except Exception as e:
        # raise e
        print("Exception: import pyensemble")
    else:
        pass
    finally:
        pass


def test_classify():
    try:
        import pyensemble as pyens
        assert pyens.classify.majority_voting
        assert pyens.classify.plurality_voting
        assert pyens.classify.weighted_voting

        import pyensemble.classify as ensem_cls
        assert ensem_cls.majority_voting
        assert ensem_cls.plurality_voting
        assert ensem_cls.weighted_voting

        assert ensem_cls.voting
        assert ensem_cls.majority_voting
        assert ensem_cls.plurality_voting
        assert ensem_cls.weighted_voting

    except Exception as e:
        # raise e
        print("Exception: pyensemble.classify")

def test_ensemble():
    try:
        import pyensemble as pyens
        assert pyens.classify.BaggingEnsembleAlgorithm
        assert pyens.classify.AdaBoostEnsembleAlgorithm

        import pyensemble.classify as ensem_cls
        assert ensem_cls.BaggingEnsembleAlgorithm
        assert ensem_cls.AdaBoostEnsembleAlgorithm

    except Exception as e:
        # raise e
        print("Exception: pyensemble.classify: ensemble")


def test_diversity():
    try:
        import pyensemble as pyens
        assert pyens.diversity
        assert pyens.diversity.pairwise
        assert pyens.diversity.nonpairwise
        assert pyens.diversity.overall
        assert pyens.diversity.utils_diver

        import pyensemble.diversity as ensem_diver
        assert ensem_diver.pairwise.Kappa_Statistic_binary
        assert ensem_diver.nonpairwise.Generalized_Diversity_multiclass
        assert ensem_diver.overall.calculate_overall_diversity
        assert ensem_diver.overall.calculate_item_in_diversity
        assert ensem_diver.utils_diver

    except Exception as e:
        # raise e
        print("Exception: pyensemble.diversity")


def test_pruning():
    try:
        import pyensemble as pyens
        assert pyens.pruning

        assert pyens.pruning.ranking
        assert pyens.pruning.optimizing
        assert pyens.pruning.composable
        assert pyens.pruning.ranking_based
        assert pyens.pruning.optimization_based

        from pyensemble import pruning as ensem_prune
        assert ensem_prune.Early_Stopping
        assert ensem_prune.DREP_Pruning
        assert ensem_prune.GMM_Algorithm
        assert ensem_prune.Local_Search

        import pyensemble.pruning.ranking_based as pru_rank
        assert pru_rank.ES
        assert pru_rank.KP
        assert pru_rank.KL
        assert pru_rank.KLplus
        assert pru_rank.RE
        assert pru_rank.OO

        import pyensemble.pruning.optimization_based as pru_opti
        assert pru_opti.DREP
        assert pru_opti.SEP
        assert pru_opti.PEP
        assert pru_opti.PEPplus

        import pyensemble.pruning.composable as pru_comp
        assert pru_comp.GMA
        assert pru_comp.LCS

    except Exception as e:
        # raise  
        print("Exception: pyensemble.pruning")

