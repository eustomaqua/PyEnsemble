# coding: utf-8
# pyfair.junior/utils


from pyfair.marble.data_classify import (
    # BaggingEnsembleAlgorithm, BoostingEnsemble_multiclass,
    EnsembleAlgorithm)

from pyfair.marble.metric_perf import (
    calc_accuracy, calc_precision, calc_recall, calc_f1_score,
    calc_f_beta, calc_sensitivity, calc_specificity,
    imba_geometric_mean, imba_discriminant_power,
    imba_balanced_accuracy, imba_Matthew_s_cc, imba_Cohen_s_kappa,
    imba_Youden_s_index, imba_likelihoods)

from pyfair.marble.metric_fair import (
    prev_unpriv_grp_one, prev_unpriv_grp_two, prev_unpriv_grp_thr,
    marginalised_pd_mat, prev_unpriv_unaware, prev_unpriv_manual,
    unpriv_group_one, unpriv_group_two, unpriv_group_thr,
    marginalised_np_mat, unpriv_unaware, unpriv_manual)


__all__ = [
    'EnsembleAlgorithm',

    'marginalised_pd_mat', 'marginalised_np_mat',
    'prev_unpriv_grp_one', 'unpriv_group_one',
    'prev_unpriv_grp_two', 'unpriv_group_two',
    'prev_unpriv_grp_thr', 'unpriv_group_thr',
    'prev_unpriv_unaware', 'unpriv_unaware',
    'prev_unpriv_manual', 'unpriv_manual',

    'calc_accuracy', 'calc_precision', 'calc_recall',
    'calc_f1_score', 'calc_f_beta',
    'calc_sensitivity', 'calc_specificity',
    'imba_geometric_mean', 'imba_discriminant_power',
    'imba_balanced_accuracy',
    'imba_Matthew_s_cc', 'imba_Cohen_s_kappa',
    'imba_Youden_s_index', 'imba_likelihoods',
]
