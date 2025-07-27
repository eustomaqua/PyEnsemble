# coding: utf-8


import numpy as np
# import pandas as pd

from pyfair.facil.utils_saver import elegant_print
# from pyfair.facil.utils_const import DTY_BOL
from pyfair.datasets import process_above, process_below, DATASETS


__all__ = [
    'adverse_perturb', 'adversarial',  # 'disturbed_data',
    'transform_X_and_y', 'transform_unpriv_tag',
    'transform_perturbed',
]


# ===========================
# Discriminative risk


# disturb, perturb, turbulence
# -------------------------------------


# '''
# def find_group(dataset, processed_data):
#   belongs_priv = dataset.find_where_belongs(processed_data)
#   if len(belongs_priv) > 1:
#     belongs_priv_with_joint = np.logical_and(
#         belongs_priv[0], belongs_priv[1]).tolist()
#   else:
#     belongs_priv_with_joint = []
#
#   return belongs_priv, belongs_priv_with_joint
# '''


def adverse_perturb(dataset, processed_data, ratio=.64):
    sens_attrs = dataset.sensitive_attrs
    priv_value = dataset.privileged_vals
    unpriv_dict = [
        processed_data[sa].unique().tolist() for sa in sens_attrs]
    for sa_list, pv in zip(unpriv_dict, priv_value):
        sa_list.remove(pv)

    disturbed_data = processed_data.copy()
    # num = len(disturbed_data)
    dim = len(sens_attrs)  # len(belongs_priv)
    if dim > 1:
        new_attr_name = '-'.join(sens_attrs)
        # disturbed_data = disturbed_data.drop(columns=[new_attr_name])

    for i, ti in enumerate(processed_data.index):
        prng = np.random.rand(dim)
        prng = prng <= ratio

        for j, sa, pv, un in zip(range(
                dim), sens_attrs, priv_value, unpriv_dict):
            # '''
            # if not prng[j]:
            #     continue
            # if disturbed_data.iloc[i][sa] != pv:
            #     disturbed_data.loc[ti, sa] = pv
            # else:
            #     disturbed_data.loc[ti, sa] = np.random.choice(un)
            # '''
            if prng[j] and disturbed_data.iloc[i][sa] != pv:
                disturbed_data.loc[ti, sa] = pv
            elif prng[j]:  # disturbed_data.iloc[i][sa]==pv
                disturbed_data.loc[ti, sa] = np.random.choice(un)

        if dim > 1:
            disturbed_data.loc[ti, new_attr_name] = '-'.join([
                disturbed_data.iloc[i][sa] for sa in sens_attrs])

    return disturbed_data


# adversarial (not adversarialize), v. adverse
def adversarial(dataset, data_frame, ratio=.4, logger=None):
    processed_data = process_above(dataset, data_frame, logger)

    # above: refer to `preprocess`
    # ------------------------------
    disturbed_data = adverse_perturb(dataset, processed_data, ratio)
    # ------------------------------
    # below: refer to `preprocess`

    processed_numerical, processed_binsensitive, \
        processed_categorical_binsensitive = process_below(
            dataset, disturbed_data)

    return {
        "original": disturbed_data,
        "numerical": processed_numerical,
        "numerical-binsensitive": processed_binsensitive,
        "categorical-binsensitive": processed_categorical_binsensitive
    }


def disturb_data(dataset_name, ratio=.6, logger=None):
    for ds in DATASETS:
        if ds.dataset_name != dataset_name:
            continue

        elegant_print(
            "--- Disturbing dataset: %s ---" % ds.dataset_name, logger)

        data_frame = ds.load_raw_dataset()
        t = adversarial(ds, data_frame, ratio, logger)

        return t
    raise ValueError("No dataset named `{}`.".format(dataset_name))


def transform_X_and_y(dataset, processed_binsensitive):
    # processed_numerical: pd.DataFrame
    y = processed_binsensitive[dataset.label_name]
    X = processed_binsensitive.drop(columns=dataset.label_name)
    return X, y


# def transform_unpriv_tag(dataset, processed_data):
#     # belongs_priv, belongs_priv_with_joint = find_group(
#     #     dataset, processed_data)
#
#     belongs_priv = dataset.find_where_belongs(processed_data)
#     if len(belongs_priv) > 1:
#         # belongs_priv_with_joint = np.logical_and(
#         # "" "
#         # belongs_priv_with_joint = np.logical_or(
#         #     belongs_priv[0], belongs_priv[1]
#         # ).astype('bool').tolist()  # DTY_INT
#         # # First submission (have modified)
#         # "" "
#
#         belongs_priv_with_joint = np.logical_and(
#             belongs_priv[0], belongs_priv[1]
#         ).astype(DTY_BOL).tolist()  # DTY_INT
#
#     else:
#         belongs_priv_with_joint = []
#
#     # belongs_priv = [i.astype(DTY_INT) for i in belongs_priv]
#     # belongs_priv = [i.astype(DTY_BOL) for i in belongs_priv]
#     return belongs_priv, belongs_priv_with_joint


def transform_unpriv_tag(dataset, processed_data,
                         joint='and'):
    assert joint in (
        'and', 'or', 'both'), "Improper joint-parameter"

    belongs_priv = dataset.find_where_belongs(processed_data)
    if len(belongs_priv) <= 1:  # not >1
        belongs_priv_with_joint = []
        return belongs_priv, belongs_priv_with_joint

    # else:  # if len(belongs_priv) > 1:  .astype(DTY_BOL)
    if joint == 'and':
        belongs_priv_with_joint = np.logical_and(
            belongs_priv[0], belongs_priv[1]).tolist()
    elif joint == 'or':
        belongs_priv_with_joint = np.logical_or(
            belongs_priv[0], belongs_priv[1]).tolist()
    elif joint == 'both':
        belongs_priv_with_joint = [
            np.logical_and(belongs_priv[0], belongs_priv[1]),
            np.logical_or(belongs_priv[0], belongs_priv[1]),
        ]
    return belongs_priv, belongs_priv_with_joint


# Deal with `data_frame`

def transform_perturbed(X_org, X_qtb, y, index, belongs_priv,
                        belongs_priv_with_joint):
    X_idx_org = X_org.iloc[index]
    X_idx_qtb = X_qtb.iloc[index]
    y_idx = y.iloc[index]

    not_unpriv = [t[index] for t in belongs_priv]
    if not belongs_priv_with_joint:
        joint_ = belongs_priv_with_joint
    else:
        joint_ = np.array(
            belongs_priv_with_joint)[index].tolist()

    return X_idx_org, X_idx_qtb, y_idx, not_unpriv, joint_


# def transform_cvs_item(split_idx_item):
#     if len(split_idx_item) == 2:
#         idx_trn, idx_tst = split_idx_item
#     else:
#         idx_trn, idx_val, idx_tst = split_idx_item
#         idx_trn += idx_val
#         # i_trn = i_trn + i_val
#     return idx_trn, idx_tst


# def transform_cv_split_ith(X, y, idx_trn, idx_tst):
#     # X, y: pd.DataFrame
#
#     X_trn = X.iloc[idx_trn]
#     y_trn = y.iloc[idx_trn]
#     X_tst = X.iloc[idx_tst]
#     y_tst = y.iloc[idx_tst]
#
#     return X_trn, y_trn, X_tst, y_tst


# -------------------------------------
# via manifold and its extension
# -------------------------------------


# -------------------------------------
