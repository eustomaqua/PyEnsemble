# coding: utf-8


from copy import deepcopy
import time
import pandas as pd
import numpy as np

# from pyfair.facil.utils_const import DTY_BOL
from pyfair.datasets import (
    process_above, make_class_attr_num,
    make_sensitive_attrs_binary)
from pyfair.preprocessing_dr import adverse_perturb


__all__ = [
    'binarized_data_set', 'transform_X_A_and_y',
    # 'transform_unpriv_tag_prime',
    'transform_disturb_prime',
    'normalise_disturb_prime', 'normalise_disturb_whole',

    'renewed_transform_disturb', 'renewed_normalise_disturb',
    'renewed_normalise_separate',
    'renewed_prep_and_adversarial', 'renewed_transform_X_A_and_y',
    'check_marginalised_indices',
    'process_addtl', 'process_addtl_multivalue',
]


# -------------------------------------
# via manifold and its extension
# -------------------------------------


# -------------------------------------
# fmanf


def binarized_data_set(processed_binsensitive):
    binarized_binsens = deepcopy(processed_binsensitive)
    for u in binarized_binsens.columns:
        if binarized_binsens[u].dtype == bool:
            binarized_binsens[u] = binarized_binsens[u].astype('int')
    return binarized_binsens


def transform_X_A_and_y(dataset, processed_binsensitive):
    y = processed_binsensitive[dataset.label_name]

    sensitive_attrs = deepcopy(dataset.sensitive_attrs)
    if len(sensitive_attrs) > 1:
        new_attr_name = '-'.join(sensitive_attrs)
        # sensitive_attrs += [new_attr_name]
        sensitive_attrs.append(new_attr_name)
    else:
        new_attr_name = None
    # A = processed_binsensitive[sensitive_attrs]
    A = deepcopy(processed_binsensitive[sensitive_attrs])

    X = deepcopy(processed_binsensitive)
    X.drop(columns=dataset.label_name, inplace=True)
    # X = processed_binsensitive.drop(columns=dataset.label_name)
    X = X.drop(columns=sensitive_attrs)
    return X, A, y, new_attr_name


# def transform_unpriv_tag_prime(dataset, processed_original,
#                                joint=('and', 'or', 'both')):
#     assert joint in ['and', 'or', 'both'], "Improper joint-parameter"
#     belongs_priv = dataset.find_where_belongs(processed_original)
#
#     # if len(belongs_priv) > 1 and joint == 'and':
#     #     belongs_priv_with_joint = np.logical_and(
#     #         belongs_priv[0],
#     #         belongs_priv[1]).astype(DTY_BOL).tolist()
#     # elif len(belongs_priv) > 1 and joint == 'or':
#     #     belongs_priv_with_joint = np.logical_or(
#     #         belongs_priv[0],
#     #         belongs_priv[1]).astype(DTY_BOL).tolist()
#     # elif len(belongs_priv) > 1 and joint == 'both':
#     #     belongs_priv_with_joint = [
#     #         np.logical_and(belongs_priv[
#     #             0], belongs_priv[1]).astype(DTY_BOL),
#     #         np.logical_or(belongs_priv[
#     #             0], belongs_priv[1]).astype(DTY_BOL),
#     #     ]
#     # else:
#     #     belongs_priv_with_joint = []
#
#     if len(belongs_priv) == 1:  # not >1
#         belongs_priv_with_joint = []
#         return belongs_priv, belongs_priv_with_joint
#     # else:  # if len(belongs_priv) > 1:
#     if joint == 'and':
#         belongs_priv_with_joint = np.logical_and(
#             belongs_priv[0],
#             belongs_priv[1]).astype(DTY_BOL).tolist()
#     elif joint == 'or':
#         belongs_priv_with_joint = np.logical_or(
#             belongs_priv[0],
#             belongs_priv[1]).astype(DTY_BOL).tolist()
#     elif joint == 'both':
#         belongs_priv_with_joint = [
#             np.logical_and(belongs_priv[
#                 0], belongs_priv[1]).astype(DTY_BOL),
#             np.logical_or(belongs_priv[
#                 0], belongs_priv[1]).astype(DTY_BOL),
#         ]
#     return belongs_priv, belongs_priv_with_joint


def transform_disturb_prime(X, A, y, index, belongs_priv,
                            belongs_priv_with_joint):
    """ X, y: np.ndarray or pd.DataFrame
    belongs_priv           : list, element is a np.ndarray
    belongs_priv_with_joint: list, element is index
    """
    if isinstance(X, pd.DataFrame):
        X_idx = X.iloc[index]
        A_idx = A if (A is None) else A.iloc[index]
        y_idx = y.iloc[index]
    elif isinstance(X, np.ndarray):
        X_idx = X[index]
        A_idx = A if (A is None) else A[index]
        y_idx = y[index]

    not_unpriv = [t[index] for t in belongs_priv]
    if not belongs_priv_with_joint:
        joint_ = belongs_priv_with_joint
    elif len(belongs_priv_with_joint) == 2:
        joint_ = [t[index] for t in belongs_priv_with_joint]
    else:  # belongs_priv_with_joint is a list
        joint_ = np.array(belongs_priv_with_joint)[index].tolist()
    return X_idx, A_idx, y_idx, not_unpriv, joint_


def normalise_disturb_prime(scaler, X_trn, A_trn,
                            X_val, A_val, X_tst, A_tst):
    # all are np.ndarray (not pd.DataFrame)
    _, nb_feat = X_trn.shape
    XA_trn = np.concatenate([X_trn, A_trn], axis=1)
    XA_tst = np.concatenate([X_tst, A_tst], axis=1)
    XA_val = []

    scaler = scaler.fit(XA_trn)
    XA_trn = scaler.transform(XA_trn)
    XA_tst = scaler.transform(XA_tst)
    X_trn, A_trn = XA_trn[:, :nb_feat], XA_trn[:, nb_feat:]
    X_tst, A_tst = XA_tst[:, :nb_feat], XA_tst[:, nb_feat:]

    if len(X_val) > 0:
        XA_val = np.concatenate([X_val, A_val], axis=1)
        XA_val = scaler.transform(XA_val)
        X_val = XA_val[:, :nb_feat]
        A_val = XA_val[:, nb_feat:]
    return scaler, X_trn, A_trn, X_val, A_val, X_tst, A_tst


def normalise_disturb_whole(scaler, X_trn, A_trn):
    _, nb_feat = X_trn.shape
    X_and_A = np.concatenate([X_trn, A_trn], axis=1)
    scaler = scaler.fit(X_and_A)
    X_and_A = scaler.transform(X_and_A)
    X_trn, A_trn = X_and_A[:, :nb_feat], X_and_A[:, nb_feat:]
    return scaler, X_trn, A_trn


# Seperate/divide datasets
#
# with only one sensitive attribute
#      multiple sensitive attributes


# def sens_attr_divided_set(A, new_attr_name=None):
#     # `param` A (sens_attr): pd.DataFrame
#     # `param` new_attr_name: str
#     tmp_A = A[new_attr_name] if new_attr_name is not None else A
#     tmp_A = tmp_A.values.reshape(-1)  # np.ndarray of np.int64
#     ele_A = np.unique(tmp_A)          # np.ndarray of np.int64
#     idx_A = {}
#     for i in ele_A:
#         idx_A[i] = tmp_A == i  # np.ndarray of np.bool_
#     return ele_A, idx_A
#
#
# def group_of_disjoint_set(X, A, y, ele_i, idx_i):
#     # for ele_i in ele_A:
#     #   idx_i = idx_A[ele_i]
#     Si_X = X[idx_i]
#     Si_A = A[idx_i]
#     Si_y = y[idx_i]
#     return Si_X, Si_A, Si_y
#
#
# def group_of_formulated(Si_X, Si_A, Si_y_fx):
#     Ti_Xy = deepcopy(Si_X)
#     Ti_Xy['label_name'] = Si_y_fx
#     return Ti_Xy.values


# -------------------------------------
# fmanf_ext


def check_bool_feat_columns(dataframe, categorical_feats):
    # dataframe should be `processed_numerical`
    numerical_feats = dataframe.columns.tolist()
    boolean_feats = []
    for attr in categorical_feats:
        boolean_feats.extend([
            i for i in numerical_feats if i.startswith(attr)])
    return boolean_feats


def make_bool_feat_numerical(dataframe, boolean_feats=None):
    # dataframe should be `processed_numerical`
    newframe = dataframe.copy()
    for attr in boolean_feats:
        if newframe[attr].dtype == 'bool':  # DTY_BOL:
            newframe[attr] = newframe[attr].replace({True: 1})
            newframe[attr] = newframe[attr].replace({False: 0})
    return newframe


def check_marginalised_group(dataframe,
                             sensitive_attrs, privileged_vals):
    # dataframe should be `processed_data['original']`
    marginalised_groups = []
    for attr, privileged in zip(sensitive_attrs, privileged_vals):
        marginal_vals = np.unique(dataframe[attr].values).tolist()
        marginal_vals.remove(privileged)
        marginalised_groups.append(marginal_vals)
    return marginalised_groups


def check_marginalised_indices(dataframe,
                               sensitive_attrs, privileged_vals,
                               marginalised_groups):
    # dataframe should be `processed_data['original']`
    marginal_indices = []
    for attr, privileged, marginal_vals in zip(
            sensitive_attrs, privileged_vals, marginalised_groups):
        tmp = (dataframe[attr] == privileged).values.reshape(-1)
        att_indices = [tmp]
        for _, marginal in enumerate(marginal_vals):
            tmp = (dataframe[attr] == marginal).values.reshape(-1)
            att_indices.append(tmp)
        marginal_indices.append(att_indices)
    return marginal_indices


def make_sens_attr_numerical(dataframe,
                             sensitive_attrs, privileged_vals,
                             marginalised_groups=None):
    # dataframe could be `processed_data['categorical_binsensitive']`
    # NO, dataframe should be `processed_data['original /numerical']`
    newframe = dataframe.copy()
    for attr, privileged, marginal_vals in zip(
            sensitive_attrs, privileged_vals, marginalised_groups):
        # replace privileged vals with 1
        newframe[attr] = newframe[attr].replace({privileged: 1})
        # replace all other vals with 0
        for i, marginal in enumerate(marginal_vals):
            newframe[attr] = newframe[attr].replace({marginal: i + 2})
    return newframe


# additional:
#   Add'l. Addt'l. Add.

def process_addtl(dataset, processed_data):
    sen_att_jt = dataset.get_sensitive_attrs_with_joint()
    # sens_attr_with_joint
    # sen_att_jt = sen_att_jt[-1] if len(sen_att_jt) > 1 else None
    if len(sen_att_jt) > 1:
        # processed_orig = processed_data.drop(columns=[sen_att_jt[-1]])
        sub_process_orig = processed_data.drop(columns=sen_att_jt[-1])
    else:
        sub_process_orig = processed_data.copy()  # original data

    # Create a one-hot encoding of the categorical variables.
    sub_process_numerical = pd.get_dummies(
        sub_process_orig, columns=dataset.categorical_feats)
    # 只修改了 非敏感属性 的字符型

    # Create a version of the numerical data for which the sensitve
    # attribute is binary.
    sensitive_attrs = dataset.sensitive_attrs
    privileged_vals = dataset.get_privileged_group('')
    sub_process_binsensitive = make_sensitive_attrs_binary(
        sub_process_numerical, sensitive_attrs, privileged_vals)
    # 既修改了 非敏感属性 的字符型，又修改了 敏感属性 的字符型（变成二类数值型）

    # Create a version of the categorical data for which the sensitive
    # attributes is binary.
    sub_process_categorical_binsensitive = make_sensitive_attrs_binary(
        sub_process_orig, sensitive_attrs, privileged_vals)
    # 只修改了 敏感属性 的字符型（变成二类数值型）

    # Make the class attribute numerical if it wasn't already (just
    # for the bin sensitive version).
    class_attr = dataset.label_name
    pos_val = dataset.positive_label  # FIXME

    sub_process_binsensitive = make_class_attr_num(
        sub_process_binsensitive, class_attr, pos_val)
    # 继续修改了 classification 的字符型（本来就是二分类问题），使之变成数值型

    #
    # ADDITIONALLY

    # return (processed_numerical, processed_categorical_binsensitive,
    #         processed_binsensitive)
    # return (sub_process_orig, sub_process_numerical, sub_process_binsensitive,
    #         sub_process_categorical_binsensitive)
    return {
        "original": sub_process_orig,
        "numerical": sub_process_numerical,
        "numerical-binsensitive": sub_process_binsensitive,
        "categorical-binsensitive": sub_process_categorical_binsensitive}


def process_addtl_multivalue(dataset, sub_process_orig):
    # def process_addtl_multivalues():
    sensitive_attrs = dataset.get_sensitive_attrs_with_joint()
    privileged_vals = dataset.get_privileged_group_with_joint("")
    sensitive_attrs = sensitive_attrs[: 2]
    privileged_vals = privileged_vals[: 2]
    categorical_feats = dataset.categorical_feats

    binarized_data = pd.get_dummies(
        sub_process_orig,
        columns=categorical_feats + sensitive_attrs)
    boolean_feats = check_bool_feat_columns(
        binarized_data, categorical_feats + sensitive_attrs)
    binarized_data = make_bool_feat_numerical(binarized_data, boolean_feats)
    del boolean_feats
    # 把 非敏感属性 和 敏感属性 的字符型 都转换成布尔值 再变成0/1数值型

    #
    # ADDITIONALLY
    # START/BEGIN ADDTIONALLY

    super_proc_numerical = pd.get_dummies(
        sub_process_orig, columns=dataset.categorical_feats)
    boolean_feats = check_bool_feat_columns(
        super_proc_numerical, dataset.categorical_feats)
    super_proc_numerical = make_bool_feat_numerical(
        super_proc_numerical, boolean_feats)
    # 只修改了 非敏感属性 的字符型，变为布尔值后 再换成0/1数值

    marginalised_groups = check_marginalised_group(
        super_proc_numerical, sensitive_attrs[:2], privileged_vals[:2])
    super_proc_binsensitive = make_sens_attr_numerical(
        super_proc_numerical, sensitive_attrs[:2], privileged_vals[:2],
        marginalised_groups)
    # 既修改了 非敏感属性 的字符型，又修改了 敏感属性 的字符型（变为多类数值型）

    super_proc_categorical_binsensitive = make_sens_attr_numerical(
        sub_process_orig, sensitive_attrs[:2], privileged_vals[:2],
        marginalised_groups)
    # 只修改了 敏感属性 的字符型（变成多类数值型）

    class_attr = dataset.label_name
    pos_val = dataset.positive_label  # FIXME
    super_proc_binsensitive = make_class_attr_num(
        super_proc_binsensitive, class_attr, pos_val)
    # 继续修改了 classification 的字符型（本来就是二分类问题），使之变成数值型

    # END ADDITIONALLY
    binarized_data = make_class_attr_num(binarized_data, class_attr, pos_val)
    return {
        # "original": sub_process_orig,
        "numerical": super_proc_numerical,
        "numerical-multisen": super_proc_binsensitive,
        "categorical-multisen": super_proc_categorical_binsensitive,
        # "numerical-multisensitive": super_proc_binsensitive,
        # "categorical-multisensitive": super_proc_categorical_binsensitive,
        "marginalised_groups": marginalised_groups,
        # "binarized-numerical-sensitive": binarized_data}
        "binarized-numerical-sen": binarized_data}


def renewed_prep_and_adversarial(dataset, data_frame, ratio=.7,
                                 logger=None):
    """ refers to:
    - def preprocess(dataset, data_frame, logger=None):
    - def adversarial(dataset, data_frame, ratio=.4, logger=None):
    """

    processed_data = process_above(dataset, data_frame, logger)
    since = time.time()
    disturbed_data = adverse_perturb(dataset, processed_data, ratio)
    tim_elapsed = time.time() - since
    del since

    preproc_bin = process_addtl(dataset, processed_data)
    preproc_mu = process_addtl_multivalue(dataset, preproc_bin['original'])
    perturb_bin = process_addtl(dataset, disturbed_data)
    perturb_mu = process_addtl_multivalue(dataset, perturb_bin['original'])

    belongs_priv = dataset.find_where_belongs(processed_data)
    if len(belongs_priv) > 1:
        belongs_priv_with_joint = [
            np.logical_and(belongs_priv[0], belongs_priv[1]),
            np.logical_or(belongs_priv[0], belongs_priv[1]),
        ]  # .astype(DTY_BOL)
        belongs_priv.extend(belongs_priv_with_joint)
    marginalised_groups = preproc_mu['marginalised_groups']
    # pdb.set_trace()
    del preproc_mu['marginalised_groups']
    del perturb_mu['marginalised_groups']

    return {
        'processed_data': processed_data,
        'disturbed_data': disturbed_data,
        'marginalised_groups': marginalised_groups,
        'belongs_priv': belongs_priv,
        'perturbation_tim_elapsed': tim_elapsed,
    }, preproc_bin, preproc_mu, perturb_bin, perturb_mu


def renewed_transform_X_A_and_y(dataset, processed_binsensitive,
                                with_joint=False):
    y = processed_binsensitive[dataset.label_name]

    sensitive_attrs = deepcopy(dataset.sensitive_attrs)
    if with_joint and len(sensitive_attrs) > 1:
        new_attr_name = '-'.join(sensitive_attrs)
        sensitive_attrs.append(new_attr_name)
    else:
        new_attr_name = None
    A = deepcopy(processed_binsensitive[sensitive_attrs])

    X = deepcopy(processed_binsensitive)
    X.drop(columns=dataset.label_name, inplace=True)
    # X = processed_binsensitive.drop(columns=dataset.label_name)
    X.drop(columns=sensitive_attrs, inplace=True)
    return X, A, y, new_attr_name


def renewed_normalise_separate(X_A_trn, X_A_val, X_A_tst,
                               # saIndex=list()):
                               saIndex=tuple()):
    nb_feat = X_A_trn.shape[1]
    non_sa = list(range(nb_feat))
    for i in saIndex:
        non_sa.remove(i)
    X_trn, A_trn = X_A_trn[:, non_sa], X_A_trn[:, saIndex]
    X_tst, A_tst = X_A_tst[:, non_sa], X_A_tst[:, saIndex]
    if len(X_A_val) > 0:
        X_val = X_A_val[:, non_sa]
        A_val = X_A_val[:, saIndex]
    else:
        X_val, A_val = [], []
    return X_trn, A_trn, X_val, A_val, X_tst, A_tst


def renewed_normalise_disturb(scaler, X_A_trn, X_A_val, X_A_tst,
                              # saIndex=list(), trans_A=False):
                              saIndex=tuple(), trans_A=False):
    # all are np.ndarray, not pd.DataFrame
    scaler = scaler.fit(X_A_trn)
    new_XA_trn = scaler.transform(X_A_trn)
    new_XA_tst = scaler.transform(X_A_tst)
    if len(X_A_val) > 0:
        new_XA_val = scaler.transform(X_A_val)
    else:
        new_XA_val = []

    if not trans_A:
        for sa_idx in saIndex:
            if len(X_A_val) > 0:
                new_XA_val[:, sa_idx] = X_A_val[:, sa_idx]
            new_XA_trn[:, sa_idx] = X_A_trn[:, sa_idx]
            new_XA_tst[:, sa_idx] = X_A_tst[:, sa_idx]

    X_trn, A_trn, X_val, A_val, X_tst, A_tst = renewed_normalise_separate(
        new_XA_trn, new_XA_val, new_XA_tst, saIndex=saIndex)

    return (scaler, new_XA_trn, new_XA_val, new_XA_tst,
            X_trn, A_trn, X_val, A_val, X_tst, A_tst)


def renewed_transform_disturb(X, A, y, Aq, index,
                              marginal_indices,
                              # belongs_priv,
                              belongs_priv_with_joint):
    """ X, A, y, Aq: np.ndarray or pd.DataFrame, where Aq may be None
    marginal_indices       : list, len(marginal_indices) == na
               each element:   for this sens attr,
                                   index of belongs_priv (val#1),
                                   index of `sen_att_val#2`,
                                   index of `sen_att_val#3`,
    belongs_priv_with_joint: list, len()= 0 or 2, joint_and/or
    """

    if isinstance(X, pd.DataFrame):
        X_idx = X.iloc[index]
        y_idx = y.iloc[index]
        A_idx = A if (A is None) else A.iloc[index]
        Aq_idx = Aq if (Aq is None) else Aq.iloc[index]
    elif isinstance(X, np.ndarray):
        X_idx = X[index]
        y_idx = y[index]
        A_idx = A if (A is None) else A[index]
        Aq_idx = Aq if (Aq is None) else Aq[index]

    start_with_not_unpriv = []
    for marg_idx in marginal_indices:
        tmp = [t[index] for t in marg_idx]
        start_with_not_unpriv.append(tmp)
    if not belongs_priv_with_joint:
        joint_ = belongs_priv_with_joint
    elif len(belongs_priv_with_joint) == 2:
        joint_ = [t[index] for t in belongs_priv_with_joint]
    else:
        joint_ = np.array(belongs_priv_with_joint)[index].tolist()
    return X_idx, A_idx, y_idx, Aq_idx, start_with_not_unpriv, joint_


# -------------------------------------
