# coding: utf-8


import os
import os.path as osp
# import numpy as np
import pandas as pd
from pyfair.facil.utils_saver import elegant_print


__all__ = [
    'Ricci', 'German', 'Adult', 'PropublicaRecidivism',
    'PropublicaViolentRecidivism', 'DATASETS',
    'DATASET_NAMES',  # 'prepare_data',
    'process_above', 'process_below', 'preprocess',

    'RAW_EXPT_DIR', 'AVAILABLE_FAIR_DATASET',
    'make_sensitive_attrs_binary', 'make_class_attr_num',
]


# ===========================
# utils


def local_root_path():
    home = os.getcwd()
    # Ubuntu : '/media/sf_GitD/FairML'
    # Windows: 'D:\\GitH\\FairML'
    # Mac    : '~/GitH/FairML'
    return home


def local_data_path():
    home = local_root_path()
    path = osp.join(home, 'data')
    # path = osp.join(home, 'pyfair', 'data')
    return path


PACKAGE_DIR = local_root_path()
RAW_DATA_DIR = local_data_path()
RAW_EXPT_DIR = os.path.join(PACKAGE_DIR, 'findings')


# ===========================
# data


class Data:
    def __init__(self):
        pass

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        self._dataset_name = value

    @property
    def label_name(self):
        # or self._class_attr
        return self._label_name

    @label_name.setter
    def label_name(self, value):
        self._label_name = value

    @property
    def positive_label(self):
        return self._positive_label

    @positive_label.setter
    def positive_label(self, value):
        self._positive_label = value

    @property
    def sensitive_attrs(self):
        return self._sensitive_attrs

    @sensitive_attrs.setter
    def sensitive_attrs(self, value):
        self._sensitive_attrs = value

    @property
    def privileged_vals(self):
        return self._privileged_vals

    @privileged_vals.setter
    def privileged_vals(self, value):
        self._privileged_vals = value

    def get_positive_class_val(self, tag):
        if tag == 'numerical-binsensitive':
            return 1
        return self._positive_label

    def get_sensitive_attrs_with_joint(self):
        sens_attrs = self._sensitive_attrs
        if len(sens_attrs) <= 1:
            return sens_attrs
        return sens_attrs + ['-'.join(sens_attrs)]

    def get_privileged_group(self, tag):
        if tag == 'numerical-binsensitive':
            return [1 for x in self._sensitive_attrs]
        return self._privileged_vals  # list

    def get_privileged_group_with_joint(self, tag):
        priv_class_names = self.get_privileged_group(tag)
        if len(priv_class_names) <= 1:
            return priv_class_names
        # in this way, at most two sensitive attributes
        return priv_class_names + [
            '-'.join(str(v) for v in priv_class_names)]

    @property
    def categorical_feats(self):
        return self._categorical_feats

    @categorical_feats.setter
    def categorical_feats(self, value):
        self._categorical_feats = value

    @property
    def feats_to_keep(self):
        return self._feats_to_keep

    @feats_to_keep.setter
    def feats_to_keep(self, value):
        self._feats_to_keep = value

    @property
    def missing_val_indicators(self):
        return self._missing_val_indicators

    @missing_val_indicators.setter
    def missing_val_indicators(self, value):
        self._missing_val_indicators = value

    @property
    def raw_filename(self):
        return "baseline_{}.csv".format(self._dataset_name)

    def load_raw_dataset(self):
        data_path = osp.join(RAW_DATA_DIR, self.raw_filename)
        data_frame = pd.read_csv(
            data_path,
            # error_bad_lines=False,
            on_bad_lines='warn',  # 'skip'
            na_values=self._missing_val_indicators,
            encoding='ISO-8859-1')
        # data_frame = self.data_specific_processing(data_frame)
        return data_frame

    def data_specific_processing(self, data_frame):
        return data_frame

    def handle_missing_data(self, data_frame):
        return data_frame

    def get_class_balance_statistics(self, data_frame):
        # if data_frame is None:
        #   data_frame = self.load_raw_dataset()
        r = data_frame.groupby(self._label_name).size()
        return r

    # def get_sensitive_attr_balance_stats(self, data_frame=None):
    def get_sens_attr_balance_stats(self, data_frame):
        # if data_frame is None:
        #   data_frame = self.load_raw_dataset()
        return [data_frame.groupby(
            a).size() for a in self._sensitive_attrs]

    def find_where_belongs(self, data_frame):
        # if data_frame is None:
        #   data_frame = self.load_raw_dataset()
        sens_attrs = self._sensitive_attrs
        priv_value = self._privileged_vals
        return [
            # (data_frame[sa] == pv).tolist()
            (data_frame[sa] == pv).to_numpy()
            for sa, pv in zip(sens_attrs, priv_value)]


# ===========================
# dataset(s)


class Adult(Data):
    def __init__(self):
        super().__init__()

        self.dataset_name = 'adult'
        self.label_name = 'income-per-year'
        self.positive_label = '>50K'
        self.sensitive_attrs = ['race', 'sex']
        self.privileged_vals = ['White', 'Male']

        self.feats_to_keep = [
            'age', 'workclass', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship',
            'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income-per-year'
        ]

        self.categorical_feats = [
            'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'native-country'
        ]

        self.missing_val_indicators = ['?']

    def handle_missing_data(self, data_frame):
        return data_frame


class German(Data):
    def __init__(self):
        super().__init__()

        self.dataset_name = 'german'
        self.label_name = 'credit'
        self.positive_label = 1
        self.sensitive_attrs = ['sex', 'age']
        self.privileged_vals = ['male', 'adult']

        self.feats_to_keep = [
            'status', 'month', 'credit_history', 'purpose',
            'credit_amount',
            'savings', 'employment', 'investment_as_income_percentage',
            'personal_status', 'other_debtors', 'residence_since',
            'property', 'age', 'installment_plans', 'housing',
            'number_of_credits', 'skill_level', 'people_liable_for',
            'telephone', 'foreign_worker', 'credit'
        ]

        self.categorical_feats = [
            'status', 'credit_history', 'purpose', 'savings',
            'employment',
            'other_debtors', 'property', 'installment_plans',
            'housing',
            'skill_level', 'telephone', 'foreign_worker'
        ]

        self.missing_val_indicators = []

    def data_specific_processing(self, data_frame):
        # adding a derived sex attribute based on personal_status
        sexdict = {'A91': 'male', 'A93': 'male', 'A94': 'male',
                   'A92': 'female', 'A95': 'female'}

        data_frame = data_frame.assign(personal_status=data_frame[
            'personal_status'].replace(to_replace=sexdict))
        data_frame = data_frame.rename(columns={
            'personal_status': 'sex'})

        # adding a derived binary age attribute (youth vs. adult) such that
        # >= 25 is adult
        # this is based on an analysis by Kamiran and Calders
        # http://ieeexplore.ieee.org/document/4909197/
        # showing that this division creates the most discriminatory
        # possibilities.

        old = data_frame['age'] >= 25
        data_frame.loc[old, 'age'] = 'adult'
        young = data_frame['age'] != 'adult'
        data_frame.loc[young, 'age'] = 'youth'

        return data_frame


def passing_grade(row):
    """ A passing grade in the Ricci data is defined as any grade above
    a 70 in the combined oral and written score. (See Miao 2010.)
    """
    if row['Combine'] >= 70.0:
        return 1
    return 0


class Ricci(Data):
    def __init__(self):
        super().__init__()

        self.dataset_name = 'ricci'
        # Class attribute will not be created until
        # data_specific_processing is run.
        self.label_name = 'Class'
        self.positive_label = 1
        self.sensitive_attrs = ['Race']
        self.privileged_vals = ['W']

        self.feats_to_keep = [
            'Position', 'Oral', 'Written', 'Race', 'Combine']
        self.categorical_feats = ['Position']
        self.missing_val_indicators = []

    def data_specific_processing(self, data_frame):
        data_frame['Class'] = data_frame.apply(passing_grade, axis=1)
        return data_frame


class PropublicaRecidivism(Data):
    def __init__(self):
        super().__init__()

        self.dataset_name = 'propublica-recidivism'
        self.label_name = 'two_year_recid'
        self.positive_label = 1
        self.sensitive_attrs = ['sex', 'race']
        self.privileged_vals = ['Male', 'Caucasian']

        self.feats_to_keep = [
            "sex", "age", "age_cat", "race", "juv_fel_count",
            "juv_misd_count", "juv_other_count", "priors_count",
            "c_charge_degree", "c_charge_desc", "decile_score",
            "score_text", "two_year_recid", "days_b_screening_arrest",
            "is_recid"
        ]

        self.categorical_feats = [
            'age_cat', 'c_charge_degree', 'c_charge_desc']
        # days_b_screening_arrest, score_text, decile_score, and
        # is_recid will be dropped after data specific processing
        # is done

        self.missing_val_indicators = []

    def data_specific_processing(self, data_frame):
        # Filter as done here:
        # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb

        data_frame = data_frame[
            (data_frame.days_b_screening_arrest <= 30) &
            (data_frame.days_b_screening_arrest >= -30) &
            (data_frame.is_recid != -1) &
            (data_frame.c_charge_degree != '0') &
            (data_frame.score_text != 'N/A')]

        data_frame = data_frame.drop(columns=[
            'days_b_screening_arrest', 'is_recid', 'decile_score',
            'score_text'])

        return data_frame


class PropublicaViolentRecidivism(Data):
    def __init__(self):
        super().__init__()

        self.dataset_name = 'propublica-violent-recidivism'
        self.label_name = 'two_year_recid'
        self.positive_label = 1
        self.sensitive_attrs = ['sex', 'race']
        self.privileged_vals = ['Male', 'Caucasian']

        self.feats_to_keep = [
            "sex", "age", "age_cat", "race", "juv_fel_count",
            "juv_misd_count", "juv_other_count", "priors_count",
            "c_charge_degree", "c_charge_desc", "decile_score",
            "score_text", "two_year_recid", "days_b_screening_arrest",
            "is_recid"
        ]

        self.categorical_feats = [
            'age_cat', 'c_charge_degree', 'c_charge_desc']
        # days_b_screening_arrest, score_text, decile_score, and is_recid
        # will be dropped after data specific processing is done

        self.missing_val_indicators = []

    def data_specific_processing(self, data_frame):
        # Filter as done here:
        # https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
        #
        # The filter for violent recidivism as done above filters out
        # v_score_text instead of score_text, but since we want to include
        # the score_text before the violent recidivism, we think the right
        # thing to do here is to filter score_text.

        data_frame = data_frame[
            (data_frame.days_b_screening_arrest <= 30) &
            (data_frame.days_b_screening_arrest >= -30) &
            (data_frame.is_recid != -1) &
            (data_frame.c_charge_degree != '0') &
            (data_frame.score_text != 'N/A')]

        data_frame = data_frame.drop(columns=[
            'days_b_screening_arrest', 'is_recid', 'decile_score',
            'score_text'])

        # '' '
        # # >>> np.all(df['two_year_recid.1'] == df.two_year_recid)
        # # True
        # data_frame = data_frame.drop(columns=['two_year_recid.1'])
        # '' '
        # # BUG. Cannot delete this column.

        return data_frame


# ===========================
# preprocess


def make_sensitive_attrs_binary(dataframe,
                                sensitive_attrs, privileged_vals):
    newframe = dataframe.copy()
    for attr, privileged in zip(sensitive_attrs, privileged_vals):
        # replace privileged vals with 1
        newframe[attr] = newframe[attr].replace({privileged: 1})
        # replace all other vals with 0
        newframe[attr] = newframe[attr].replace("[^1]", 0, regex=True)
    return newframe


def make_class_attr_num(dataframe, class_attr, positive_val):
    # don't change the class attribute unless its a string (pandas
    # type: object)
    if dataframe[class_attr].dtype == 'object':
        dataframe[class_attr] = dataframe[
            class_attr].replace({positive_val: 1})
        dataframe[class_attr] = dataframe[
            class_attr].replace("[^1]", 0, regex=True)
    return dataframe


def process_above(dataset, data_frame, logger):
    # Remove any columns not included in the list of features to keep.
    smaller_data = data_frame[dataset.feats_to_keep]

    # Handle missing data.
    missing_processed = dataset.handle_missing_data(smaller_data)

    # Remove any rows that have missing data.
    missing_data_removed = missing_processed.dropna()
    missing_data_count = missing_processed.shape[
        0] - missing_data_removed.shape[0]
    if missing_data_count > 0:
        elegant_print(
            "Missing Data: " + str(missing_data_count) + 
            " rows removed from dataset " + dataset.dataset_name, logger)

    # Do any data specific processing.
    processed_data = dataset.data_specific_processing(
        missing_data_removed)

    elegant_print("\n-------------------", logger)
    elegant_print("Balance statistics:", logger)
    elegant_print("\nClass:", logger)
    elegant_print(
        dataset.get_class_balance_statistics(processed_data), logger)
    elegant_print("\nSensitive Attribute:", logger)
    for r in dataset.get_sens_attr_balance_stats(processed_data):
        elegant_print(r, logger)
        elegant_print("\n", logger)
    elegant_print("\n", logger)

    # Handle multiple sensitive attributes by creating a new attribute
    # that's the joint distribution of all of those attributes.
    # For example, if a dataset has both 'Race' and 'Gender', the
    # combined feature 'Race-Gender' is created that has attributes,
    # e.g., 'White-Woman'.
    sensitive_attrs = dataset.sensitive_attrs
    if len(sensitive_attrs) > 1:
        new_attr_name = '-'.join(sensitive_attrs)
        # TODO: the below may fail for non-string attributes
        processed_data = processed_data.assign(
            temp_name=processed_data[
                sensitive_attrs].apply('-'.join, axis=1))
        processed_data = processed_data.rename(
            columns={'temp_name': new_attr_name})
        # dataset.append_sensitive_attribute(new_attr_name)
        # privileged_joint_vals = '-'.join(dataset.get_privileged_class_names(""))
        # dataset.get_privileged_class_names("").append(privileged_joint_vals)

    return processed_data


def process_below(dataset, processed_data):
    # Create a one-hot encoding of the categorical variables.
    processed_numerical = pd.get_dummies(
        processed_data,
        columns=dataset.categorical_feats)

    # Create a version of the numerical data for which the sensitive
    # attribute is binary.
    sensitive_attrs = dataset.get_sensitive_attrs_with_joint()
    privileged_vals = dataset.get_privileged_group_with_joint("")
    processed_binsensitive = make_sensitive_attrs_binary(
        processed_numerical, sensitive_attrs, privileged_vals)

    # Create a version of the categorical data for which the sensitive
    # attributes is binary.
    processed_categorical_binsensitive = make_sensitive_attrs_binary(
        processed_data, sensitive_attrs,
        dataset.get_privileged_group(""))  # FIXME
    # Make the class attribute numerical if it wasn't already (just
    # for the bin sensitive version).
    class_attr = dataset.label_name
    pos_val = dataset.positive_label  # FIXME

    processed_binsensitive = make_class_attr_num(
        processed_binsensitive, class_attr, pos_val)

    return processed_numerical, processed_binsensitive, processed_categorical_binsensitive


def preprocess(dataset, data_frame, logger=None):
    """ The preprocess function takes a pandas data frame and returns
    two modified data frames:
    1) all the data as given with any features that should not be used
       for training or fairness analysis removed.
    2) only the numerical and ordered categorical data, sensitive
       attributes, and class attribute.
       Categorical attributes are one-hot encoded.
    3) the numerical data (#2) but with a binary (numerical) sensitive
       attribute
    """

    processed_data = process_above(dataset, data_frame, logger)

    processed_numerical, processed_binsensitive, \
        processed_categorical_binsensitive = process_below(
            dataset, processed_data)

    return {
        "original": processed_data,
        "numerical": processed_numerical,
        "numerical-binsensitive": processed_binsensitive,
        "categorical-binsensitive": processed_categorical_binsensitive}


DATASETS = [
    # Real datasets:
    Ricci(),
    Adult(),   # Income
    German(),  # Credit
    PropublicaRecidivism(),
    PropublicaViolentRecidivism(),
]

DATASET_NAMES = [
    'ricci', 'adult', 'german',
    'propublica-recidivism',
    'propublica-violent-recidivism',
]

AVAILABLE_FAIR_DATASET = [
    # 'ricci', 'german', 'adult', 'ppc', 'ppvc'
    'ricci', 'german', 'adult', 'ppr', 'ppvr'
]


def prepare_data(dataset_name, logger=None):
    for ds in DATASETS:
        if ds.dataset_name != dataset_name:
            continue

        elegant_print(
            "--- Processing dataset: %s ---" % ds.dataset_name, logger)
        data_frame = ds.load_raw_dataset()
        d = preprocess(ds, data_frame, logger)

        return d
    raise ValueError("No dataset named `{}`.".format(dataset_name))


# ===========================
#
