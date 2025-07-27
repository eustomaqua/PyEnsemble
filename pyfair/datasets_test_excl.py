# coding: utf-8

# from copy import deepcopy
# import pdb


def excl_test_datasets():
    from pyfair.datasets import (
        Ricci, German, Adult, PropublicaRecidivism,
        PropublicaViolentRecidivism, preprocess)

    dt = Ricci()
    dt = German()
    dt = Adult()
    dt = PropublicaRecidivism()
    dt = PropublicaViolentRecidivism()

    df = dt.load_raw_dataset()
    ans = preprocess(dt, df)
    assert isinstance(ans, dict)
    # pdb.set_trace()
    return


def test_preprocessing():
    # from fairml.preprocessing import (
    #     adversarial)#,transform_X_and_y,transform_unpriv_tag)
    from pyfair.datasets import preprocess, DATASETS
    from pyfair.preprocessing_dr import (
        adversarial, transform_X_and_y, transform_unpriv_tag)
    from pyfair.preprocessing_hfm import (
        binarized_data_set, transform_X_A_and_y,
        # transform_unpriv_tag_prime, renewed_prep_and_adversarial,
        # renewed_transform_X_A_and_y, check_marginalised_indices)
        renewed_prep_and_adversarial, renewed_transform_X_A_and_y,
        check_marginalised_indices)

    for dt in DATASETS[1: 2]:
        df = dt.load_raw_dataset()
        ans = preprocess(dt, df)
        adv = adversarial(dt, df, ratio=.95)

        for k in [
            'original', 'numerical', 'numerical-binsensitive',
                'categorical-binsensitive']:
            assert ans[k].shape == adv[k].shape
        # pdb.set_trace()

        proc_dt = ans['numerical-binsensitive']  # processed_dat
        dist_dt = adv['numerical-binsensitive']  # disturbed_dat
        X, y = transform_X_and_y(dt, proc_dt)
        Xp, _ = transform_X_and_y(dt, dist_dt)
        non_sa, _ = transform_unpriv_tag(dt, df)
        assert X.shape[0] == y.shape[0] == Xp.shape[0]

        pos_label = dt.get_positive_class_val(
            'numerical-binsensitive')  # '')
        priv_nam = dt.get_sensitive_attrs_with_joint()[:2]
        priv_val = dt.get_privileged_group('numerical-binsensitive')
        # priv_val = dt.get_privileged_group_with_joint(
        #     'numerical-binsensitive')[:2]
        assert len(non_sa) == len(priv_nam) == len(priv_val)
        assert pos_label == 1

        proc_dt_bin = binarized_data_set(proc_dt)
        dist_dt_bin = binarized_data_set(dist_dt)
        X, A, y, new_attr = transform_X_A_and_y(dt, proc_dt_bin)
        _, Ap, _, _ = transform_X_A_and_y(dt, dist_dt_bin)
        # nsa_bin, idx_jt = transform_unpriv_tag_prime(dt, df, 'both')
        nsa_bin, idx_jt = transform_unpriv_tag(dt, df, 'both')
        assert len(A) == len(Ap) == len(X) == len(y)
        assert (not new_attr) or isinstance(new_attr, str)
        assert (non_sa[0] == nsa_bin[0]).all()      # idx_non_sa#1
        if idx_jt:  # or if new_attr:
            assert (non_sa[1] == nsa_bin[1]).all()  # idx_non_sa#2

        re_dft = renewed_prep_and_adversarial(dt, df, ratio=.97)
        tmp = re_dft[0]['processed_data']
        X1, A1, y1, new_attr = renewed_transform_X_A_and_y(
            dt, tmp, with_joint=False)  # renew_dtf[1],
        X2, A2, y2, _ = renewed_transform_X_A_and_y(
            dt, tmp, with_joint=True)
        X3, A3, y3, _ = transform_X_A_and_y(dt, tmp)
        assert new_attr not in X1.columns.tolist()
        assert X1.shape[1] - 1 == X2.shape[1] == X3.shape[1]
        assert A1.shape[1] + 1 == A2.shape[1] == A3.shape[1]
        assert (X2 == X3).all().all()
        assert (A2 == A3).all().all()
        assert id(y1) != id(y2) != id(y3)
        assert (y1 == y2).all() and (y2 == y3).all()

        tmp = re_dft[1]['numerical-binsensitive']
        assert new_attr not in tmp.columns.tolist()
        X1, A1, _, _ = renewed_transform_X_A_and_y(dt, tmp, False)
        # X2, A2, _, _ = renewed_transform_X_A_and_y(dt, tmp, True)
        # X3, A3, _, _ = transform_X_A_and_y(dt, tmp)
        # tmp = re_dft[3]['numerical-binsensitive']
        tmp = re_dft[2]['numerical-multisen']
        assert new_attr not in tmp.columns.tolist()
        X2, A2, _, _ = renewed_transform_X_A_and_y(dt, tmp, False)
        assert (X1 == X2).all().all()
        assert X1.shape == X2.shape
        assert A1.shape == A2.shape

        # tmp = re_dft[0]['perturbation_tim_elapsed']
        mrg_grp = re_dft[0]['marginalised_groups']
        sen_att_indices = check_marginalised_indices(
            df, priv_nam,  # dt.sensitive_attrs,
            dt.get_privileged_group(''), mrg_grp)
        assert len(sen_att_indices) == len(mrg_grp) == 2
        # pdb.set_trace()
        assert len(sen_att_indices[0]) == len(mrg_grp[0]) + 1
        assert len(sen_att_indices[1]) == len(mrg_grp[1]) + 1
    return
