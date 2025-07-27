# coding: utf-8

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


import numpy as np
from pyfair.facil.utils_const import check_equal, synthetic_set

nb_inst, nb_labl = 100, 5
nb_cls, nb_pru = 11, 3
bi_y, bi_yt, bi_c = synthetic_set(2, nb_inst, nb_cls)
mu_y, mu_yt, mu_c = synthetic_set(nb_labl, nb_inst, nb_cls)
tr_yt = np.array(bi_yt) * 2 - 1
tr_yt = tr_yt.tolist()
tr_y = [i * 2 - 1 for i in bi_y]

ki, kj = 0, 1  # idx
tr_ha, tr_hb = tr_yt[ki], tr_yt[kj]
bi_ha, bi_hb = bi_yt[ki], bi_yt[kj]
mu_ha, mu_hb = mu_yt[ki], mu_yt[kj]


# ----------------------------------
# margineantu1997pruning


def test_early_stopping():
    from pyfair.granite.ensem_pruning import Early_Stopping
    _, tr_P, tr_seq = Early_Stopping(tr_yt, nb_cls, nb_pru)
    _, bi_P, bi_seq = Early_Stopping(bi_yt, nb_cls, nb_pru)
    _, mu_P, mu_seq = Early_Stopping(mu_yt, nb_cls, nb_pru)

    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_seq, bi_seq))
    assert np.sum(tr_P) == len(tr_seq) == nb_pru
    assert np.sum(bi_P) == len(bi_seq) == nb_pru

    assert np.all(np.equal(bi_P, mu_P))
    assert np.all(np.equal(bi_seq, mu_seq))
    assert np.sum(mu_P) == len(mu_seq) == nb_pru
    assert len(set(map(id, [tr_P, bi_P, mu_P]))) == 3

    assert len(set(map(id, [tr_seq, bi_seq, mu_seq]))) == 3
    assert id(tr_P) != id(tr_seq)
    assert id(bi_P) != id(bi_seq)
    assert id(mu_P) != id(mu_seq)


def test_KL_divergence():
    from pyfair.granite.ensem_pruning import (
        _softmax, _KLD, _KLD_vectors,
        _JU_set_of_vectors, _U_next_idx,
        KL_divergence_Pruning)
    assert check_equal(np.sum(_softmax(tr_y)), 1.)
    assert check_equal(np.sum(_softmax(bi_y)), 1.)
    assert check_equal(np.sum(_softmax(mu_y)), 1.)
    tr_ans = np.sum(_softmax(tr_yt), axis=0)
    bi_ans = np.sum(_softmax(bi_yt), axis=0)
    mu_ans = np.sum(_softmax(mu_yt), axis=0)
    assert all(check_equal(i, 1) for i in tr_ans)
    assert all(check_equal(i, 1) for i in bi_ans)
    assert all(check_equal(i, 1) for i in mu_ans)

    import scipy.stats as stats
    from pyfair.marble.data_entropy import prob
    px, _ = prob(tr_ha)
    py, _ = prob(tr_hb)
    assert check_equal(_KLD(px, py), stats.entropy(px, py))
    px, _ = prob(bi_ha)
    py, _ = prob(bi_hb)
    assert check_equal(_KLD(px, py), stats.entropy(px, py))
    px, _ = prob(mu_ha)
    py, _ = prob(mu_hb)
    assert check_equal(_KLD(px, py), stats.entropy(px, py))

    tr_ans = _KLD_vectors(tr_ha, tr_hb)
    bi_ans = _KLD_vectors(bi_ha, bi_hb)
    mu_ans = _KLD_vectors(mu_ha, mu_hb)
    assert tr_ans == bi_ans  # check_equal(tr_ans, bi_ans)
    assert all(isinstance(
        i, float) for i in [tr_ans, bi_ans, mu_ans])
    # assert tr_ans != KLD_vectors(tr_hb, tr_ha)
    # assert bi_ans != KLD_vectors(bi_hb, bi_ha)
    # assert mu_ans != KLD_vectors(mu_hb, mu_ha)
    assert len(set([id(tr_ans), id(bi_ans), id(mu_ans)])) == 3

    px, py = prob(tr_ha)[0], prob(tr_hb)[0]
    assert _KLD_vectors(tr_ha, tr_hb) == _KLD_vectors(bi_ha, bi_hb)
    assert _KLD_vectors(tr_hb, tr_ha) == _KLD_vectors(bi_hb, bi_ha)
    if np.all(np.equal(px, py)) or np.all(np.equal(px, py[::-1])):
        assert _KLD_vectors(tr_ha, tr_hb) == _KLD_vectors(
            tr_hb, tr_ha)
        assert _KLD_vectors(bi_ha, bi_hb) == _KLD_vectors(
            bi_hb, bi_ha)
    else:
        assert _KLD_vectors(tr_ha, tr_hb) != _KLD_vectors(
            tr_hb, tr_ha)
        assert _KLD_vectors(bi_ha, bi_hb) != _KLD_vectors(
            bi_hb, bi_ha)
    px, py = prob(mu_ha)[0], prob(mu_hb)[0]
    # assert (_KLD_vectors(mu_ha, mu_hb), _KLD_vectors(mu_hb, mu_ha))

    tr_ans = _JU_set_of_vectors(tr_yt)
    bi_ans = _JU_set_of_vectors(bi_yt)
    mu_ans = _JU_set_of_vectors(mu_yt)
    assert tr_ans == bi_ans
    assert all(isinstance(
        i, float) for i in [tr_ans, bi_ans, mu_ans])
    assert len(set(map(id, [tr_ans, bi_ans, mu_ans]))) == 3

    P = np.random.randint(2, size=nb_cls, dtype='bool').tolist()
    tr_ans = _U_next_idx(tr_yt, P)
    bi_ans = _U_next_idx(bi_yt, P)
    mu_ans = _U_next_idx(mu_yt, P)
    assert tr_ans == bi_ans
    assert all(isinstance(i, int) for i in [tr_ans, bi_ans, mu_ans])
    assert id(tr_ans) == id(bi_ans)
    # might achieve:  assert id(tr_ans) != id(mu_ans)
    # might achieve:  assert id(bi_ans) != id(mu_ans)

    _, tr_P, tr_seq = KL_divergence_Pruning(tr_yt, nb_cls, nb_pru)
    _, bi_P, bi_seq = KL_divergence_Pruning(bi_yt, nb_cls, nb_pru)
    _, mu_P, mu_seq = KL_divergence_Pruning(mu_yt, nb_cls, nb_pru)
    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_seq, bi_seq))
    assert sum(tr_P) == len(tr_seq) == nb_pru
    assert sum(bi_P) == len(bi_seq) == nb_pru
    assert sum(mu_P) == len(mu_seq) == nb_pru
    assert all(isinstance(i, int) for i in mu_seq)
    assert len(set(map(id, mu_seq))) == nb_pru


def test_KL_divergence_modify():
    from pyfair.granite.ensem_pruning import (
        _KLD_pq, _J, _KL_find_next,
        KL_divergence_Pruning_modify)
    tr_ans = _KLD_pq(tr_ha, tr_hb)
    bi_ans = _KLD_pq(bi_ha, bi_hb)
    mu_ans = _KLD_pq(mu_ha, mu_hb)
    assert all(isinstance(i, float) for i in [tr_ans, bi_ans, mu_ans])
    assert len(set(map(id, [tr_ans, bi_ans, mu_ans]))) == 3

    tr_ans = _J(tr_yt)
    bi_ans = _J(bi_yt)
    mu_ans = _J(mu_yt)
    assert all(isinstance(i, float) for i in [tr_ans, bi_ans, mu_ans])
    assert len(set(map(id, [tr_ans, bi_ans, mu_ans]))) == 3

    P = np.random.randint(2, size=nb_cls, dtype='bool').tolist()
    tr_ans = _KL_find_next(tr_yt, P)
    bi_ans = _KL_find_next(bi_yt, P)
    mu_ans = _KL_find_next(mu_yt, P)
    assert all(isinstance(i, int) for i in [tr_ans, bi_ans, mu_ans])
    assert len(set(map(id, [tr_ans, bi_ans, mu_ans]))) <= 3  # == 2
    # might not achieve there: assert tr_ans == bi_ans

    _, tr_P, tr_seq = KL_divergence_Pruning_modify(
        tr_yt, nb_cls, nb_pru)
    _, bi_P, bi_seq = KL_divergence_Pruning_modify(
        bi_yt, nb_cls, nb_pru)
    _, mu_P, mu_seq = KL_divergence_Pruning_modify(
        mu_yt, nb_cls, nb_pru)
    assert sum(tr_P) == len(tr_seq) == nb_pru
    assert sum(bi_P) == len(bi_seq) == nb_pru
    assert sum(mu_P) == len(mu_seq) == nb_pru
    # NOT TRUE# assert np.all(np.equal(tr_P, bi_P))
    # NOT TRUE# assert np.all(np.equal(tr_seq, bi_seq))
    # Since it uses `softmax`, {-1,+1} {0,1} might get different values
    # of `J(U)`, leading to different values in `ansJ`, which would
    # probably cause different return of `KL_find_next`.


def test_kappa():
    from pyfair.granite.ensem_pruning import (
        Kappa_Pruning_kuncheva, Kappa_Pruning_zhoubimu)
    _, tr_Pk, tr_seq_k = Kappa_Pruning_kuncheva(
        tr_y, tr_yt, nb_cls, nb_pru)
    _, bi_Pk, bi_seq_k = Kappa_Pruning_kuncheva(
        bi_y, bi_yt, nb_cls, nb_pru)
    _, mu_Pk, mu_seq_k = Kappa_Pruning_kuncheva(
        mu_y, mu_yt, nb_cls, nb_pru)
    _, tr_Pz, tr_seq_z = Kappa_Pruning_zhoubimu(
        tr_y, tr_yt, nb_cls, nb_pru)
    _, bi_Pz, bi_seq_z = Kappa_Pruning_zhoubimu(
        bi_y, bi_yt, nb_cls, nb_pru)
    _, mu_Pz, mu_seq_z = Kappa_Pruning_zhoubimu(
        mu_y, mu_yt, nb_cls, nb_pru)

    assert sum(tr_Pk) == len(
        tr_seq_k) == sum(tr_Pz) == len(tr_seq_z) == nb_pru
    assert sum(bi_Pk) == len(
        bi_seq_k) == sum(bi_Pz) == len(bi_seq_z) == nb_pru
    assert sum(mu_Pk) == len(
        mu_seq_k) == sum(mu_Pz) == len(mu_seq_z) == nb_pru
    assert np.all(np.equal(tr_seq_k, bi_seq_k))
    assert np.all(np.equal(tr_seq_z, bi_seq_z))

    assert all(isinstance(i, int) for i in mu_seq_k)
    assert all(isinstance(i, int) for i in mu_seq_z)
    assert len(set(map(id, mu_seq_k))) == nb_pru
    assert len(set(map(id, mu_seq_z))) == nb_pru


# ----------------------------------
# martine2006pruning


def test_orientation_ordering():
    from pyfair.granite.ensem_pruning import (
        _angle, _signature_vector, _average_signature_vector,
        _reference_vector, Orientation_Ordering_Pruning)
    tr_ans = _angle(tr_ha, tr_hb)
    bi_ans = _angle(bi_ha, bi_hb)
    mu_ans = _angle(mu_ha, mu_hb)
    assert 0 <= tr_ans <= np.pi and isinstance(tr_ans, float)
    assert 0 <= bi_ans <= np.pi and isinstance(bi_ans, float)
    assert 0 <= mu_ans <= np.pi and isinstance(mu_ans, float)

    tr_ans = _signature_vector(tr_yt, tr_y)
    bi_ans = _signature_vector(bi_yt, bi_y)
    mu_ans = _signature_vector(mu_yt, mu_y)
    assert np.all(np.equal(tr_ans, bi_ans))
    assert np.all(np.abs(tr_ans) <= 1)
    assert np.all(np.abs(bi_ans) <= 1)
    assert np.all(np.abs(mu_ans) <= 1)

    tr_ans, tr_res = _average_signature_vector(tr_yt, tr_y)
    bi_ans, bi_res = _average_signature_vector(bi_yt, bi_y)
    mu_ans, mu_res = _average_signature_vector(mu_yt, mu_y)
    assert np.all(np.equal(tr_ans, bi_ans))
    assert np.all(np.equal(tr_res, bi_res))
    tr_ans = np.array(tr_ans)
    bi_ans = np.array(bi_ans)
    mu_ans = np.array(mu_ans)
    assert np.all(-1 <= tr_ans) and np.all(tr_ans <= 1)
    assert np.all(-1 <= bi_ans) and np.all(bi_ans <= 1)
    assert np.all(-1 <= mu_ans) and np.all(mu_ans <= 1)

    tr_ans, tr_res = _reference_vector(nb_inst, tr_ans.tolist())
    bi_ans, bi_res = _reference_vector(nb_inst, bi_ans.tolist())
    mu_ans, mu_res = _reference_vector(nb_inst, mu_ans.tolist())
    assert np.all(np.equal(tr_ans, bi_ans))
    assert np.abs(tr_res) <= np.pi
    assert np.abs(bi_res) <= np.pi
    assert np.abs(mu_res) <= np.pi
    assert tr_res == bi_res

    tr_yo, tr_P, tr_seq, tr_fg = Orientation_Ordering_Pruning(
        tr_y, tr_yt)
    bi_yo, bi_P, bi_seq, bi_fg = Orientation_Ordering_Pruning(
        bi_y, bi_yt)
    mu_yo, mu_P, mu_seq, mu_fg = Orientation_Ordering_Pruning(
        mu_y, mu_yt)
    assert 1 <= sum(tr_P) == len(
        tr_seq) == sum(bi_P) == len(bi_seq) < nb_cls
    assert 1 <= sum(mu_P) == len(mu_seq) < nb_cls
    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_seq, bi_seq))
    assert len(set(map(id, [tr_yo, bi_yo, mu_yo]))) == 3
    assert len(set(map(id, [tr_P, bi_P, mu_P]))) == 3
    assert len(set(map(id, [tr_seq, bi_seq, mu_seq]))) == 3
    assert -np.pi <= tr_fg == bi_fg <= np.pi
    assert np.abs(mu_fg) <= np.pi
    assert 0 <= tr_fg <= bi_fg <= np.pi and 0 <= mu_fg <= np.pi


# ----------------------------------
# martinez2009analysis


def test_error_reduce():
    from pyfair.granite.ensem_pruning import Reduce_Error_Pruning
    _, tr_P, tr_seq = Reduce_Error_Pruning(
        tr_y, tr_yt, nb_cls, nb_pru)
    _, bi_P, bi_seq = Reduce_Error_Pruning(
        bi_y, bi_yt, nb_cls, nb_pru)
    _, mu_P, mu_seq = Reduce_Error_Pruning(
        mu_y, mu_yt, nb_cls, nb_pru)
    assert sum(tr_P) == len(tr_seq) == sum(
        bi_P) == len(bi_seq) == nb_pru
    assert sum(mu_P) == len(mu_seq) == nb_pru
    assert np.all(np.equal(tr_seq, bi_seq)) and np.all(
        np.equal(tr_P, bi_P))
    assert len(set(map(id, [tr_P, bi_P, mu_P]))) == 3
    assert len(set(map(id, [tr_seq, bi_seq, mu_seq]))) == 3


def test_complementary():
    from pyfair.granite.ensem_pruning import \
        Complementarity_Measure_Pruning as CMPruning
    _, tr_P, tr_seq = CMPruning(tr_y, tr_yt, nb_cls, nb_pru)
    _, bi_P, bi_seq = CMPruning(bi_y, bi_yt, nb_cls, nb_pru)
    _, mu_P, mu_seq = CMPruning(mu_y, mu_yt, nb_cls, nb_pru)
    assert sum(tr_P) == len(tr_seq) == sum(
        bi_P) == len(bi_seq) == nb_pru
    assert sum(mu_P) == len(mu_seq) == nb_pru
    assert np.all(np.equal(tr_seq, bi_seq)) and np.all(
        np.equal(tr_P, bi_P))
    assert len(set(map(id, [tr_P, bi_P, mu_P]))) == 3
    assert len(set(map(id, [tr_seq, bi_seq, mu_seq]))) == 3


# ----------------------------------
# tsoumakas2009ensemble


# ----------------------------------
# indyk2014composable
# aghamolaei2015diversity, abbassi2013diversity


def test_GMA_diversity():
    from pyfair.granite.ensem_pruning import (
        _GMM_Kappa_sum, GMM_Algorithm)
    tr_ans = _GMM_Kappa_sum(tr_yt[0], tr_yt[1:], tr_y)
    bi_ans = _GMM_Kappa_sum(bi_yt[0], bi_yt[1:], bi_y)
    mu_ans = _GMM_Kappa_sum(mu_yt[0], mu_yt[1:], mu_y)

    assert tr_ans == bi_ans
    assert all(isinstance(i, float) for i in [tr_ans, bi_ans, mu_ans])
    assert np.abs(tr_ans) <= nb_cls - 1
    assert np.abs(bi_ans) <= nb_cls - 1
    assert np.abs(mu_ans) <= nb_cls - 1

    tr_yo, tr_P, tr_seq = GMM_Algorithm(tr_y, tr_yt, nb_cls, nb_pru)
    bi_yo, bi_P, bi_seq = GMM_Algorithm(tr_y, tr_yt, nb_cls, nb_pru)
    assert id(tr_yo) != id(bi_yo)
    assert id(tr_P) != id(bi_P)
    assert id(tr_seq) != id(bi_seq)

    bi_yo, bi_P, bi_seq = GMM_Algorithm(bi_y, bi_yt, nb_cls, nb_pru)
    mu_yo, mu_P, mu_seq = GMM_Algorithm(mu_y, mu_yt, nb_cls, nb_pru)
    assert sum(tr_P) == len(tr_seq) == sum(bi_P) == len(bi_seq) == nb_pru
    assert sum(mu_P) == len(mu_seq) == nb_pru

    # NOT TRUE# assert np.all(np.equal(tr_P, bi_P))
    # NOT TRUE# assert np.all(np.equal(tr_seq, bi_seq))
    assert len(set(map(id, [tr_yo, bi_yo, mu_yo]))) == 3
    assert len(set(map(id, [tr_P, bi_P, mu_P]))) == 3
    assert len(set(map(id, [tr_seq, bi_seq, mu_seq]))) == 3


def test_LCS_diversity():
    from pyfair.granite.ensem_pruning import (
        _LocalSearch_kappa_sum,
        _LCS_sub_get_index, _LCS_sub_idx_renew,
        Local_Search)  # ,_LCS_sub_get_index_alt)
    tr_ans = _LocalSearch_kappa_sum(tr_yt, tr_y)
    bi_ans = _LocalSearch_kappa_sum(bi_yt, bi_y)
    mu_ans = _LocalSearch_kappa_sum(mu_yt, mu_y)

    assert tr_ans == bi_ans
    assert all(isinstance(i, float) for i in [tr_ans, bi_ans, mu_ans])
    assert np.abs(tr_ans) <= 1  # nb_cls
    assert np.abs(bi_ans) <= 1  # nb_cls
    assert np.abs(mu_ans) <= 1  # nb_cls

    row, col = 0, 1
    idx4, idx2_A = _LCS_sub_get_index(nb_cls, nb_pru, row, col)
    temp, idx2_B = _LCS_sub_get_index(nb_cls, nb_pru, row, col)
    idx2_A = idx2_A[: nb_pru]
    idx2_B = idx2_B[: nb_pru]
    assert id(idx4) != id(temp)
    assert len(set(idx4 + [row, col])) == nb_pru
    assert 0 <= nb_pru - len(idx4) <= 2

    P = np.random.randint(2, size=nb_cls, dtype='bool')
    epsilon = 1e-3  # 1e-6
    S_within = np.where(P)[0].tolist()
    S_without = np.where(np.logical_not(P))[0].tolist()
    p, q, T_within, T_without, flag = _LCS_sub_idx_renew(
        tr_y, np.array(tr_yt), nb_pru, epsilon,
        S_within, S_without)

    if flag:
        assert p in S_within and p not in T_within
        assert q in S_without and q not in T_without
        assert len(S_within) == len(T_within)
        assert len(S_without) == len(T_without)
        assert len(T_within) + len(T_without) == nb_cls
    else:
        assert np.all(np.equal(S_within, T_within))
        assert np.all(np.equal(S_without, T_without))

    p, q, R_within, R_without, flag = _LCS_sub_idx_renew(
        bi_y, np.array(bi_yt), nb_pru, epsilon,
        S_within, S_without)
    assert np.all(np.equal(T_within, R_within))
    assert np.all(np.equal(T_without, R_without))
    p, q, R_within, R_without, flag = _LCS_sub_idx_renew(
        mu_y, np.array(mu_yt), nb_pru, epsilon,
        S_within, S_without)
    assert len(R_within) + len(R_without) == nb_cls

    tr_yo, tr_P, tr_seq = Local_Search(
        tr_y, tr_yt, nb_cls, nb_pru, epsilon)
    bi_yo, bi_P, bi_seq = Local_Search(
        tr_y, tr_yt, nb_cls, nb_pru, epsilon)
    assert id(tr_yo) != id(bi_yo)
    assert id(tr_P) != id(bi_P)
    assert id(tr_seq) != id(bi_seq)

    bi_yo, bi_P, bi_seq = Local_Search(
        bi_y, bi_yt, nb_cls, nb_pru, epsilon)
    mu_yo, mu_P, mu_seq = Local_Search(
        mu_y, mu_yt, nb_cls, nb_pru, epsilon)
    assert sum(tr_P) == len(tr_seq) == nb_pru
    assert sum(bi_P) == len(bi_seq) == nb_pru
    assert sum(mu_P) == len(mu_seq) == nb_pru

    assert len(set(map(id, [tr_yo, bi_yo, mu_yo]))) == 3
    assert len(set(map(id, [tr_P, bi_P, mu_P]))) == 3
    assert len(set(map(id, [tr_seq, bi_seq, mu_seq]))) == 3


# ----------------------------------
# li2012diversity


def test_DREP_binary():
    from pyfair.granite.ensem_pruning import (
        _DREP_fxH, _DREP_diff, _DREP_sub_find_idx,
        DREP_Pruning)
    tr_ans = _DREP_fxH(tr_yt)
    bi_ans = _DREP_fxH(tr_yt)
    assert id(tr_ans) != id(bi_ans)

    tr_ans = _DREP_fxH(tr_yt[0])
    bi_ans = _DREP_fxH(tr_yt[0])
    assert id(tr_ans) != id(bi_ans)

    tr_ans = _DREP_diff(tr_ha, tr_hb)
    bi_ans = _DREP_diff(tr_ha, tr_hb)
    assert id(tr_ans) != id(bi_ans)
    assert -1 <= tr_ans == bi_ans <= 1

    rho = 0.3
    P = np.random.randint(2, size=nb_cls, dtype='bool').tolist()
    idx, flag = _DREP_sub_find_idx(tr_y, tr_yt, rho, P)
    if not flag:
        assert idx < 0
    else:
        assert 0 <= idx < nb_cls
        assert not P[idx]

    tr_yo, tr_P, tr_seq = DREP_Pruning(tr_y, tr_yt, nb_cls, rho)
    bi_yo, bi_P, bi_seq = DREP_Pruning(tr_y, tr_yt, nb_cls, rho)
    assert id(tr_yo) != id(bi_yo)
    assert id(tr_P) != id(bi_P)
    assert id(tr_seq) != id(bi_seq)

    bi_yo, bi_P, bi_seq = DREP_Pruning(bi_y, bi_yt, nb_cls, rho)
    # NOT TRUE# assert np.all(np.equal(tr_P, bi_P))
    # because `DREP_fxH` in `DREP_Pruning` involves some random
    # NOT ALWAYS TRUE# assert np.all(np.equal(tr_seq, bi_seq))
    # fg = sum(tr_P) == len(tr_seq) == sum(bi_P) == len(bi_seq)
    # if not fg: pdb.set_trace()
    assert 1 <= sum(tr_P) == len(tr_seq) < nb_cls
    assert 1 <= sum(bi_P) == len(bi_seq) < nb_cls
    # assert sum(tr_P) == len(tr_seq) == sum(bi_P) == len(bi_seq)
    # assert 1 <= len(tr_seq) == len(bi_seq) < nb_cls


def test_drep_multi_modify():
    from pyfair.granite.ensem_pruning import (
        _drep_multi_modify_diff,
        _drep_multi_modify_findidx)
    from pyfair.granite.ensem_pruning import \
        drep_multi_modify_pruning as drep_prune

    tr_ans = _drep_multi_modify_diff(tr_ha, tr_hb)
    bi_ans = _drep_multi_modify_diff(tr_ha, tr_hb)
    assert id(tr_ans) != id(bi_ans)
    bi_ans = _drep_multi_modify_diff(bi_ha, bi_hb)
    mu_ans = _drep_multi_modify_diff(mu_ha, mu_hb)
    assert -1 <= tr_ans == bi_ans <= 1 and abs(mu_ans) <= 1

    rho = 0.3
    P = np.random.randint(2, size=nb_cls, dtype='bool').tolist()
    tr_idx, tr_fg = _drep_multi_modify_findidx(tr_y, tr_yt, rho, P)
    bi_idx, bi_fg = _drep_multi_modify_findidx(tr_y, tr_yt, rho, P)
    if tr_fg:
        assert (not P[tr_idx]) and (not P[bi_idx])
        assert id(tr_idx) != id(bi_idx)
    else:
        assert id(tr_idx) == id(bi_idx)
    assert id(tr_fg) == id(bi_fg)  # ??
    # assert id(tr_fg) != id(bi_fg)  # ??
    bi_idx, bi_fg = _drep_multi_modify_findidx(bi_y, bi_yt, rho, P)
    mu_idx, mu_fg = _drep_multi_modify_findidx(mu_y, mu_yt, rho, P)
    assert tr_idx == bi_idx and tr_fg == bi_fg
    assert -1 <= mu_idx < nb_cls and mu_fg

    tr_yo, tr_P, tr_seq = drep_prune(tr_y, tr_yt, nb_cls, rho)
    bi_yo, bi_P, bi_seq = drep_prune(tr_y, tr_yt, nb_cls, rho)
    assert id(tr_yo) != id(bi_yo)
    assert id(tr_P) != id(bi_P)
    assert id(tr_seq) != id(bi_seq)

    bi_yo, bi_P, bi_seq = drep_prune(bi_y, bi_yt, nb_cls, rho)
    _, mu_P, mu_seq = drep_prune(mu_y, mu_yt, nb_cls, rho)  # mu_yo,
    assert sum(tr_P) == len(tr_seq) == sum(bi_P) == len(bi_seq)
    assert 1 <= len(tr_seq) == len(bi_seq) < nb_cls
    assert 1 <= sum(mu_P) == len(mu_seq) < nb_cls
    assert np.all(np.equal(tr_P, bi_P))
    assert np.all(np.equal(tr_seq, bi_seq))

    from pyfair.granite.ensem_pruning import DREP_Pruning
    # Notice that DREP_fxH involves random, so DREP_Pruning results changing
    # therefore the next two assert might not be true
    # but `drep_multi_modify_pruning` doesn't involve random
    _, re_P, re_seq = DREP_Pruning(tr_y, tr_yt, nb_cls, rho)
    # NOT TRUE# assert np.all(np.equal(tr_seq, re_seq))
    assert 1 <= sum(re_P) == len(re_seq) < nb_cls
    _, re_P, re_seq = DREP_Pruning(bi_y, bi_yt, nb_cls, rho)
    # NOT TRUE# assert np.all(np.equal(bi_seq, re_seq))
    assert 1 <= sum(re_P) == len(re_seq) < nb_cls


# ----------------------------------
# qian2015pareto


def test_PEP_prelim():
    from pyfair.granite.ensem_pruning import (
        _PEP_Hs_x, _PEP_f_Hs,
        _PEP_diff_hihj, _PEP_err_hi)
    tr_ans = _PEP_diff_hihj(tr_ha, tr_hb)
    bi_ans = _PEP_diff_hihj(tr_ha, tr_hb)
    assert tr_ans == bi_ans
    assert id(tr_ans) != id(bi_ans)
    assert all(isinstance(i, float) for i in [tr_ans, bi_ans])
    assert 0 <= tr_ans <= 1

    tr_ans = _PEP_err_hi(tr_y, tr_yt[-1])
    bi_ans = _PEP_err_hi(tr_y, tr_yt[-1])
    assert tr_ans == bi_ans
    assert id(tr_ans) != id(bi_ans)
    assert all(isinstance(i, float) for i in [tr_ans, bi_ans])
    assert 0 <= tr_ans <= 1

    from pyfair.granite.ensem_pruning import (
        _pep_multi_modify_diff_hihj,
        _pep_multi_modify_err_hi)
    tr_ans = _pep_multi_modify_diff_hihj(tr_ha, tr_hb)
    bi_ans = _pep_multi_modify_diff_hihj(bi_ha, bi_hb)
    mu_ans = _pep_multi_modify_diff_hihj(mu_ha, mu_hb)
    temp = _PEP_diff_hihj(tr_ha, tr_hb)
    assert 0 <= tr_ans == bi_ans == temp <= 1
    assert 0 <= mu_ans <= 1 and isinstance(mu_ans, float)
    tr_ans = _pep_multi_modify_err_hi(tr_y, tr_yt[-1])
    bi_ans = _pep_multi_modify_err_hi(bi_y, bi_yt[-1])
    mu_ans = _pep_multi_modify_err_hi(mu_y, mu_yt[-1])
    temp = _PEP_err_hi(tr_y, tr_yt[-1])
    assert 0 <= tr_ans == bi_ans == temp <= 1
    assert 0 <= mu_ans <= 1 and isinstance(mu_ans, float)

    s = np.random.randint(2, size=nb_cls, dtype='bool').tolist()
    tr_ans = _PEP_Hs_x(tr_y, tr_yt, s)
    bi_ans = _PEP_Hs_x(tr_y, tr_yt, s)
    assert id(tr_ans) != id(bi_ans)
    bi_ans = _PEP_Hs_x(bi_y, bi_yt, s)
    mu_ans = _PEP_Hs_x(mu_y, mu_yt, s)
    assert np.all(np.equal(np.unique(tr_ans), np.unique(tr_y)))
    assert np.all(np.equal(np.unique(bi_ans), np.unique(bi_y)))
    assert np.all(np.equal(np.unique(mu_ans), np.unique(mu_y)))

    tr_ans, tr_re = _PEP_f_Hs(tr_y, tr_yt, s)
    bi_ans, bi_re = _PEP_f_Hs(tr_y, tr_yt, s)
    assert id(tr_re) != id(bi_re)
    assert id(tr_ans) != id(bi_ans)
    bi_ans, bi_re = _PEP_f_Hs(bi_y, bi_yt, s)
    mu_ans, mu_re = _PEP_f_Hs(mu_y, mu_yt, s)
    assert 0 <= tr_ans == bi_ans <= 1
    assert 0 <= mu_ans <= 1
    assert np.all(np.equal(tr_re, [i * 2 - 1 for i in bi_re]))


def test_pep_OEP_SEP():
    from pyfair.granite.ensem_pruning import (
        _PEP_flipping_uniformly, PEP_SEP, PEP_OEP)
    s = np.random.randint(2, size=nb_cls, dtype='bool').tolist()
    sp = _PEP_flipping_uniformly(s)
    assert len(s) == len(sp) == nb_cls
    assert all((i == j or i == 1 - j) for i, j in zip(s, sp))
    assert id(s) != id(sp)

    rho = 0.3
    tr_yo, tr_P, tr_seq = PEP_SEP(tr_y, tr_yt, nb_cls, rho)
    bi_yo, bi_P, bi_seq = PEP_SEP(tr_y, tr_yt, nb_cls, rho)
    assert id(tr_yo) != id(bi_yo)
    assert id(tr_P) != id(bi_P)
    assert id(tr_seq) != id(bi_seq)

    bi_yo, bi_P, bi_seq = PEP_SEP(bi_y, bi_yt, nb_cls, rho)
    _, mu_P, mu_seq = PEP_SEP(mu_y, mu_yt, nb_cls, rho)  # mu_yo,
    assert 1 <= sum(tr_P) == len(tr_seq) < nb_cls
    assert 1 <= sum(bi_P) == len(bi_seq) < nb_cls
    assert 1 <= sum(mu_P) == len(mu_seq) < nb_cls
    # NOT TRUE# assert 1 <= len(tr_seq) == len(bi_seq) < nb_cls
    # because `PEP_flipping_uniformly` involves random

    tr_yo, tr_P, tr_seq = PEP_OEP(tr_y, tr_yt, nb_cls)
    bi_yo, bi_P, bi_seq = PEP_OEP(tr_y, tr_yt, nb_cls)
    assert id(tr_yo) != id(bi_yo)
    assert id(tr_P) != id(bi_P)
    assert id(tr_seq) != id(bi_seq)

    bi_yo, bi_P, bi_seq = PEP_OEP(bi_y, bi_yt, nb_cls)
    _, mu_P, mu_seq = PEP_OEP(mu_y, mu_yt, nb_cls)  # mu_yo,
    assert 1 <= sum(tr_P) == len(tr_seq) < nb_cls
    assert 1 <= sum(bi_P) == len(bi_seq) < nb_cls
    assert 1 <= sum(mu_P) == len(mu_seq) < nb_cls
    assert len(tr_seq) == len(bi_seq)
    assert np.all(np.equal(tr_seq, bi_seq))


def test_pep_dominate():
    from pyfair.granite.ensem_pruning import (
        _PEP_bi_objective,
        _PEP_weakly_dominate, _PEP_dominate)
    s = np.random.randint(2, size=nb_cls).tolist()
    if sum(s) == nb_cls:
        s[s.index(1)] = 0
    # for robustness

    tr_ans = _PEP_bi_objective(tr_y, tr_yt, s)
    bi_ans = _PEP_bi_objective(tr_y, tr_yt, s)
    assert id(tr_ans) != id(bi_ans)  # type(tr_ans) <class 'tuple'>
    assert id(tr_ans[0]) != id(bi_ans[0])  # <class 'float'>
    assert id(tr_ans[1]) == id(bi_ans[1])  # <class 'int'>

    bi_ans = _PEP_bi_objective(bi_y, bi_yt, s)
    mu_ans = _PEP_bi_objective(mu_y, mu_yt, s)
    assert np.all(np.equal(tr_ans, bi_ans))
    assert 0 <= tr_ans[0] <= 1 and 1 <= tr_ans[1] < nb_cls
    assert 0 <= bi_ans[0] <= 1 and 1 <= bi_ans[1] < nb_cls
    assert 0 <= mu_ans[0] <= 1 and 1 <= mu_ans[1] < nb_cls

    gs1, gs2 = (0.48, 3), (0.56, 3)
    assert _PEP_weakly_dominate(gs1, gs2) and _PEP_dominate(gs1, gs2)
    gs1, gs2 = (0.56, 3), (0.56, 4)
    assert _PEP_weakly_dominate(gs1, gs2) and _PEP_dominate(gs1, gs2)
    gs1, gs2 = (0.48, 3), (0.48, 3)
    assert _PEP_weakly_dominate(gs1, gs2) and not _PEP_dominate(gs1, gs2)
    gs1, gs2 = (0.48, 3), (0.47, 4)
    assert not _PEP_weakly_dominate(gs1, gs2)
    gs1, gs2 = (0.48, 4), (0.48, 3)
    assert not _PEP_weakly_dominate(gs1, gs2)


def test_pep_VDS_PEP():
    from pyfair.granite.ensem_pruning import _PEP_VDS, PEP_PEP
    s = np.random.randint(2, size=nb_cls, dtype='bool').tolist()
    tr_Q, tr_L = _PEP_VDS(tr_y, tr_yt, nb_cls, s)
    bi_Q, bi_L = _PEP_VDS(tr_y, tr_yt, nb_cls, s)
    assert id(tr_Q) != id(bi_Q) and id(tr_L) != id(bi_L)

    bi_Q, bi_L = _PEP_VDS(bi_y, bi_yt, nb_cls, s)
    mu_Q, mu_L = _PEP_VDS(mu_y, mu_yt, nb_cls, s)
    assert np.all(np.equal(tr_Q, bi_Q))
    assert np.all(np.equal(tr_L, bi_L))
    assert np.shape(mu_Q) == (nb_cls, nb_cls)
    assert len(mu_L) == nb_cls and all(isinstance(i, int) for i in mu_L)
    assert len(set(tr_L)) == len(set(bi_L)) == len(set(mu_L)) == nb_cls

    rho = 0.3
    tr_yo, tr_P, tr_seq = PEP_PEP(tr_y, tr_yt, nb_cls, rho)
    bi_yo, bi_P, bi_seq = PEP_PEP(tr_y, tr_yt, nb_cls, rho)
    assert id(tr_yo) != id(bi_yo)
    assert id(tr_P) != id(bi_P)
    assert id(tr_seq) != id(bi_seq)

    bi_yo, bi_P, bi_seq = PEP_PEP(bi_y, bi_yt, nb_cls, rho)
    _, mu_P, mu_seq = PEP_PEP(mu_y, mu_yt, nb_cls, rho)  # mu_yo,
    assert 1 <= sum(tr_P) == len(tr_seq) < nb_cls
    assert 1 <= sum(bi_P) == len(bi_seq) < nb_cls
    assert 1 <= sum(mu_P) == len(mu_seq) < nb_cls
    # PEP involves random so it's possible tr_seq,bi_seq are different


def test_pep_refine():
    from pyfair.granite.ensem_pruning import (
        _pep_pep_split_up_nexists,
        _pep_pep_refresh_weakly_domi,
        pep_pep_integrate)
    P = np.random.randint(2, size=(nb_pru, nb_cls)).tolist()
    s = np.random.randint(2, size=nb_cls).tolist()

    tr_fg, tr_gz = _pep_pep_split_up_nexists(tr_y, tr_yt, P, s)
    bi_fg, bi_gz = _pep_pep_split_up_nexists(bi_y, bi_yt, P, s)
    mu_fg, mu_gz = _pep_pep_split_up_nexists(mu_y, mu_yt, P, s)
    assert tr_fg == bi_fg and tr_gz == bi_gz
    if tr_gz:
        assert tr_fg and tr_gz in P
        assert bi_fg and bi_gz in P
    else:
        assert (not tr_gz) and (not bi_gz)
    if not mu_gz:
        assert not mu_fg
    else:
        assert mu_fg and mu_gz in P
    assert id(tr_gz) != id(bi_gz)
    assert id(tr_gz) != id(mu_gz)
    assert id(bi_gz) != id(mu_gz)

    tr_P, tr_ans = _pep_pep_refresh_weakly_domi(tr_y, tr_yt, P, s)
    bi_P, bi_ans = _pep_pep_refresh_weakly_domi(bi_y, bi_yt, P, s)
    mu_P, mu_ans = _pep_pep_refresh_weakly_domi(mu_y, mu_yt, P, s)
    assert np.all(np.equal(tr_ans, bi_ans))
    assert len(tr_P) == len(tr_ans) + 1
    assert len(bi_P) == len(bi_ans) + 1
    assert len(mu_P) == len(mu_ans) + 1
    assert id(tr_P) != id(bi_P) and id(tr_ans) != id(bi_ans)
    assert id(tr_P) != id(mu_P) and id(tr_ans) != id(mu_ans)
    assert id(bi_P) != id(mu_P) and id(bi_ans) != id(mu_ans)

    rho = 0.3
    tr_yo, tr_P, tr_seq = pep_pep_integrate(tr_y, tr_yt, nb_cls, rho)
    bi_yo, bi_P, bi_seq = pep_pep_integrate(bi_y, bi_yt, nb_cls, rho)
    _, mu_P, mu_seq = pep_pep_integrate(mu_y, mu_yt, nb_cls, rho)  # mu_yo,
    assert id(tr_yo) != id(bi_yo)
    assert id(tr_P) != id(bi_P)
    assert id(tr_seq) != id(bi_seq)
    assert 1 <= sum(tr_P) == len(tr_seq) < nb_cls
    assert 1 <= sum(bi_P) == len(bi_seq) < nb_cls
    assert 1 <= sum(mu_P) == len(mu_seq) < nb_cls


def test_pep_PEP_modify():
    from pyfair.granite.ensem_pruning import (
        PEP_PEP_modify,
        # pep_pep_integrate_modify,
        pep_pep_re_modify)
    rho = 0.3

    tr_yo, tr_P, tr_seq = PEP_PEP_modify(tr_y, tr_yt, nb_cls, rho)
    bi_yo, bi_P, bi_seq = PEP_PEP_modify(bi_y, bi_yt, nb_cls, rho)
    _, mu_P, mu_seq = PEP_PEP_modify(mu_y, mu_yt, nb_cls, rho)  # mu_yo,
    assert id(tr_yo) != id(bi_yo)
    assert id(tr_P) != id(bi_P)
    assert id(tr_seq) != id(bi_seq)
    assert 1 <= sum(tr_P) == len(tr_seq) < nb_cls
    assert 1 <= sum(bi_P) == len(bi_seq) < nb_cls
    assert 1 <= sum(mu_P) == len(mu_seq) < nb_cls

    tr_yo, tr_P, tr_seq = pep_pep_re_modify(tr_y, tr_yt, nb_cls, rho)
    bi_yo, bi_P, bi_seq = pep_pep_re_modify(bi_y, bi_yt, nb_cls, rho)
    _, mu_P, mu_seq = pep_pep_re_modify(mu_y, mu_yt, nb_cls, rho)  # mu_yo,
    assert id(tr_yo) != id(bi_yo)
    assert id(tr_P) != id(bi_P)
    assert id(tr_seq) != id(bi_seq)
    assert 1 <= sum(tr_P) == len(tr_seq) < nb_cls
    assert 1 <= sum(bi_P) == len(bi_seq) < nb_cls
    assert 1 <= sum(mu_P) == len(mu_seq) < nb_cls


# ----------------------------------
# contrastive

# Cannot fix the size of pruned sub-ensemble:
#
#   Orientation_Ordering_Pruning
#   DREP_Pruning, drep_multi_modify_pruning
#   PEP_SEP, PEP_OEP, PEP_PEP,
#   pep_pep_integrate
#


def test_contrastive():
    from pyfair.granite.ensem_pruning import contrastive_pruning_methods
    from pyfair.facil.utils_remark import AVAILABLE_NAME_PRUNE

    epsilon, rho = 1e-3, 0.3
    for name_pru in AVAILABLE_NAME_PRUNE:
        tr_yo, tr_P, tr_seq, tr_fg = contrastive_pruning_methods(
            name_pru, nb_cls, nb_pru, tr_y, tr_yt, epsilon, rho)
        bi_yo, bi_P, bi_seq, bi_fg = contrastive_pruning_methods(
            name_pru, nb_cls, nb_pru, tr_y, tr_yt, epsilon, rho)
        assert id(tr_yo) != id(bi_yo)
        assert id(tr_P) != id(bi_P)
        assert id(tr_seq) != id(bi_seq)

        if name_pru == "DREP":
            bi_yo, bi_P, bi_seq, bi_fg = contrastive_pruning_methods(
                name_pru, nb_cls, nb_pru, bi_y, bi_yt, epsilon, rho)
            assert 1 <= sum(tr_P) == len(tr_seq) < nb_cls
            assert 1 <= sum(bi_P) == len(bi_seq) < nb_cls
            continue

        bi_yo, bi_P, bi_seq, bi_fg = contrastive_pruning_methods(
            name_pru, nb_cls, nb_pru, bi_y, bi_yt, epsilon, rho)
        _, mu_P, mu_seq, mu_fg = contrastive_pruning_methods(
            name_pru, nb_cls, nb_pru, mu_y, mu_yt, epsilon, rho)  # mu_yo,
        assert 1 <= sum(tr_P) == len(tr_seq) < nb_cls
        assert 1 <= sum(bi_P) == len(bi_seq) < nb_cls
        assert 1 <= sum(mu_P) == len(mu_seq) < nb_cls

        if name_pru == "OO":
            assert 0 <= tr_fg <= np.pi
            assert 0 <= bi_fg <= np.pi
            assert 0 <= mu_fg <= np.pi

        if name_pru in ["ES", "KL", "KL+", "KPk", "KPz",
                        "RE", "CM", "GMA", "LCS"]:
            assert len(tr_seq) == len(bi_seq) == len(mu_seq) == nb_pru
        elif name_pru in ["OO", "DREP", "SEP", "OEP", "PEP",
                          "PEP+", "pepre", "pepr+", "drepm"]:
            pass


def test_compared_utus():
    from pyfair.granite.ensem_pruning import \
        contrastive_pruning_according_validation
    from pyfair.facil.utils_remark import AVAILABLE_NAME_PRUNE
    from sklearn import tree
    epsilon, rho = 1e-3, 0.3
    nb_labl = 2

    y_trn, y_insp, coef = synthetic_set(nb_labl, nb_inst, nb_cls)
    y_val, y_cast, _ = synthetic_set(nb_labl, nb_inst, nb_cls)
    _, y_pred, _ = synthetic_set(nb_labl, nb_inst, nb_cls)  # y_tst,
    clfs = [tree.DecisionTreeClassifier() for _ in range(nb_cls)]

    for name_pru in AVAILABLE_NAME_PRUNE:
        (opt_coef, opt_clfs, _, _, _, _, _, P, seq,
         # ys_insp,ys_cast,ys_pred,ut,us,_,_,flag
         _) = contrastive_pruning_according_validation(
            name_pru, nb_cls, nb_pru, y_val, y_cast, epsilon, rho,
            y_insp, y_pred, coef, clfs)
        assert len(opt_coef) == len(opt_clfs) == sum(P) == len(seq)
        assert 1 <= len(seq) < nb_cls

        (opt_coef, opt_clfs, _, _, _, _, _, P, seq,
         # _, ys_insp, ys_pred,ut,us,_,_,flag
         _) = contrastive_pruning_according_validation(
            name_pru, nb_cls, nb_pru, y_trn, y_insp, epsilon, rho,
            [], y_pred, coef, clfs)
        # ys_cast = []
        assert len(opt_coef) == len(opt_clfs) == sum(P) == len(seq)
        assert 1 <= len(seq) < nb_cls


# ----------------------------------
