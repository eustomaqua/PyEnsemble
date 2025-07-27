# coding: utf-8


import numpy as np
from pyfair.facil.utils_const import check_equal, synthetic_dat

nb_inst, nb_feat, nb_lbl = 1210, 7, 3
nb_bin = 4
X_trn, y_trn = synthetic_dat(nb_lbl, nb_inst, nb_feat)


def test_ent_convert():
    from pyfair.marble.data_entropy import binsMDL, prob, jointProb
    data = binsMDL(X_trn, nb_bin=nb_bin)
    assert len(np.unique(data)) == nb_bin
    assert np.shape(data) == np.shape(X_trn)

    data = np.array(data)
    X = data[:, 0].tolist()
    px, vX = prob(X)
    assert all([i in range(nb_bin) for i in set(vX)])
    assert check_equal(sum(px), 1.)  # sum(px) == 1.

    Y = data[:, 1].tolist()
    pxy, vX, vY = jointProb(X, Y)
    assert all([(i in range(nb_bin)) for i in set(vX)])
    assert all([(j in range(nb_bin)) for j in set(vY)])
    assert 1 <= np.shape(pxy)[0] == len(vX) <= nb_bin
    assert 1 <= np.shape(pxy)[1] == len(vY) <= nb_bin
    assert check_equal(np.sum(pxy), 1.)  # np.sum(pxy)==1.

    temn = binsMDL(X_trn, nb_bin=nb_bin)
    assert id(data) != id(temn)
    assert np.all(np.equal(data, temn))
    temn = 5
    for i in range(temn):
        X = data[:, i].tolist()
        px, vX = prob(X)
        py, vY = prob(X)
        assert id(px) != id(py) and id(vX) != id(vY)
        assert np.all(np.equal(px, py))
        assert np.all(np.equal(vX, vY))

    for i in range(temn - 1):
        X = data[:, i].tolist()
        for j in range(i + 1, temn):
            Y = data[:, j].tolist()
            pxy, vX, vY = jointProb(X, Y)
            pab, vA, vB = jointProb(X, Y)
            assert id(pxy) != id(pab)
            assert id(vX) != id(vA) and id(vY) != id(vB)
            assert np.all(np.equal(pxy, pab))
            assert np.all(np.equal(vX, vA))
            assert np.all(np.equal(vY, vB))
    return


def test_data_entropy():
    # from fairml.widget.data_entropy import H, H1, H2
    # from fairml.widget.data_entropy import I, MI, VI
    # from fairml.widget.data_entropy import DIST, DIV1, DIV2
    from pyfair.marble.data_entropy import (
        H, H1, H2, I, MI, VI, DIST, DIV1, DIV2)

    nb_spl, nb_lbl = 223, 4
    X = np.random.randint(nb_lbl, size=nb_spl).tolist()
    Y = np.random.randint(nb_lbl, size=nb_spl).tolist()
    L = np.random.randint(nb_lbl, size=nb_spl).tolist()
    S = np.random.randint(nb_lbl, size=(nb_spl, nb_feat)).tolist()

    def subr_ent_HI(X, Y):
        for p in np.arange(0, 1.1, 0.1):
            assert 0. <= H(p) <= 1.
        assert H1(X) >= 0. and H1(Y) >= 0.
        assert H2(X, Y) >= 0.
        # assert H2(X, Y) == H2(Y, X)
        assert H(0.) == 0. and H(1e-16) > 0.
        assert H(1.) == 0. and H(1 + 1e-8) < 0.
        assert check_equal(H2(X, Y), H2(Y, X))

        ans1, ans2 = H1(X), H1(X)
        assert id(ans1) != id(ans2) and ans1 == ans2
        ans1, ans2 = H2(X, Y), H2(X, Y)
        assert id(ans1) != id(ans2) and ans1 == ans2
        assert check_equal(I(X, Y), I(Y, X))
        assert check_equal(MI(X, Y), MI(Y, X))
        assert check_equal(VI(X, Y), VI(Y, X))

        ans1, ans2 = I(X, Y), I(X, Y)
        assert id(ans1) != id(ans2) and ans1 == ans2
        ans1, ans2 = MI(X, Y), MI(X, Y)
        assert id(ans1) != id(ans2) and ans1 == ans2
        ans1, ans2 = VI(X, Y), VI(X, Y)
        assert id(ans1) != id(ans2) and ans1 == ans2

    def subr_ent_DIST(X, Y, L, S):
        max_value = nb_feat * nb_feat / 2.
        for lam in np.arange(0, 1.1, 0.1):
            assert 0 <= DIST(X, Y, L, lam) <= 1
            ans1 = DIV1(S, L, lam)
            ans2 = DIV2(S, L, lam)
            assert check_equal(ans1, ans2, 1e-13)
            assert 0 <= ans1 <= max_value
            assert 0 <= ans2 <= max_value
        lam = 0.5

        ans1 = DIST(X, Y, L, lam)
        ans2 = DIST(X, Y, L, lam)
        assert id(ans1) != id(ans2) and ans1 == ans2
        ans1 = DIV1(S, L, lam)
        ans2 = DIV1(S, L, lam)
        assert id(ans1) != id(ans2) and ans1 == ans2
        ans1 = DIV2(S, L, lam)
        ans2 = DIV2(S, L, lam)
        assert id(ans1) != id(ans2) and ans1 == ans2

    subr_ent_HI(X, Y)
    subr_ent_DIST(X, Y, L, S)
    return


def test_distributed():
    from pyfair.marble.data_entropy import (
        _dist_sum, _arg_max_p, Greedy, _choose_proper_platform,
        _find_idx_in_sub, _randomly_partition, DDisMI)
    nb_spl, nb_lbl = 121, 5

    T = np.random.randint(nb_lbl, size=(nb_spl, nb_feat)).tolist()
    L = np.random.randint(nb_lbl, size=nb_spl).tolist()
    p = np.random.randint(nb_lbl, size=nb_spl).tolist()
    S = np.random.randint(2, size=nb_feat, dtype='bool').tolist()
    N = np.random.randint(nb_lbl, size=(nb_spl, nb_feat))

    def subr_Greedy(p, T, L, S):
        for lam in np.arange(0, 1.04, 0.05):
            assert 0 <= _dist_sum(p, T, L, lam) <= nb_feat
            idx = _arg_max_p(T, S, L, lam)
            assert -1 <= idx < nb_feat
            assert (not S[idx] if idx > -1 else True)

        lam = 0.5
        ans1 = _dist_sum(p, T, L, lam)
        ans2 = _dist_sum(p, T, L, lam)
        assert id(ans1) != id(ans2) and ans1 == ans2
        ans1 = _arg_max_p(T, S, L, lam)
        ans2 = _arg_max_p(T, S, L, lam)
        assert id(ans1) != id(ans2) and ans1 == ans2
        assert (ans1 == ans2 == -1 if -1 in [ans1, ans2] else True)

        k = 3
        for lam in np.arange(0, 1.04, 0.25):
            ans1 = Greedy(T, k, L, lam)
            ans2 = Greedy(T, k, L, lam)
            assert id(ans1) != id(ans2)
            assert sum(ans1) == sum(ans2) == k
        # Notice that ans1 might not be equal to ans2
        #             due to randomness in Greedy

    def subr_DDisMI(N, L):
        for nb in (11, 17, 21, 27, 31, 37, 41):
            for pr in np.arange(0.05, 0.54, 0.05):
                k, m = _choose_proper_platform(nb, pr)
                assert 1 <= k <= int(nb * pr) + 1
                assert m >= 1 and nb / m >= k
        for n in (11, 17, 21, 27, 31, 37, 41):
            for m in (1, 2, 3, 4, 5, 6, 7):
                idx = _randomly_partition(n, m)
                tmp = n // m
                for i in range(m):
                    alt = np.sum(np.equal(idx, i))
                    assert (alt == tmp if n % m == 0 else (
                        alt in [tmp, tmp + 1]))

        k, m, lam = 2, 3, 0.5
        Tl = np.array(_randomly_partition(nb_feat, m))
        for i in range(m):
            idx = _find_idx_in_sub(i, Tl, N, k, L, lam)
            assert all((Tl[j] == i) for j in idx)
        N = N.tolist()
        for lam in np.arange(0, 1.04, 0.25):
            ans1 = DDisMI(N, k, m, L, lam)
            ans2 = DDisMI(N, k, m, L, lam)
            assert id(ans1) != id(ans2)
            assert sum(ans1) == sum(ans2) == k
            # It's possible that ans1 != ans2 (they are not the same)

    subr_Greedy(p, T, L, S)
    subr_DDisMI(N, L)
    return


def test_data_distance():
    from pyfair.marble.data_distance import (
        # from fairml.facils.metric_dist import (
        KL_divergence, JS_divergence, f_divergence,
        Hellinger_dist_v1, Hellinger_dist_v2, Wasserstein_dis,
        Wasserstein_distance, Bhattacharyya_dist)

    _len = 11
    p = np.random.rand(_len)  # .tolist()
    q = np.random.rand(_len)  # .tolist()
    p /= np.sum(p)
    q /= np.sum(q)
    p = p.tolist()
    q = q.tolist()

    def subr_dist(p, q):
        ans_1 = KL_divergence(p, q)
        ans_2 = KL_divergence(q, p)
        assert ans_1 != ans_2
        assert 0 <= ans_1 and 0 <= ans_2

        ans_1 = JS_divergence(p, q)
        ans_2 = JS_divergence(q, p)
        assert 0 <= ans_1 == ans_2 <= 1

        ans_1 = f_divergence(p, q)
        ans_2 = f_divergence(q, p)
        assert 0 <= ans_1 and 0 <= ans_2
        assert ans_1 != ans_2

        ans_1 = Bhattacharyya_dist(p, q)
        ans_2 = Bhattacharyya_dist(q, p)
        assert ans_1 == ans_2

        ans_1 = Hellinger_dist_v1(p, q)
        ans_2 = Hellinger_dist_v1(q, p)
        res_1 = Hellinger_dist_v2(p, q)
        res_2 = Hellinger_dist_v2(q, p)
        assert ans_1 == ans_2
        assert res_1 == res_2
        assert check_equal(ans_1, res_1)

    def subr_others(p, q):
        from pyfair.marble.data_distance import (
            JS_div, _f_div, _BC_dis,
            _discrete_bar_counts, _discrete_joint_cnts)
        _mx, _n = 15, 11  # 5,11, _max,indexes

        indices = np.random.randint(_mx, size=(3, _n)).tolist()
        freq_x, freq_y = _discrete_bar_counts(indices, False)
        # assert (len(freq_x) == _mx) and (sum(freq_y) == 3 * _n)
        fg = (len(freq_x) <= _mx) and (sum(freq_y) == 3 * _n)
        assert fg  # if not fg: pdb.set_trace()
        freq_x, freq_y = _discrete_bar_counts(indices, True)
        fg = len(freq_x) <= _mx and check_equal(sum(freq_y), 1)
        assert fg  # if not fg: pdb.set_trace()

        X, Y = indices[: 2]  # .tolist()
        px, py, v = _discrete_joint_cnts(X, Y, False)
        fg = (len(v) <= _mx) and (sum(px) == sum(py) == _n)
        assert fg  # if not fg: pdb.set_trace()
        px, py, _ = _discrete_joint_cnts(X, Y, True, freq_x)
        # assert sum(px) == sum(py) == 1
        assert check_equal(1, [sum(px), sum(py)])

        ans_1 = JS_divergence(px, py)
        ans_2 = JS_divergence(py, px)
        res_1 = JS_div(X, Y, _mx)
        res_2 = JS_div(Y, X, _mx)
        # assert ans_1 == ans_2 == res_1 == res_2
        assert check_equal([ans_1, res_1, ans_2],
                           [ans_2, res_2, res_1])

        assert _f_div(0)
        assert _f_div(.1)
        assert _BC_dis(p, q) == _BC_dis(q, p)

    def subr_Wasserstein():
        P = np.asarray([[0.4, 100, 40, 22],
                        [0.3, 211, 20, 2],
                        [0.2, 32, 190, 150],
                        [0.1, 2, 100, 100]], np.float32)
        Q = np.array([[0.5, 0, 0, 0],
                      [0.3, 50, 100, 80],
                      [0.2, 255, 255, 255]], np.float32)

        P = [1, 1, 3, 0, 1, 1, 0, 2, 4, 1, 3]
        Q = [0, 1, 3, 4, 0, 3, 4, 0, 1, 3, 3][: -1]

        D = np.array([
            [0.93567741, 0.34600973, 0.27497965, 0.4246211, 0.75069193,
             0.35346927, 0.99816927, 0.71456577, 0.83063506, 0.78303612],
            [0.2807255, 0.74804571, 0.25581535, 0.75933327, 0.9266441,
             0.87676772, 0.22450438, 0.19204409, 0.284512, 0.32865576],
            [0.84234205, 0.98525291, 0.24993327, 0.09657005, 0.34258716,
             0.68564655, 0.62006413, 0.27505453, 0.25916905, 0.93834445],
            [0.65565054, 0.0015819, 0.80277026, 0.54452282, 0.21638088,
             0.89420282, 0.02075447, 0.58425187, 0.03557015, 0.71154912],
            [0.25794754, 0.67905002, 0.59239805, 0.36946834, 0.55272395,
             0.92108286, 0.29793572, 0.5181298, 0.51423737, 0.02359331],
            [0.75631957, 0.5938635, 0.18456913, 0.22295727, 0.42088002,
             0.80519667, 0.21067406, 0.01092239, 0.25789742, 0.70930335],
            [0.36418754, 0.83065765, 0.19611084, 0.22163539, 0.67689765,
             0.10083083, 0.7343346, 0.10871391, 0.71905828, 0.57137656],
            [0.34371387, 0.19048136, 0.50948028, 0.04665333, 0.81732085,
             0.05715832, 0.64291096, 0.70375603, 0.17183316, 0.74101405],
            [0.05366221, 0.33590286, 0.24301574, 0.54062827, 0.0509917,
             0.10521303, 0.82893334, 0.3896138, 0.46337714, 0.69849168],
            [0.91117572, 0.26605836, 0.79068549, 0.67219381, 0.36649096,
             0.45386944, 0.17232333, 0.77632621, 0.22291717, 0.15580118],
            [0.21171312, 0.54628479, 0.13150931, 0.94915583, 0.53609526,
             0.09711085, 0.86418277, 0.77878942, 0.66041811, 0.6650721]
        ]).tolist()

        ans_1 = Wasserstein_dis(P, Q)
        ans_2 = Wasserstein_dis(Q, P)
        assert ans_1 == ans_2

        res_1 = Wasserstein_distance(P, Q, D)
        res_2 = Wasserstein_distance(Q, P, np.transpose(D))
        assert res_1 != res_2

    def subr_Hellinger():
        P = np.asarray([0.65, 0.25, 0.07, 0.03])
        Q = np.array([0.6, 0.25, 0.1, 0.05])

        # Two ways
        h1 = 1 / np.sqrt(2) * np.linalg.norm(
            np.sqrt(P) - np.sqrt(Q))
        h2 = np.sqrt(1 - np.sum(np.sqrt(P * Q)))
        assert check_equal(h1, h2)

        res_1 = Hellinger_dist_v1(P, Q)
        res_2 = Hellinger_dist_v1(Q, P)
        assert res_1 == res_2 == h1

        res_1 = Hellinger_dist_v2(P, Q)
        res_2 = Hellinger_dist_v2(Q, P)
        assert res_1 == res_2 == h2

    subr_dist(p, q)
    subr_others(p, q)
    subr_Wasserstein()
    subr_Hellinger()
    return
