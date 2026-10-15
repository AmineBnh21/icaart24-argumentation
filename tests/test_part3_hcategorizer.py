"""Tests of the weighted h-categorizer on small frameworks with closed-form degrees.

Run from the repository root:  python -m pytest tests -q
"""

import math

import numpy as np
import pytest

from src.part3 import frameworks as fw
from src.part3.hcategorizer import ArgumentationFramework, h_categorizer, h_categorizer_batch

GOLDEN = (math.sqrt(5) - 1) / 2  # solution of x = 1 / (1 + x)
EPS = 1e-12


def close(x, y, tol=1e-9):
    return abs(x - y) < tol


def test_isolated_degree_is_weight():
    d = h_categorizer(fw.isolated(2), {"a": 0.3, "b": 0.8}, EPS)
    assert close(d["a"], 0.3) and close(d["b"], 0.8)


def test_single_attack():
    d = h_categorizer(fw.small_frameworks()["attack2"], {"a": 0.6, "b": 0.9}, EPS)
    assert close(d["a"], 0.6)
    assert close(d["b"], 0.9 / 1.6)


def test_mutual_attack_closed_form():
    wa, wb = 0.7, 0.4
    d = h_categorizer(fw.small_frameworks()["mutual2"], {"a": wa, "b": wb}, EPS)
    # x_a - x_b = w_a - w_b, and x_b (1 + x_a) = w_b
    delta = wa - wb
    xb = (-(1 + delta) + math.sqrt((1 + delta) ** 2 + 4 * wb)) / 2
    assert close(d["b"], xb) and close(d["a"], xb + delta)


def test_mutual_attack_unit_weights_is_golden_ratio():
    d = h_categorizer(fw.small_frameworks()["mutual2"], [1, 1], EPS)
    assert close(d["a"], GOLDEN) and close(d["b"], GOLDEN)


def test_self_attack():
    for w in (0.0, 0.25, 1.0):
        d = h_categorizer(fw.small_frameworks()["self1"], [w], EPS)
        assert close(d["a"], (-1 + math.sqrt(1 + 4 * w)) / 2)


def test_chain3():
    d = h_categorizer(fw.chain(3), {"a": 1.0, "b": 1.0, "c": 1.0}, EPS)
    assert close(d["a"], 1.0) and close(d["b"], 0.5) and close(d["c"], 1 / 1.5)


def test_cycle3_unit_weights_is_golden_ratio():
    d = h_categorizer(fw.cycle(3), [1, 1, 1], EPS)
    assert all(close(v, GOLDEN) for v in d.values())


def test_star3():
    d = h_categorizer(fw.star(3), {"a": 0.5, "b": 0.2, "c": 0.9}, EPS)
    assert close(d["c"], 0.9 / 1.7)


def test_zero_weight_gives_zero_degree():
    d = h_categorizer(fw.cycle(3), [0.0, 0.7, 0.4], EPS)
    assert d["a"] == 0.0


def test_solution_is_a_fixed_point_on_random_frameworks():
    rng = np.random.default_rng(0)
    for seed in range(20):
        af = fw.random_af(6, 0.4, seed)
        W = rng.random((50, len(af)))
        X = h_categorizer_batch(af, W, EPS)
        assert np.allclose(X, W / (1 + X @ af.matrix), atol=1e-10)
        assert np.all(X >= 0) and np.all(X <= W + 1e-15)


def test_batch_matches_single():
    af = fw.cycle(4)
    W = np.random.default_rng(1).random((10, 4))
    X = h_categorizer_batch(af, W, EPS)
    for w, x in zip(W, X):
        d = h_categorizer(af, w, EPS)
        assert np.allclose([d[a] for a in af.arguments], x, atol=1e-10)


def test_epsilon_controls_precision_and_iterations():
    af = fw.complete(4)
    w = [1.0, 0.9, 0.8, 0.7]
    reference = h_categorizer(af, w, 1e-15)
    prev_iter = 0
    for eps in (1e-2, 1e-4, 1e-6, 1e-8, 1e-10):
        d, it = h_categorizer(af, w, eps, return_iterations=True)
        assert it >= prev_iter
        prev_iter = it
        # the error is of the same order as the last step (contraction)
        assert max(abs(d[a] - reference[a]) for a in af.arguments) < 10 * eps


def test_invalid_inputs():
    af = fw.chain(2)
    with pytest.raises(ValueError):
        h_categorizer(af, [0.5, 0.5], epsilon=0)
    with pytest.raises(ValueError):
        h_categorizer(af, [1.5, 0.5])
    with pytest.raises(ValueError):
        ArgumentationFramework(["a"], [("a", "z")])


def test_parser():
    af = fw.parse("a, b, c : a>b, b>c, c>a")
    assert af.arguments == ["a", "b", "c"]
    assert af.attackers("a") == ["c"]


# ------------------------------------------------ acceptability space (Part B)

from src.part3 import space as sp  # noqa: E402


def test_inverse_map_recovers_weights():
    for af in fw.experiment_frameworks()[:10]:
        W, X, _ = sp.sample_space(af, 2000, seed=3, epsilon=1e-13)
        assert np.allclose(sp.inverse_weights(af, X), W, atol=1e-10)
        assert sp.in_space(af, X, tol=1e-9).all()


def test_points_outside_space_are_rejected():
    af = fw.small_frameworks()["attack2"]
    # x_b <= 1 / (1 + x_a): (1, 0.6) is not reachable, (1, 0.5) is.
    assert not sp.in_space(af, np.array([[1.0, 0.6]]))[0]
    assert sp.in_space(af, np.array([[1.0, 0.5]]))[0]


def test_true_volume_single_attack_is_ln2():
    _, X, _ = sp.sample_space(fw.small_frameworks()["attack2"], 200_000, seed=0)
    vol, se = sp.true_volume(fw.small_frameworks()["attack2"], X)
    assert abs(vol - math.log(2)) < 4 * se


def test_chain3_and_star3_have_same_volume():
    # both equal 3 ln 3 - 4 ln 2
    target = 3 * math.log(3) - 4 * math.log(2)
    for af in (fw.chain(3), fw.star(3)):
        _, X, _ = sp.sample_space(af, 200_000, seed=0)
        vol, se = sp.true_volume(af, X)
        assert abs(vol - target) < 4 * se


def test_exact_cell_masses_sum_to_one():
    for name in ("attack2", "mutual2", "isolated2"):
        H = sp.exact_cell_mass_2d(fw.small_frameworks()[name], bins=20)
        assert abs(H.sum() - 1) < 1e-3
