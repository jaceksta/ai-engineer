"""Specification tests for Week 1 -- tensors, gradients, and a scalar autograd
engine.

Every test here fails with ``NotImplementedError`` on a fresh clone. When a
test passes, that part of ``src/week1_autograd.py`` is correct. Do not edit the
tests to make them pass; they define the contract.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.week1_autograd import (
    Value,
    analytic_grad_linear,
    gradient_descent_linear,
    gradient_descent_linear_torch,
    gradient_descent_polynomial,
    mse_loss,
    polynomial_features,
)

RNG = np.random.default_rng(0)


# --- helpers ---------------------------------------------------------------
def _torch_linear_grads(X, y, w, b):
    """Reference dL/dw, dL/db for mean-squared-error of ``X @ w + b``."""
    Xt = torch.tensor(X, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    wt = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    bt = torch.tensor(float(b), dtype=torch.float64, requires_grad=True)
    loss = ((Xt @ wt + bt) - yt).pow(2).mean()
    loss.backward()
    return wt.grad.numpy(), float(bt.grad)


# ===========================================================================
# Part 1 -- linear regression, NumPy
# ===========================================================================
def test_mse_loss_value():
    y_pred = np.array([1.0, 2.0, 3.0])
    y_true = np.array([1.0, 0.0, 3.0])
    assert mse_loss(y_pred, y_true) == pytest.approx(4.0 / 3.0)


def test_analytic_grad_linear_matches_torch():
    X = RNG.normal(size=(64, 3))
    true_w = np.array([1.5, -2.0, 0.7])
    y = X @ true_w + 0.3 + RNG.normal(scale=0.1, size=64)

    w = np.array([0.1, 0.2, -0.1])
    b = 0.05
    grad_w, grad_b = analytic_grad_linear(X, y, w, b)
    exp_w, exp_b = _torch_linear_grads(X, y, w, b)

    assert np.asarray(grad_w).shape == (3,)
    np.testing.assert_allclose(grad_w, exp_w, rtol=1e-8, atol=1e-8)
    assert grad_b == pytest.approx(exp_b, abs=1e-8)


def test_gradient_descent_linear_recovers_params():
    X = RNG.normal(size=(200, 2))
    true_w = np.array([2.0, -3.0])
    true_b = 1.0
    y = X @ true_w + true_b

    w, b, history = gradient_descent_linear(X, y, lr=0.1, n_steps=500)

    assert len(history) == 500
    assert history[-1] < history[0]
    # non-increasing (allow tiny numerical wiggle)
    assert all(history[i + 1] <= history[i] + 1e-9 for i in range(len(history) - 1))
    np.testing.assert_allclose(w, true_w, atol=1e-2)
    assert b == pytest.approx(true_b, abs=1e-2)


def test_numpy_and_torch_linear_agree():
    X = RNG.normal(size=(128, 3))
    y = X @ np.array([0.5, -1.0, 2.0]) + 0.25

    w_np, b_np, hist_np = gradient_descent_linear(X, y, lr=0.05, n_steps=300)
    w_t, b_t, hist_t = gradient_descent_linear_torch(X, y, lr=0.05, n_steps=300)

    np.testing.assert_allclose(w_np, w_t, atol=1e-5)
    assert b_np == pytest.approx(b_t, abs=1e-5)
    np.testing.assert_allclose(hist_np, hist_t, atol=1e-5)


# ===========================================================================
# Part 2 -- polynomial regression
# ===========================================================================
def test_polynomial_features_shape_and_values():
    x = np.array([2.0, 3.0])
    feats = polynomial_features(x, degree=3)
    assert feats.shape == (2, 3)
    np.testing.assert_allclose(feats[0], [2.0, 4.0, 8.0])
    np.testing.assert_allclose(feats[1], [3.0, 9.0, 27.0])


def test_gradient_descent_polynomial_fits_cubic():
    x = np.linspace(-1.0, 1.0, 200)
    y = 1.0 + 2.0 * x - 1.5 * x**2 + 0.5 * x**3

    _, _, history = gradient_descent_polynomial(x, y, degree=3, lr=0.1, n_steps=2000)
    assert history[-1] < 1e-3


# ===========================================================================
# Part 4 -- the scalar autograd engine
# ===========================================================================
def test_value_construction():
    v = Value(2.0)
    assert v.data == pytest.approx(2.0)
    assert v.grad == pytest.approx(0.0)


def test_value_add_and_mul_grads_match_torch():
    a = Value(-4.0)
    b = Value(2.0)
    c = a * b + b
    c.backward()

    at = torch.tensor(-4.0, requires_grad=True)
    bt = torch.tensor(2.0, requires_grad=True)
    ct = at * bt + bt
    ct.backward()

    assert c.data == pytest.approx(ct.item())
    assert a.grad == pytest.approx(at.grad.item())
    assert b.grad == pytest.approx(bt.grad.item())


def test_value_pow_and_scalar_ops_match_torch():
    a = Value(3.0)
    b = Value(-1.0)
    d = a**2 + 2.0 * b - a / b
    d.backward()

    at = torch.tensor(3.0, requires_grad=True)
    bt = torch.tensor(-1.0, requires_grad=True)
    dt = at**2 + 2.0 * bt - at / bt
    dt.backward()

    assert d.data == pytest.approx(dt.item())
    assert a.grad == pytest.approx(at.grad.item())
    assert b.grad == pytest.approx(bt.grad.item())


def test_value_tanh_matches_torch():
    """The spec lets you pick tanh or relu; this test checks ``tanh``, so
    implement ``Value.tanh`` (you may implement ``relu`` as well).
    """
    x = Value(0.7)
    y = x.tanh()
    y.backward()

    xt = torch.tensor(0.7, requires_grad=True)
    yt = torch.tanh(xt)
    yt.backward()

    assert y.data == pytest.approx(yt.item())
    assert x.grad == pytest.approx(xt.grad.item())


def test_value_reused_node_sums_gradient():
    a = Value(3.0)
    b = a * a
    b.backward()
    assert a.grad == pytest.approx(6.0)


def test_value_backward_accumulates_across_calls():
    a = Value(3.0)
    b = Value(4.0)
    c = a * b
    c.backward()
    grad_after_one = a.grad
    c.backward()
    assert a.grad == pytest.approx(2 * grad_after_one)
