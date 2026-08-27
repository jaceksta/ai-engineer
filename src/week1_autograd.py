"""Week 1 -- Tensors and gradients.  *** YOU IMPLEMENT THIS FILE. ***

Goal: make backpropagation stop being magic. You will implement gradient
descent twice for the same problem (pure NumPy with hand-derived gradients,
then PyTorch tensors + autograd), and then build a tiny scalar autograd
engine from scratch.

Rules for this week:
    * No ``nn.Module``, no optimizer class, no ``nn.functional``.
    * The parameter update is written by hand: ``w = w - lr * grad``.
    * For the NumPy version you derive the gradients on paper first.

Every function/class below raises ``NotImplementedError``. Replace the bodies.
The tests in ``tests/test_week1.py`` are the specification -- read them.

Read ``specs/week1.md`` for the full task and the "you're stuck if" section.
"""

from __future__ import annotations

import numpy as np

# ===========================================================================
# Part 1 -- Linear regression in pure NumPy, hand-derived gradients
# ===========================================================================


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error: ``mean((y_pred - y_true) ** 2)``.

    Parameters
    ----------
    y_pred, y_true : np.ndarray, shape (n_samples,)

    Returns
    -------
    float -- the scalar loss.
    """
    raise NotImplementedError


def analytic_grad_linear(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float
) -> tuple[np.ndarray, float]:
    """Gradient of the MSE loss of ``y_hat = X @ w + b`` w.r.t. ``w`` and ``b``.

    You derive this by hand (chain rule on ``mean((X @ w + b - y) ** 2)``).

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    w : np.ndarray, shape (n_features,)
    b : float

    Returns
    -------
    (grad_w, grad_b) : (np.ndarray shape (n_features,), float)
        The partial derivatives. ``tests/test_week1.py`` checks these against
        ``torch.autograd`` to a tight tolerance.
    """
    raise NotImplementedError


def gradient_descent_linear(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float,
    n_steps: int,
    w_init: np.ndarray | None = None,
    b_init: float = 0.0,
) -> tuple[np.ndarray, float, list[float]]:
    """Fit ``y ~ X @ w + b`` by full-batch gradient descent (NumPy only).

    Use ``analytic_grad_linear`` for the gradients and the hand-written update
    ``w -= lr * grad_w`` / ``b -= lr * grad_b``. Record the loss once per step.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    lr : float
        Learning rate.
    n_steps : int
        Number of full-batch updates.
    w_init : np.ndarray or None
        Starting weights; default to zeros of shape (n_features,).
    b_init : float
        Starting bias.

    Returns
    -------
    (w, b, loss_history) where ``loss_history`` has length ``n_steps`` and is
    monotonically non-increasing for a small enough ``lr``.
    """
    raise NotImplementedError


# ===========================================================================
# Part 2 -- Polynomial regression
# ===========================================================================


def polynomial_features(x: np.ndarray, degree: int) -> np.ndarray:
    """Design matrix ``[x**1, x**2, ..., x**degree]`` (no bias column).

    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
    degree : int, >= 1

    Returns
    -------
    np.ndarray, shape (n_samples, degree)

    Note: feature scaling matters a lot here -- high powers of x explode.
    The spec discusses this.
    """
    raise NotImplementedError


def gradient_descent_polynomial(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: int,
    lr: float,
    n_steps: int,
) -> tuple[np.ndarray, float, list[float]]:
    """Fit a degree-``degree`` polynomial by gradient descent (NumPy only).

    Build features with ``polynomial_features`` and reuse the linear machinery.

    Returns
    -------
    (coeffs, bias, loss_history) where ``coeffs`` has shape ``(degree,)``.
    """
    raise NotImplementedError


# ===========================================================================
# Part 3 -- The same fits with PyTorch autograd (still a hand-written update)
# ===========================================================================


def gradient_descent_linear_torch(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float,
    n_steps: int,
) -> tuple[np.ndarray, float, list[float]]:
    """Same as ``gradient_descent_linear`` but gradients come from autograd.

    Create ``w`` and ``b`` as tensors with ``requires_grad=True``, compute the
    loss, call ``.backward()``, then update inside ``torch.no_grad()`` and
    zero the grads. No optimizer, no ``nn``.

    Returns
    -------
    (w, b, loss_history) with ``w`` as a NumPy array and ``b`` a float, so the
    result is directly comparable to the NumPy version under a fixed seed.
    """
    raise NotImplementedError


def gradient_descent_polynomial_torch(
    x: np.ndarray,
    y: np.ndarray,
    *,
    degree: int,
    lr: float,
    n_steps: int,
) -> tuple[np.ndarray, float, list[float]]:
    """Polynomial fit with autograd. See ``gradient_descent_linear_torch``."""
    raise NotImplementedError


# ===========================================================================
# Part 4 -- A minimal scalar autograd engine (micrograd equivalent)
# ===========================================================================


class Value:
    """A scalar node in an autograd graph.

    Supports ``+``, ``*``, ``**`` (integer/float exponent), and at least one
    nonlinearity (``tanh`` or ``relu``). ``backward()`` populates ``.grad`` on
    this node and every node it depends on, using reverse-mode autodiff over a
    topological order of the graph.

    Required behaviour (see ``tests/test_week1.py``):
        * ``Value(2.0).data == 2.0`` and ``.grad`` starts at ``0.0``.
        * Operations with plain ints/floats work: ``Value(2.0) + 1``, ``2 * a``.
        * After ``y.backward()``, ``x.grad`` matches ``torch.autograd`` for the
          same expression, on several small graphs.
        * Gradients *accumulate*: calling ``backward()`` twice without zeroing
          doubles ``.grad`` (a node used in two places sums its contributions).

    Suggested internal state:
        data : float
        grad : float                      -- initialised to 0.0
        _backward : Callable[[], None]     -- local grad rule, default no-op
        _prev : set[Value]                 -- direct inputs
        _op : str                         -- label, for debugging/repr only
    """

    def __init__(self, data: float, _children: tuple = (), _op: str = "") -> None:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        raise NotImplementedError

    def __add__(self, other: Value | float) -> Value:
        raise NotImplementedError

    def __mul__(self, other: Value | float) -> Value:
        raise NotImplementedError

    def __pow__(self, other: float) -> Value:
        raise NotImplementedError

    def __radd__(self, other: Value | float) -> Value:
        raise NotImplementedError

    def __rmul__(self, other: Value | float) -> Value:
        raise NotImplementedError

    def __neg__(self) -> Value:
        raise NotImplementedError

    def __sub__(self, other: Value | float) -> Value:
        raise NotImplementedError

    def __truediv__(self, other: Value | float) -> Value:
        raise NotImplementedError

    def tanh(self) -> Value:
        raise NotImplementedError

    def relu(self) -> Value:
        raise NotImplementedError

    def backward(self) -> None:
        """Seed ``self.grad = 1.0`` and propagate to all upstream nodes.

        Does *not* zero existing grads first -- accumulation is intentional and
        tested. Build the topological order, then apply each node's local
        ``_backward`` in reverse.
        """
        raise NotImplementedError
