"""Week 2 -- Your first network.  *** YOU IMPLEMENT THIS FILE. ***

Goal: build and train an MLP for shot-outcome classification (goal / no goal)
from first principles, *then* earn the abstractions by refactoring to
``nn.Module`` + ``DataLoader`` + ``torch.optim`` and showing the two versions
agree given a fixed seed.

Two implementations live here:
    Part A  ``ScratchMLP`` + ``train_scratch``  -- parameters are bare tensors,
            the training loop is written out by hand (forward, loss, backward,
            manual SGD step, zero grad, evaluate). Autograd may compute the
            gradients; the optimiser may not.
    Part B  ``MLP`` (a real ``nn.Module``) + ``train_idiomatic`` -- the same
            model the idiomatic way.

Both take numeric features and integer-coded categorical features (the codes
come from ``src.data_prep.encode_categoricals``). Categoricals go through
embedding layers.

The tests in ``tests/test_week2.py`` are the specification. Read
``specs/week2.md`` for the full task.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# ===========================================================================
# Shared helpers
# ===========================================================================


def default_emb_dim(cardinality: int) -> int:
    """Rule-of-thumb embedding width for a categorical with ``cardinality``
    distinct codes (including the reserved 0 = unknown slot).

    Use: ``min(50, (cardinality + 1) // 2)`` -- capped small because our
    categoricals are tiny (body part, shot type, ...). ``tests/test_week2.py``
    checks these exact values, so implement the formula as written.
    """
    raise NotImplementedError


def bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Numerically stable binary cross-entropy from logits, written by hand.

    ``mean( softplus(logits) - targets * logits )`` (derive this from
    ``-[y log p + (1-y) log(1-p)]`` with ``p = sigmoid(logits)``).

    Parameters
    ----------
    logits, targets : torch.Tensor, shape (batch,)
        ``targets`` are 0.0 / 1.0 floats.

    Returns
    -------
    Scalar tensor. Must match ``torch.nn.functional.binary_cross_entropy_with_logits``
    to within 1e-6 (the test checks this).
    """
    raise NotImplementedError


# ===========================================================================
# Part A -- from scratch
# ===========================================================================


class ScratchMLP:
    """An MLP whose parameters are plain ``requires_grad=True`` tensors.

    NOT an ``nn.Module``. You allocate every weight and bias yourself, seed the
    RNG for reproducibility, and expose them through ``parameters()``.

    Architecture:
        * one ``embedding table`` (a raw tensor) per categorical feature
        * concatenate [numeric features, looked-up embeddings]
        * ``len(hidden_sizes)`` hidden layers with ReLU
        * a final linear layer to a single logit per row

    Parameters
    ----------
    n_numeric : int
        Number of numeric input features.
    cat_cardinalities : list[int]
        ``num_embeddings`` for each categorical feature (max code + 1).
    emb_dims : list[int]
        Embedding width per categorical feature (same length as above).
    hidden_sizes : list[int]
        Hidden layer widths.
    seed : int
        Seeds the parameter initialisation.
    """

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: list[int],
        emb_dims: list[int],
        hidden_sizes: list[int],
        seed: int = 0,
    ) -> None:
        raise NotImplementedError

    def parameters(self) -> list[torch.Tensor]:
        """Every trainable tensor (embedding tables, weights, biases)."""
        raise NotImplementedError

    def forward(self, x_numeric: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x_numeric : torch.Tensor, shape (batch, n_numeric), float
        x_categorical : torch.Tensor, shape (batch, n_categorical), long

        Returns
        -------
        logits : torch.Tensor, shape (batch,)   -- raw scores, not probabilities
        """
        raise NotImplementedError

    def __call__(self, x_numeric: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
        return self.forward(x_numeric, x_categorical)


def train_scratch(
    model: ScratchMLP,
    x_numeric: torch.Tensor,
    x_categorical: torch.Tensor,
    y: torch.Tensor,
    *,
    lr: float,
    n_steps: int,
    batch_size: int | None = None,
    seed: int = 0,
) -> dict:
    """Hand-written training loop for ``ScratchMLP``.

    Each step: forward -> ``bce_with_logits`` -> ``loss.backward()`` -> for each
    parameter ``p``: ``p.data -= lr * p.grad`` inside ``torch.no_grad()`` ->
    zero every ``p.grad``. Record the (full-dataset) loss once per step.

    ``batch_size=None`` means full-batch. With a fixed ``seed`` and full-batch,
    this must converge to the same place as ``train_idiomatic`` with plain SGD.

    Returns
    -------
    dict with at least:
        "loss_history" : list[float]   -- length n_steps
        "model"        : ScratchMLP    -- the trained model
    """
    raise NotImplementedError


# ===========================================================================
# Part B -- the idiomatic version
# ===========================================================================


class MLP(nn.Module):
    """The same architecture as ``ScratchMLP``, done properly.

    Store the per-feature embedding layers in ``self.embeddings`` as an
    ``nn.ModuleList`` of ``nn.Embedding`` (``tests/test_week2.py`` inspects
    ``self.embeddings`` to check embedding dimensionality). Use ``nn.Linear``
    and ``nn.ReLU`` for the rest.

    Constructor signature matches ``ScratchMLP`` for an apples-to-apples
    comparison.
    """

    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: list[int],
        emb_dims: list[int],
        hidden_sizes: list[int],
        seed: int = 0,
    ) -> None:
        super().__init__()
        raise NotImplementedError

    def forward(self, x_numeric: torch.Tensor, x_categorical: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape (batch,). Same contract as ScratchMLP."""
        raise NotImplementedError


def make_dataset(
    x_numeric: torch.Tensor, x_categorical: torch.Tensor, y: torch.Tensor
) -> torch.utils.data.Dataset:
    """Wrap the three tensors in a ``TensorDataset``-like dataset yielding
    ``(x_numeric_row, x_categorical_row, y_row)``.
    """
    raise NotImplementedError


def make_dataloader(
    x_numeric: torch.Tensor,
    x_categorical: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    """Build a ``DataLoader`` over ``make_dataset(...)`` with a seeded generator
    so shuffling is reproducible.
    """
    raise NotImplementedError


def train_idiomatic(
    model: MLP,
    train_loader: torch.utils.data.DataLoader,
    *,
    lr: float,
    n_epochs: int,
    weight_decay: float = 0.0,
    seed: int = 0,
) -> dict:
    """Idiomatic training loop: ``torch.optim.SGD``, ``optimizer.zero_grad()``,
    ``loss.backward()``, ``optimizer.step()``.

    Use ``torch.nn.functional.binary_cross_entropy_with_logits`` here (the
    hand-rolled ``bce_with_logits`` was the point of Part A).

    Returns
    -------
    dict with at least:
        "loss_history" : list[float]   -- one entry per epoch (mean batch loss)
        "model"        : MLP
    """
    raise NotImplementedError


@torch.no_grad()
def evaluate(
    model: nn.Module,
    x_numeric: torch.Tensor,
    x_categorical: torch.Tensor,
    y: torch.Tensor,
) -> dict:
    """Evaluate a trained model (either class) on a dataset.

    Returns
    -------
    dict with at least:
        "loss"     : float   -- mean BCE
        "accuracy" : float   -- threshold at p = 0.5
    """
    raise NotImplementedError


def predict_proba(
    model: nn.Module, x_numeric: torch.Tensor, x_categorical: torch.Tensor
) -> np.ndarray:
    """Return goal probabilities (sigmoid of the logits) as a NumPy array,
    shape (batch,).
    """
    raise NotImplementedError
