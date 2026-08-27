"""Specification tests for Week 2 -- an MLP for shot-outcome classification,
built from scratch and then the idiomatic way.

Every test fails with ``NotImplementedError`` on a fresh clone. The tests are
the contract; don't edit them to pass.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from src.week2_mlp import (
    MLP,
    ScratchMLP,
    bce_with_logits,
    default_emb_dim,
    evaluate,
    make_dataloader,
    predict_proba,
    train_idiomatic,
    train_scratch,
)

CAT_CARDINALITIES = [5, 3]
EMB_DIMS = [4, 2]
HIDDEN = [16, 8]


def make_shot_data(n: int, seed: int = 0):
    """Synthetic (x_numeric, x_categorical, y) with a learnable signal."""
    rng = np.random.default_rng(seed)
    x_num = rng.normal(size=(n, 3)).astype(np.float32)
    x_cat = np.stack(
        [rng.integers(0, CAT_CARDINALITIES[0], n), rng.integers(0, CAT_CARDINALITIES[1], n)],
        axis=1,
    ).astype(np.int64)
    logit = 1.3 * x_num[:, 0] - 0.8 * x_num[:, 1] + 0.9 * (x_cat[:, 0] == 1)
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(np.float32)
    return torch.from_numpy(x_num), torch.from_numpy(x_cat), torch.from_numpy(y)


def new_scratch(seed: int = 0) -> ScratchMLP:
    return ScratchMLP(3, CAT_CARDINALITIES, EMB_DIMS, HIDDEN, seed=seed)


def new_mlp(seed: int = 0) -> MLP:
    return MLP(3, CAT_CARDINALITIES, EMB_DIMS, HIDDEN, seed=seed)


# ===========================================================================
# helpers
# ===========================================================================
def test_default_emb_dim_formula():
    assert default_emb_dim(2) == 1
    assert default_emb_dim(10) == 5
    assert default_emb_dim(11) == 6
    assert default_emb_dim(200) == 50


def test_bce_with_logits_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(32)
    targets = (torch.rand(32) > 0.5).float()
    got = bce_with_logits(logits, targets)
    expected = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(got, expected, atol=1e-6)


# ===========================================================================
# shapes and ranges
# ===========================================================================
def test_scratch_forward_shape():
    x_num, x_cat, _ = make_shot_data(24)
    logits = new_scratch().forward(x_num, x_cat)
    assert logits.shape == (24,)


def test_mlp_forward_shape():
    x_num, x_cat, _ = make_shot_data(24)
    logits = new_mlp().forward(x_num, x_cat)
    assert logits.shape == (24,)


def test_predict_proba_in_unit_interval():
    x_num, x_cat, _ = make_shot_data(24)
    proba = predict_proba(new_mlp(), x_num, x_cat)
    proba = np.asarray(proba)
    assert proba.shape == (24,)
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_embeddings_have_expected_dimensionality():
    model = new_mlp()
    embs = list(model.embeddings)
    assert len(embs) == len(CAT_CARDINALITIES)
    for emb, card, dim in zip(embs, CAT_CARDINALITIES, EMB_DIMS, strict=True):
        assert isinstance(emb, nn.Embedding)
        assert emb.num_embeddings == card
        assert emb.embedding_dim == dim


# ===========================================================================
# training behaviour
# ===========================================================================
def test_scratch_training_reduces_loss():
    x_num, x_cat, y = make_shot_data(256)
    out = train_scratch(new_scratch(), x_num, x_cat, y, lr=0.1, n_steps=200, seed=0)
    history = out["loss_history"]
    assert len(history) == 200
    assert history[-1] < history[0] * 0.9


def test_scratch_overfits_twenty_samples():
    x_num, x_cat, y = make_shot_data(20, seed=1)
    out = train_scratch(new_scratch(), x_num, x_cat, y, lr=0.2, n_steps=3000, seed=0)
    assert out["loss_history"][-1] < 0.05


def test_idiomatic_training_reduces_loss():
    x_num, x_cat, y = make_shot_data(256)
    loader = make_dataloader(x_num, x_cat, y, batch_size=64, shuffle=True, seed=0)
    out = train_idiomatic(new_mlp(), loader, lr=0.1, n_epochs=50, seed=0)
    assert out["loss_history"][-1] < out["loss_history"][0] * 0.9


def test_scratch_and_idiomatic_agree_under_fixed_seed():
    """Full-batch plain SGD, identical seed and data: the two implementations
    should land in nearly the same place. Getting this to pass means your
    parameter initialisation matches between ScratchMLP and MLP.
    """
    x_num, x_cat, y = make_shot_data(128, seed=2)

    scratch_out = train_scratch(
        new_scratch(seed=0), x_num, x_cat, y, lr=0.05, n_steps=300, batch_size=None, seed=0
    )
    loader = make_dataloader(x_num, x_cat, y, batch_size=len(y), shuffle=False, seed=0)
    idiomatic_out = train_idiomatic(new_mlp(seed=0), loader, lr=0.05, n_epochs=300, seed=0)

    p_scratch = predict_proba(scratch_out["model"], x_num, x_cat)
    p_idiomatic = predict_proba(idiomatic_out["model"], x_num, x_cat)
    np.testing.assert_allclose(np.asarray(p_scratch), np.asarray(p_idiomatic), atol=1e-2)

    e_scratch = evaluate(scratch_out["model"], x_num, x_cat, y)
    e_idiomatic = evaluate(idiomatic_out["model"], x_num, x_cat, y)
    assert e_scratch["loss"] == pytest.approx(e_idiomatic["loss"], abs=5e-3)


def test_evaluate_reports_loss_and_accuracy():
    x_num, x_cat, y = make_shot_data(128)
    metrics = evaluate(new_mlp(), x_num, x_cat, y)
    assert "loss" in metrics and "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
