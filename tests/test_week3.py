"""Specification tests for Week 3 -- determinism, early stopping, LR schedules.

The determinism, early-stopping and scheduler tests fail with
``NotImplementedError`` on a fresh clone. (The split-helper requirement from
``specs/week3.md`` is tested in ``tests/test_data_prep.py`` because that helper
lives in the already-implemented ``src/data_prep.py``.)
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.week3_training import (
    EarlyStopping,
    make_scheduler,
    set_seed,
)


# ===========================================================================
# Determinism
# ===========================================================================
def _tiny_train_run(seed: int) -> list[torch.Tensor]:
    set_seed(seed)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    x = torch.randn(64, 4)
    y = torch.randn(64, 1)
    for _ in range(20):
        opt.zero_grad()
        loss = (model(x) - y).pow(2).mean()
        loss.backward()
        opt.step()
    return [p.detach().clone() for p in model.parameters()]


def test_set_seed_makes_runs_bit_identical():
    run_a = _tiny_train_run(0)
    run_b = _tiny_train_run(0)
    for pa, pb in zip(run_a, run_b, strict=True):
        assert torch.equal(pa, pb)


def test_set_seed_different_seeds_differ():
    run_a = _tiny_train_run(0)
    run_b = _tiny_train_run(1)
    assert any(not torch.equal(pa, pb) for pa, pb in zip(run_a, run_b, strict=True))


# ===========================================================================
# EarlyStopping
# ===========================================================================
def _feed(stopper: EarlyStopping, curve: list[float]) -> list[bool]:
    return [stopper.step(v) for v in curve]


def test_early_stopping_triggers_after_patience():
    stopper = EarlyStopping(patience=3)
    curve = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8]
    results = _feed(stopper, curve)

    assert results == [False, False, False, False, False, True]
    assert stopper.best == pytest.approx(0.8)
    assert stopper.best_epoch == 2
    assert stopper.num_bad_epochs == 3
    assert stopper.should_stop is True


def test_early_stopping_respects_min_delta():
    stopper = EarlyStopping(patience=2, min_delta=0.05, mode="min")
    #        e0    e1    e2 (improves vs 1.0-0.05)   e3    e4 -> stop
    curve = [1.0, 0.98, 0.90, 0.88, 0.88]
    results = _feed(stopper, curve)

    assert results == [False, False, False, False, True]
    assert stopper.best == pytest.approx(0.90)
    assert stopper.best_epoch == 2


def test_early_stopping_max_mode():
    stopper = EarlyStopping(patience=2, mode="max")
    curve = [0.50, 0.60, 0.55, 0.55]
    results = _feed(stopper, curve)

    assert results == [False, False, False, True]
    assert stopper.best == pytest.approx(0.60)
    assert stopper.best_epoch == 1


def test_early_stopping_keeps_counting_to_new_best():
    stopper = EarlyStopping(patience=2)
    # dips, recovers to a new best, then plateaus
    _feed(stopper, [1.0, 1.1, 0.7])
    assert stopper.num_bad_epochs == 0
    assert stopper.best == pytest.approx(0.7)
    assert stopper.should_stop is False


# ===========================================================================
# LR schedulers
# ===========================================================================
def _optimizer() -> torch.optim.Optimizer:
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)


def test_make_scheduler_step():
    opt = _optimizer()
    sched = make_scheduler(opt, "step", step_size=1, gamma=0.1)
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.1)
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)


def test_make_scheduler_unknown_kind_raises_value_error():
    with pytest.raises(ValueError):
        make_scheduler(_optimizer(), "not-a-real-scheduler")
