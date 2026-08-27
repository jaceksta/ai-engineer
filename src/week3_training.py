"""Week 3 -- Making it trustworthy.  *** YOU IMPLEMENT THE STUBS IN THIS FILE. ***

Goal: the discipline layer. Take the Week 2 model, overfit it on purpose and
watch validation loss diverge, then add the tools that make training
trustworthy: dropout, weight decay, early stopping, LR scheduling, seeded
determinism, and experiment tracking with MLflow.

What you implement (raises ``NotImplementedError`` below):
    * ``set_seed`` / ``make_generator`` / ``seed_worker`` -- determinism utils
    * ``EarlyStopping`` -- the patience-based stopper
    * ``make_scheduler`` -- LR schedule factory
    * ``train_with_early_stopping`` -- the full regularised training loop
    * ``MLflowLogger`` -- a thin logging wrapper

What is already done for you (plumbing, not learning):
    * ``TrainConfig`` -- a plain record for one run's hyper-parameters
    * ``results_table`` -- turn a list of run dicts into a sorted DataFrame

The tests in ``tests/test_week3.py`` are the specification. Read
``specs/week3.md`` for the full task.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pandas as pd
import torch

# ===========================================================================
# Determinism  (YOU IMPLEMENT)
# ===========================================================================


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that affects training so two runs are bit-identical.

    Must cover: Python ``random``, NumPy, ``torch`` (CPU + CUDA), and -- when
    ``deterministic`` -- ``torch.backends.cudnn.deterministic = True`` /
    ``benchmark = False`` and ``torch.use_deterministic_algorithms(True)``.

    ``tests/test_week3.py`` trains a tiny model twice with the same seed and
    asserts the resulting parameter tensors are exactly equal.
    """
    raise NotImplementedError


def make_generator(seed: int) -> torch.Generator:
    """Return a ``torch.Generator`` seeded with ``seed`` (for DataLoader shuffle)."""
    raise NotImplementedError


def seed_worker(worker_id: int) -> None:
    """``worker_init_fn`` for ``DataLoader`` so each worker's NumPy/random state
    is derived deterministically from ``torch.initial_seed()``.
    """
    raise NotImplementedError


# ===========================================================================
# Early stopping  (YOU IMPLEMENT)
# ===========================================================================


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Semantics (the tests pin these down exactly -- implement as written):

    * Construct with ``patience`` (int >= 1), ``min_delta`` (float >= 0,
      default 0.0), ``mode`` in {"min", "max"} (default "min").
    * Call ``step(metric)`` exactly once per epoch, in order. Epochs are
      counted from 0 (the first ``step`` call is epoch 0).
    * An epoch *improves* if:
          mode == "min":  metric <  best - min_delta
          mode == "max":  metric >  best + min_delta
      The first ``step`` call always counts as an improvement.
    * On improvement: ``best`` and ``best_epoch`` are updated and
      ``num_bad_epochs`` resets to 0.
    * Otherwise ``num_bad_epochs += 1``. When ``num_bad_epochs >= patience``,
      ``should_stop`` becomes True.
    * ``step`` returns ``should_stop``.

    Public attributes tests read: ``best`` (float), ``best_epoch`` (int),
    ``num_bad_epochs`` (int), ``should_stop`` (bool).
    """

    def __init__(self, patience: int, min_delta: float = 0.0, mode: str = "min") -> None:
        raise NotImplementedError

    def step(self, metric: float) -> bool:
        raise NotImplementedError


# ===========================================================================
# LR scheduling  (YOU IMPLEMENT)
# ===========================================================================


def make_scheduler(
    optimizer: torch.optim.Optimizer, kind: str, **kwargs: Any
) -> torch.optim.lr_scheduler.LRScheduler:
    """Factory for a small set of schedulers, selected by ``kind``:

        "step"     -> StepLR(step_size=..., gamma=...)
        "cosine"   -> CosineAnnealingLR(T_max=...)
        "plateau"  -> ReduceLROnPlateau(mode=..., factor=..., patience=...)

    Pass the scheduler-specific args through ``**kwargs``. Raise ``ValueError``
    for an unknown ``kind``.
    """
    raise NotImplementedError


# ===========================================================================
# The regularised training loop  (YOU IMPLEMENT)
# ===========================================================================


def train_with_early_stopping(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    *,
    lr: float,
    max_epochs: int,
    weight_decay: float = 0.0,
    patience: int = 10,
    scheduler_kind: str | None = None,
    scheduler_kwargs: dict | None = None,
    seed: int = 0,
    logger: MLflowLogger | None = None,
) -> dict:
    """Full training loop with weight decay, optional LR scheduling, early
    stopping, and optional MLflow logging.

    Behaviour:
        * call ``set_seed(seed)`` first
        * optimiser: ``torch.optim.SGD`` (or Adam) with ``weight_decay``
        * after each epoch: compute train + val loss, step the scheduler,
          step ``EarlyStopping`` on val loss, log metrics via ``logger``
        * keep a copy of the best (lowest val loss) ``state_dict`` and reload
          it into ``model`` before returning

    Returns
    -------
    dict with at least:
        "train_loss" : list[float]
        "val_loss"   : list[float]
        "best_epoch" : int
        "best_val_loss" : float
        "stopped_early" : bool
        "model" : torch.nn.Module   -- with best weights restored
    """
    raise NotImplementedError


# ===========================================================================
# MLflow logging wrapper  (YOU IMPLEMENT)
# ===========================================================================


class MLflowLogger:
    """Thin wrapper over the MLflow fluent API for one experiment.

    Usage you should support::

        logger = MLflowLogger(experiment_name="week3", tracking_uri="file:./mlruns")
        with logger.run(run_name="baseline"):
            logger.log_params({"lr": 0.01, "weight_decay": 0.0})
            for epoch, (tr, va) in enumerate(...):
                logger.log_metrics({"train_loss": tr, "val_loss": va}, step=epoch)

    Keep it small: set the tracking URI + experiment in ``__init__``; ``run`` is
    a context manager around ``mlflow.start_run``; the log methods forward to
    ``mlflow.log_params`` / ``mlflow.log_metrics``.
    """

    def __init__(self, experiment_name: str, tracking_uri: str = "file:./mlruns") -> None:
        raise NotImplementedError

    @contextmanager
    def run(self, run_name: str | None = None) -> Iterator[None]:
        raise NotImplementedError

    def log_params(self, params: dict) -> None:
        raise NotImplementedError

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        raise NotImplementedError


# ===========================================================================
# Plumbing -- already implemented
# ===========================================================================


@dataclasses.dataclass
class TrainConfig:
    """One row of your hyper-parameter sweep. A plain record, nothing clever.

    Log ``dataclasses.asdict(config)`` to MLflow so every run is reproducible.
    """

    lr: float = 1e-2
    weight_decay: float = 0.0
    dropout: float = 0.0
    hidden_sizes: tuple[int, ...] = (64, 32)
    batch_size: int = 256
    max_epochs: int = 200
    patience: int = 15
    scheduler_kind: str | None = None
    seed: int = 0
    note: str = ""

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def results_table(runs: list[dict], sort_by: str = "best_val_loss") -> pd.DataFrame:
    """Turn a list of run-summary dicts into a tidy, sorted DataFrame.

    Each dict is typically ``{**config.as_dict(), **history_summary}``. Sorting
    ascending is right for a loss column; pass ``sort_by`` for anything else.
    """
    df = pd.DataFrame(runs)
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=True).reset_index(drop=True)
    return df
