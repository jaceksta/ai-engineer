# Week 3 — Making it trustworthy

## Learning objective

The discipline layer — the difference between running a tutorial and shipping a
model. By the end you can:

- recognise overfitting from a train/validation loss plot and say when to stop
- apply dropout, weight decay, early stopping, and LR scheduling, and explain
  what each one actually does
- make a training run bit-for-bit reproducible
- track experiments so "which run was that?" is never a question

## Build task

Stubs in `src/week3_training.py`. Tests: `tests/test_week3.py` (plus the
split-helper test in `tests/test_data_prep.py`).

1. **Determinism** — `set_seed`, `make_generator`, `seed_worker`. After
   `set_seed(0)`, two identical training runs must produce identical weights.
2. **`EarlyStopping`** — patience-based, `mode` in `{"min", "max"}`, `min_delta`.
   Exact semantics are in the class docstring and pinned by the tests. Call
   `step(metric)` once per epoch; it returns whether to stop.
3. **`make_scheduler`** — factory for `StepLR`, `CosineAnnealingLR`,
   `ReduceLROnPlateau`; `ValueError` on anything else.
4. **`train_with_early_stopping`** — the full loop: weight decay via the
   optimizer, optional LR schedule, early stopping on validation loss, restore
   the best weights before returning, log every epoch via `MLflowLogger`.
5. **`MLflowLogger`** — thin wrapper: set tracking URI + experiment, a `run`
   context manager, `log_params`, `log_metrics`.

Then, in `notebooks/week3_regularization.ipynb`:

- Deliberately overfit the Week 2 model (tiny data, big model, no regularisation,
  many epochs). Plot train vs validation loss diverging. Keep the plot.
- Add each regulariser in turn. Log **every** run to MLflow.
- Produce a results table (`results_table(...)`) comparing **at least 10**
  configurations — vary dropout, weight decay, LR schedule, patience, seed.

## Definition of done

- `uv run pytest tests/test_week3.py tests/test_data_prep.py` is all green.
- `test_set_seed_makes_runs_bit_identical` passes — no hidden nondeterminism.
- MLflow UI (`uv run mlflow ui`) shows your runs with params and per-epoch metrics.
- A committed results table (markdown or CSV under `notes/`) with ≥ 10 rows and a
  one-paragraph reading of it: which regulariser helped, which didn't, and your
  best guess why.
- You can explain why the train/val split is **by match**, not by row (see
  `data_prep.split_by_match` — the docstring has the argument).

## You're stuck if…

- **Runs still differ after `set_seed`.** You missed a source: `DataLoader`
  shuffling (needs a seeded `generator` + `worker_init_fn`), or
  `torch.use_deterministic_algorithms(True)` not set, or the model built before
  the seed was set.
- **Early-stopping tests fail by one epoch.** Re-read the docstring: epochs count
  from 0, the first `step` is always an improvement, stop fires when
  `num_bad_epochs >= patience` (not `>`).
- **Validation loss is *lower* than training loss.** Usually dropout: it's active
  in train, off in eval. Confirm you call `model.train()` / `model.eval()`.
- **Weight decay seems to do nothing.** Values like `1e-5` are too small to see
  on a short run; try a sweep from `1e-4` to `1e-1`. Also note SGD weight decay ≠
  AdamW weight decay.
- **MLflow writes runs somewhere surprising.** Set the tracking URI explicitly
  (`file:./mlruns`) before `start_run`; `mlruns/` is gitignored on purpose.
- **`ReduceLROnPlateau` errors on `.step()`.** It needs the metric:
  `scheduler.step(val_loss)`, unlike the others.
