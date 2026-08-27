# AI Engineer — self-study repo

A 6-month, build-first program to go from experienced data engineer to
production AI Engineer. This repo holds the **infrastructure, exercise specs,
and a test suite**; the deep-learning code is written by the learner.

If you just cloned this: read [`SETUP.md`](SETUP.md) first, then start with
[`specs/week1.md`](specs/week1.md).

## How this repo works

- **`specs/weekN.md`** — what to build that week, the definition of done, and a
  "you're stuck if…" section.
- **`src/weekN_*.py`** — stubs with detailed docstrings. You replace the
  `NotImplementedError` bodies. (`src/data_prep.py` is the exception — it's data
  engineering, already implemented.)
- **`tests/test_weekN.py`** — the specification as executable tests. On a fresh
  clone every deep-learning test fails with `NotImplementedError`. Make them pass.
  Don't edit them.
- **`notebooks/`** — scratch space for plots and exploration. Outputs are
  stripped before commit.
- **`notes/`** — a short write-up per week. Committed.

Workflow each week: read the spec → implement in `src/` → `make test` until green
→ use the notebook to *see* it working → write the note.

## The 6-month arc

| Month | Focus |
|------:|-------|
| **1** | **Deep learning foundations** — tensors, autograd, MLPs, training discipline *(expanded below)* |
| 2 | Real data & the modern stack — embeddings in depth, CNNs, intro sequence models, data/version tooling |
| 3 | Transformers & NLP — attention from scratch, tokenization, fine-tuning, the Hugging Face ecosystem |
| 4 | LLM applications — retrieval-augmented generation, evaluation harnesses, agent/tool patterns |
| 5 | Productionization — model serving, APIs, latency/cost, monitoring, containers, CI/CD for models |
| 6 | Capstone — an end-to-end system with MLOps: pipelines, drift detection, observability, a written portfolio piece |

### Month 1, expanded

The dataset throughout is **StatsBomb Open Data** (public, freely licensed):
predict whether a shot becomes a goal — an "expected goals" (xG) model.

| Week | Spec | Build | Key test |
|-----:|------|-------|----------|
| 1 | [week1](specs/week1.md) | Gradient descent by hand (NumPy + autograd); a scalar autograd engine | analytic grads match `torch.autograd`; `Value` matches torch |
| 2 | [week2](specs/week2.md) | An MLP with embeddings — hand-rolled training loop, then the idiomatic `nn.Module` version | overfit 20 samples to ~0 loss; both versions agree under a seed |
| 3 | [week3](specs/week3.md) | Regularisation & discipline — dropout, weight decay, early stopping, LR schedules, MLflow | seeded runs are bit-identical; early stopping fires at the right epoch |
| 4 | [week4](specs/week4.md) | Capstone — a calibrated xG model benchmarked against gradient boosting | *(no tests — you evaluate)* |

## Quickstart

```bash
# one-time, per machine — see SETUP.md for the full checklist (git identity, SSH)
make setup

# download the default StatsBomb subset (2018 World Cup, ~64 matches)
make data

# run the test suite (expect NotImplementedError until you implement the exercises)
make test

# work in notebooks
make lab

# lint / format check
make lint
```

Other: `uv run pytest tests/test_week1.py -x` to focus on one week;
`uv run mlflow ui` to browse Week 3 experiments.

## Progress checklist

### Month 1

- [ ] Environment set up on machine A (`make setup`, nbstripout verified)
- [ ] Environment set up on machine B
- [ ] `make data` succeeds
- [ ] **Week 1** — `pytest tests/test_week1.py` green
- [ ] Week 1 note written (`notes/week1.md`)
- [ ] **Week 2** — `pytest tests/test_week2.py` green
- [ ] Week 2 note written
- [ ] **Week 3** — `pytest tests/test_week3.py` green
- [ ] Week 3 results table committed (≥ 10 configs)
- [ ] Week 3 note written
- [ ] **Week 4** — capstone track chosen and completed
- [ ] `notes/month-1.md` written

### Later months

- [ ] Month 2 planned and started
- [ ] Month 3 …
- [ ] Month 4 …
- [ ] Month 5 …
- [ ] Month 6 capstone shipped + written up

## Data & privacy

Public datasets only. No employer data and no third-party football club data
belongs in this repo — see `SETUP.md`. If you publish analysis from the
StatsBomb data, credit StatsBomb as the source (their licence asks for it).
