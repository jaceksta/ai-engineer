# Week 2 — First network

## Learning objective

Build and train a multilayer perceptron from first principles, then earn the
PyTorch abstractions by re-deriving them. By the end you can:

- write a full training loop by hand: forward → loss → backward → step → eval
- explain what `nn.Module`, `DataLoader`, and `torch.optim.SGD` each replace
- use embedding layers for categorical features and say why they beat one-hot here
- run the standard "overfit a tiny batch" sanity check and know what it proves

## Build task

Everything goes in `src/week2_mlp.py`. Tests: `tests/test_week2.py`.
Data: `src/data_prep.py` (run `make data` first).

### Part A — from scratch (`ScratchMLP` + `train_scratch`)

- Parameters are bare tensors with `requires_grad=True`. No `nn.Module`.
- One embedding table per categorical feature; concatenate
  `[numeric features, looked-up embeddings]` and pass through hidden layers with
  ReLU, then a final linear layer to **one logit per row**.
- `bce_with_logits(logits, targets)` — implement numerically stable binary
  cross-entropy from logits by hand (`softplus(z) − y·z`, mean-reduced).
- `train_scratch` — the loop: forward, `bce_with_logits`, `loss.backward()`,
  `p.data -= lr * p.grad` for each parameter under `torch.no_grad()`, zero grads.
  `batch_size=None` means full-batch.

### Part B — the idiomatic version (`MLP` + `train_idiomatic`)

- `MLP(nn.Module)` — same architecture. Store embeddings in `self.embeddings`
  as an `nn.ModuleList`. Use `nn.Linear`, `nn.ReLU`.
- `make_dataset` / `make_dataloader` — wrap the tensors, seed the shuffle.
- `train_idiomatic` — `torch.optim.SGD`, `optimizer.zero_grad()`,
  `loss.backward()`, `optimizer.step()`. Use
  `F.binary_cross_entropy_with_logits` here.
- `evaluate` / `predict_proba` — shared eval helpers.

### The agreement check

With the same seed, same data, full-batch, plain SGD, same `lr` and step count,
`ScratchMLP`/`train_scratch` and `MLP`/`train_idiomatic` must produce nearly
identical predictions. This forces your hand-rolled initialisation to match what
`nn.Linear` / `nn.Embedding` do by default. That is the point.

## Definition of done

- `uv run pytest tests/test_week2.py` is all green, including
  `test_scratch_and_idiomatic_agree_under_fixed_seed`.
- `test_scratch_overfits_twenty_samples` passes — your model drives training loss
  on 20 samples to essentially zero. If it can't, the model or the loop is broken.
- In the notebook: a loss curve that goes down, and the two implementations'
  curves sitting on top of each other.
- You can explain why `test_scratch_overfits_twenty_samples` is a meaningful test
  and what a *failure* of it would tell you.

## You're stuck if…

- **Overfit-20 won't reach zero loss.** Bug in the backward path or the update.
  Check: are all parameters in `parameters()`? Are grads zeroed each step? Is the
  logit shape `(batch,)` not `(batch, 1)` (broadcasting silently ruins the loss)?
- **`bce_with_logits` doesn't match torch.** You implemented
  `−[y log σ(z) + (1−y) log(1−σ(z))]` directly and it overflows. Use the
  `softplus` form. Watch the sign.
- **The two versions don't agree.** `nn.Linear` initialises weights with
  Kaiming-uniform and a specific bias range; `nn.Embedding` uses `N(0, 1)`. Match
  those exactly in `ScratchMLP.__init__`, seeding right before each allocation.
  Also confirm `train_idiomatic` uses `momentum=0` and `weight_decay=0`.
- **Embeddings error with `index out of range`.** A categorical code ≥
  `num_embeddings`. Your `cat_cardinalities` must be `max_code + 1`; unknown
  codes must map to `0` (see `data_prep.build_category_maps`).
- **Loss is `nan` after one step.** `lr` too high, or you forgot to zero grads so
  they accumulated across steps.
