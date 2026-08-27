# Week 1 — Tensors and gradients

## Learning objective

Make backpropagation stop being magic. By the end of the week you can:

- derive the gradient of a mean-squared-error loss by hand and verify it against `torch.autograd`
- write a gradient-descent loop where **you** compute the parameter update
- explain what a computation graph is and implement reverse-mode autodiff over one

## Build task

Everything goes in `src/week1_autograd.py`. Tests: `tests/test_week1.py`.

### Part 1 — linear regression in pure NumPy

1. `mse_loss(y_pred, y_true)` — the scalar loss.
2. `analytic_grad_linear(X, y, w, b)` — derive `∂L/∂w` and `∂L/∂b` on paper for
   `L = mean((Xw + b − y)²)`, then implement them. The test checks your formula
   against autograd to `1e-8`.
3. `gradient_descent_linear(X, y, lr, n_steps, ...)` — full-batch loop:
   `w ← w − lr · grad_w`, `b ← b − lr · grad_b`. Record the loss each step.

No `nn`, no optimizer, no `nn.functional`.

### Part 2 — polynomial regression

4. `polynomial_features(x, degree)` — design matrix `[x¹, x², …, x^degree]`.
5. `gradient_descent_polynomial(x, y, degree, lr, n_steps)` — reuse the linear
   machinery on the expanded features. You will discover that raw powers of `x`
   have wildly different scales; think about what to do about it.

### Part 3 — the same fits with autograd

6. `gradient_descent_linear_torch` / `gradient_descent_polynomial_torch` — same
   loops, but `w` and `b` are tensors with `requires_grad=True`. Call
   `loss.backward()`, update inside `torch.no_grad()`, then zero the grads
   yourself. Still no optimizer.

### Part 4 — a scalar autograd engine

7. `Value` — a micrograd-style scalar node supporting `+`, `*`, `**` (numeric
   exponent), and `tanh`. Implement `backward()` with a topological sort so each
   node's local gradient rule runs after everything downstream of it.
   - Gradients **accumulate** (`+=`), they do not overwrite. A node used twice
     sums both contributions. Do not zero grads inside `backward()`.

## Definition of done

- `uv run pytest tests/test_week1.py` is all green.
- You can run `notebooks/week1_tensors.ipynb` and see the NumPy and autograd
  loss curves lie on top of each other.
- You can state, without notes, the gradient of `mean((Xw + b − y)²)` w.r.t. `w`.
- Your `Value` engine reproduces `torch`'s gradients on an expression you make up.

## You're stuck if…

- **The analytic gradient is off by a constant factor.** You probably dropped or
  double-counted the `2` from the square, or the `1/n` from the mean.
- **Loss goes to `inf` / `nan`.** Learning rate too high, or (polynomial case)
  features unscaled. Standardise the features or shrink `lr`.
- **NumPy and torch versions disagree.** Check dtype (use `float64` on both while
  debugging), check you zero `grad` after every torch step, check both start
  from the same initial `w`, `b`.
- **`Value.backward()` gives wrong grads on reused nodes.** You're overwriting
  `node.grad` instead of `+=`, or your topological order visits a node before
  all of its consumers.
- **`RuntimeError: element 0 of tensors does not require grad`.** You rebuilt the
  parameter tensor (e.g. `w = w - lr*grad` creates a new non-leaf tensor). Update
  in place: `w -= lr * w.grad` inside `torch.no_grad()`.
