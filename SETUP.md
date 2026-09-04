# Setup — fresh machine checklist

Do this once per machine. It works the same on a locked-down corporate laptop
and a personal machine; the differences are called out.

---

## 1. Prerequisites

- **Python 3.11+** — check with `python3 --version`.
- **uv** — the package manager. Install:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh      # macOS / Linux
  # or: pipx install uv   /   brew install uv
  ```
  Corporate machine with no `curl` to the internet: install `uv` from your
  internal package mirror or `pipx`.
- **git** ≥ 2.30.

---

## 2. Clone

```bash
git clone <this-repo-url> ai-engineer
cd ai-engineer
```

(SSH clone URL — see step 6 for the per-repo SSH key.)

---

## 3. Environment

```bash
uv sync                 # creates .venv/ and installs everything from uv.lock
```

Activate it when you want a shell (optional — `uv run <cmd>` works without):

```bash
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate      # Windows
```

Verify:

```bash
uv run python -c "import torch, mlflow, pandas; print(torch.__version__)"
uv run pytest -q            # every deep-learning test fails with NotImplementedError — that's correct
```

### Behind a corporate proxy

`uv` and `requests` both read `HTTP_PROXY` / `HTTPS_PROXY` from the environment.

```bash
cp .env.example .env
# edit .env: set HTTP_PROXY / HTTPS_PROXY / NO_PROXY
```

`data/download_statsbomb.py` loads `.env` automatically. For `uv sync` itself,
export the proxy vars in your shell (or your company usually sets them globally).
If TLS fails because of a corporate inspection proxy, point
`REQUESTS_CA_BUNDLE` / `SSL_CERT_FILE` at your company CA bundle.

### Switching to a CUDA build of PyTorch

The default is CPU-only torch, which is all the exercises need. On a machine with
an NVIDIA GPU:

```bash
# pick the CUDA version that matches your driver (cu121, cu124, ...)
uv pip install --python .venv torch --index-url https://download.pytorch.org/whl/cu124
```

Or add a source override in `pyproject.toml` under `[tool.uv.sources]` and
re-lock. Keep the CUDA change **local** — don't commit a CUDA-pinned `uv.lock`,
it won't resolve on the CPU-only laptop.

---

## 4. Pre-commit hooks

```bash
./scripts/install_hooks.sh     # `make setup` already runs this
```

This installs the git hook that runs `nbstripout`, `ruff`, and the basic
file-hygiene checks on every commit.

### If you see "Cowardly refusing to install hooks with `core.hooksPath` set"

Plain `pre-commit install` bails out when git's `core.hooksPath` points
somewhere else — a corporate secret scanner such as ggshield sets this
globally. `scripts/install_hooks.sh` handles it: it tries `pre-commit install`
first, and on that specific failure writes `.git/hooks/pre-commit` directly
instead. Scanner wrappers normally call the repo's own hook too, so both end up
running. **Do not** unset `core.hooksPath` globally — that disables secret
scanning in every repo on the machine.

The script warns you if the wrapper does *not* delegate to `.git/hooks/pre-commit`,
in which case section 5 below is the check that matters.

---

## 5. Verify nbstripout actually strips notebook output

**This matters.** Notebook output diffs must never reach a commit — they're huge,
they leak data, and they make review impossible.

```bash
# 1. open a notebook, run a cell that produces output (e.g. print(1+1)), save it
uv run jupyter lab notebooks/week1_tensors.ipynb

# 2. confirm the saved file now contains output
grep -c '"output_type"' notebooks/week1_tensors.ipynb        # > 0

# 3. stage it and let pre-commit run
git add notebooks/week1_tensors.ipynb
uv run pre-commit run nbstripout --files notebooks/week1_tensors.ipynb

# 4. confirm output is gone from the staged + working copy
grep -c '"output_type"' notebooks/week1_tensors.ipynb        # 0
git diff --cached --stat                                     # should show ~no real change
```

If step 4 still shows output, install the filter manually as a fallback:

```bash
uv run nbstripout --install --attributes .gitattributes
git check-attr filter notebooks/week1_tensors.ipynb          # -> filter: nbstripout
```

---

## 6. Per-repo git identity (never leak a work email)

Set the identity **locally in this repo**, not globally, so the wrong email can
never end up in a commit here:

```bash
git config --local user.name  "Your Name"
git config --local user.email "you@personal-domain.example"

git config --local user.email        # verify — must NOT be a work address
```

Optional guard: make git refuse to commit if no local identity is set —

```bash
git config --local user.useConfigOnly true
```

Then a machine-wide missing global identity produces an error instead of
silently falling back to `you@corp.example`.

---

## 7. A separate SSH key for this repo's host

Don't reuse a work key or your default personal key. Make a dedicated one:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_aieng -C "ai-engineer study repo"
# add the PUBLIC key (~/.ssh/id_ed25519_aieng.pub) to your GitHub/GitLab account
```

Add a host alias in `~/.ssh/config`:

```sshconfig
# ~/.ssh/config
Host github-aieng
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_aieng
    IdentitiesOnly yes
```

Now clone / set the remote using the alias, so this repo always uses that key:

```bash
git remote set-url origin git@github-aieng:<your-user>/ai-engineer.git
git remote -v
ssh -T git@github-aieng        # confirms which account the key authenticates as
```

On a corporate laptop, keep this key out of any work SSH agent / config include.

---

## 8. Data & privacy — read once

- **Public datasets only.** This repo uses StatsBomb Open Data. Do not add:
  - any employer data, internal datasets, or credentials
  - any third-party football club's private/proprietary data
- `data/` is gitignored except the `.py` scripts. `.env`, `mlruns/`,
  `checkpoints/`, and `*.pt` are gitignored too. Don't force-add them.
- `check-added-large-files` in pre-commit will block anything over 500 KB —
  if it fires, stop and check what you're about to commit.
- If you publish analysis from this data, credit StatsBomb as the source.

---

## 9. Running notebooks in Google Colab

Colab is fine for the notebooks (free GPU for Week 4 Track 2). Nothing sensitive
is involved — public data, public code.

In the first cell of a Colab notebook:

```python
# clone the repo (public HTTPS clone, or use a fine-grained token for a private repo)
!git clone https://github.com/<your-user>/ai-engineer.git
%cd ai-engineer

# install deps — uv works in Colab:
!pip -q install uv && uv pip install --system -r <(uv export --no-hashes)
# or just: !pip -q install torch numpy pandas scikit-learn matplotlib mlflow tqdm

# make src importable
import sys; sys.path.insert(0, "/content/ai-engineer")
```

Then run `data/download_statsbomb.py` inside Colab to pull the data into the
Colab VM. Do **not** mount Google Drive for this — there's nothing to persist
that shouldn't just be re-downloaded, and mounting Drive pulls in your personal
files unnecessarily.

To get your edited notebook back out: `File → Download → .ipynb`, then run it
through `nbstripout` locally before committing (step 5).

---

## Done

```bash
make test      # deep-learning tests fail with NotImplementedError, data_prep tests pass
```

Start with [`specs/week1.md`](specs/week1.md).
