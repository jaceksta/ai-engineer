# Week 4 — Capstone

No new stubs, no new tests. You drive. Pick one track; **Depth is recommended**.

## Track 1 — Depth: a properly evaluated xG model (recommended)

### Objective

Take the Week 2/3 model from "it trains" to "I trust this number". The skill
being built is *evaluation*, not architecture.

### Build task

1. **Feature engineering.** Beyond distance and angle: body part, shot type, play
   pattern, under pressure, first-time. Optionally parse `shot.freeze_frame` for
   defenders/keeper position (the raw data has it). Split by match.
2. **Train** your best regularised MLP from Week 3.
3. **Calibrate and evaluate:**
   - reliability curve (predicted probability vs observed goal rate, binned)
   - Brier score, and log loss, on the held-out test set
   - compare raw model probabilities to Platt scaling / isotonic regression
   - compare your predictions to StatsBomb's own `statsbomb_xg` column
4. **Benchmark** against gradient-boosted trees (`sklearn.ensemble.
   HistGradientBoostingClassifier` or `xgboost`) on the same features and split.

### Definition of done

- A test-set reliability curve and Brier score for the MLP and the GBM.
- A short written comparison: which model is better calibrated, which has lower
  Brier score, and whether the difference is meaningful given the test-set size.
- **The deep model may well lose to the GBM.** That is a valid and informative
  result. Tabular data with ~10 features and a few thousand rows is GBM
  territory. Write up *why* — it is one of the most useful things you can learn
  this month about when not to reach for deep learning.

## Track 2 — Breadth: one small CNN on CIFAR-10

### Objective

One afternoon, so convolution isn't a black box when it comes up later.

### Build task

- `torchvision.datasets.CIFAR10`, a small conv net (2–3 conv blocks + a head),
  trained to a non-embarrassing accuracy (~70%+ is fine).
- Visualise the first-layer filters and a few feature maps.

### Definition of done

- Training curve, final test accuracy, and the filter visualisation.
- Two or three sentences on what a convolution layer computes and why weight
  sharing matters.

## Both tracks

Write `notes/month-1.md` in your own words (see the template). This is not
optional — writing it is where the month consolidates.
