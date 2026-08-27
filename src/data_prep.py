"""Turn raw StatsBomb event JSON into a flat shot-level table for goal/no-goal
classification.

This module is *data engineering*, not deep learning, so it is fully
implemented. You will import from it in weeks 2-4.

Pipeline
--------
1. ``data/download_statsbomb.py`` writes raw JSON to ``data/raw/``.
2. ``build_shot_dataframe`` / ``shots_for_competition`` flatten the shot
   events into one row per shot with engineered geometry + categorical IDs.
3. ``split_by_match`` gives you disjoint train/val/test frames split *by match*.
4. ``build_category_maps`` + ``encode_categoricals`` produce the contiguous
   integer codes you feed to ``nn.Embedding`` in week 2.

Pitch / coordinate conventions (StatsBomb):
    * The pitch is 120 (length) x 80 (width) units, measured in yards.
    * A team always attacks towards x = 120 in its own event stream, so every
      shot's target goal mouth is the segment x = 120, 36 <= y <= 44.
    * ``location`` is ``[x, y]``; ``shot.end_location`` is ``[x, y]`` or
      ``[x, y, z]``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

# --- Pitch geometry -------------------------------------------------------
PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_CENTER = (120.0, 40.0)
GOAL_POST_LEFT = (120.0, 36.0)
GOAL_POST_RIGHT = (120.0, 44.0)
GOAL_WIDTH = GOAL_POST_RIGHT[1] - GOAL_POST_LEFT[1]  # 8.0

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Categorical columns that later become embedding inputs. Kept as raw
# StatsBomb IDs here; `build_category_maps` turns them into contiguous codes.
CATEGORICAL_COLS = [
    "body_part_id",
    "shot_type_id",
    "play_pattern_id",
    "technique_id",
    "position_id",
]

# Numeric feature columns produced by `build_shot_dataframe`.
NUMERIC_COLS = [
    "x",
    "y",
    "distance_to_goal",
    "angle_to_goal",
    "under_pressure",
    "first_time",
    "is_penalty",
]


# --- Raw file loading ---------------------------------------------------------
def _read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_competitions(raw_dir: Path | str = RAW_DIR) -> list[dict]:
    """Return the parsed ``competitions.json`` list."""
    return _read_json(Path(raw_dir) / "competitions.json")


def load_matches(competition_id: int, season_id: int, raw_dir: Path | str = RAW_DIR) -> list[dict]:
    """Return the match list for one competition/season."""
    path = Path(raw_dir) / "matches" / str(competition_id) / f"{season_id}.json"
    return _read_json(path)


def load_events(match_id: int, raw_dir: Path | str = RAW_DIR) -> list[dict]:
    """Return the full event list for one match."""
    return _read_json(Path(raw_dir) / "events" / f"{match_id}.json")


def match_ids_for_competition(
    competition_id: int, season_id: int, raw_dir: Path | str = RAW_DIR
) -> list[int]:
    """All ``match_id`` values for a competition/season, sorted."""
    matches = load_matches(competition_id, season_id, raw_dir=raw_dir)
    return sorted(int(m["match_id"]) for m in matches)


# --- Geometry ---------------------------------------------------------------
def distance_to_goal(x: float, y: float) -> float:
    """Euclidean distance from ``(x, y)`` to the centre of the goal mouth."""
    gx, gy = GOAL_CENTER
    return math.hypot(gx - x, gy - y)


def angle_to_goal(x: float, y: float) -> float:
    """Angle (radians) subtended by the two goalposts at ``(x, y)``.

    Uses the law of cosines on the triangle (shot, left post, right post).
    Returns 0.0 for degenerate positions (on the goal line between the posts,
    or exactly on a post). Wide, tight-angle chances get a small value;
    central, close chances get a value approaching pi.
    """
    a = math.hypot(GOAL_POST_LEFT[0] - x, GOAL_POST_LEFT[1] - y)
    b = math.hypot(GOAL_POST_RIGHT[0] - x, GOAL_POST_RIGHT[1] - y)
    if a == 0.0 or b == 0.0:
        return 0.0
    cos_theta = (a * a + b * b - GOAL_WIDTH * GOAL_WIDTH) / (2 * a * b)
    # Floating point can push this a hair outside [-1, 1].
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


# --- Shot extraction --------------------------------------------------------
def iter_shot_events(events: Iterable[dict]) -> Iterator[dict]:
    """Yield the event dicts that are shots (``type.name == "Shot"``).

    Own goals are a different event type and are not yielded.
    """
    for event in events:
        if event.get("type", {}).get("name") == "Shot":
            yield event


def _nested(d: dict, *keys: str, default=None):
    cur: object = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def shot_event_to_row(event: dict, match_id: int) -> dict | None:
    """Flatten one StatsBomb shot event into a plain dict.

    Returns ``None`` if the shot has no usable ``location`` (rare, but a few
    events in the open data are missing coordinates and cannot be given
    geometry features).
    """
    location = event.get("location")
    if not location or len(location) < 2:
        return None
    x, y = float(location[0]), float(location[1])

    shot = event.get("shot", {})
    outcome_name = _nested(shot, "outcome", "name")
    shot_type_name = _nested(shot, "type", "name")

    return {
        "match_id": int(match_id),
        "event_id": event.get("id"),
        "index": event.get("index"),
        "period": event.get("period"),
        "minute": event.get("minute"),
        "second": event.get("second"),
        "team_id": _nested(event, "team", "id"),
        "team_name": _nested(event, "team", "name"),
        "possession_team_id": _nested(event, "possession_team", "id"),
        "player_id": _nested(event, "player", "id"),
        "player_name": _nested(event, "player", "name"),
        "position_id": _nested(event, "position", "id"),
        "position_name": _nested(event, "position", "name"),
        # geometry
        "x": x,
        "y": y,
        "distance_to_goal": distance_to_goal(x, y),
        "angle_to_goal": angle_to_goal(x, y),
        # categoricals (raw StatsBomb IDs; encode later for embeddings)
        "body_part_id": _nested(shot, "body_part", "id"),
        "body_part_name": _nested(shot, "body_part", "name"),
        "shot_type_id": _nested(shot, "type", "id"),
        "shot_type_name": shot_type_name,
        "play_pattern_id": _nested(event, "play_pattern", "id"),
        "play_pattern_name": _nested(event, "play_pattern", "name"),
        "technique_id": _nested(shot, "technique", "id"),
        "technique_name": _nested(shot, "technique", "name"),
        # binary flags -> stored as 0/1 ints
        "under_pressure": int(bool(event.get("under_pressure", False))),
        "first_time": int(bool(shot.get("first_time", False))),
        "is_penalty": int(shot_type_name == "Penalty"),
        # targets / benchmarks
        "outcome_name": outcome_name,
        "goal": int(outcome_name == "Goal"),
        "statsbomb_xg": shot.get("statsbomb_xg"),
    }


def build_shot_dataframe(
    match_ids: Iterable[int],
    raw_dir: Path | str = RAW_DIR,
    include_penalties: bool = True,
) -> pd.DataFrame:
    """Build a one-row-per-shot dataframe from a set of match IDs.

    Parameters
    ----------
    match_ids:
        Matches to read from ``raw_dir/events/{match_id}.json``.
    raw_dir:
        Root of the downloaded data (default: ``data/raw``).
    include_penalties:
        Penalties have a fixed, very high conversion rate and no useful
        geometry variation. Keep them in only if you plan to model them
        explicitly; drop them for a cleaner open-play xG model.

    Returns
    -------
    DataFrame with the columns described in the module docstring. Categorical
    ID columns are nullable integers; missing IDs are ``pd.NA``.
    """
    rows: list[dict] = []
    for match_id in match_ids:
        events = load_events(match_id, raw_dir=raw_dir)
        for event in iter_shot_events(events):
            row = shot_event_to_row(event, match_id)
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if not include_penalties:
        df = df[df["is_penalty"] == 0].reset_index(drop=True)

    # Nullable integer dtype keeps categorical IDs as ints even with gaps.
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("Int64")

    return df


def shots_for_competition(
    competition_id: int,
    season_id: int,
    raw_dir: Path | str = RAW_DIR,
    include_penalties: bool = True,
) -> pd.DataFrame:
    """Convenience wrapper: every shot in one competition/season."""
    match_ids = match_ids_for_competition(competition_id, season_id, raw_dir=raw_dir)
    return build_shot_dataframe(match_ids, raw_dir=raw_dir, include_penalties=include_penalties)


# --- Splitting -------------------------------------------------------------
def split_by_match(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 0,
    match_col: str = "match_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into train/val/test so that no match spans two splits.

    Why split by match and not by row
    ---------------------------------
    Shots from the same match are not independent: they share the two teams,
    the tactical set-up, the pitch, the weather, the referee, and the game
    state. A row-wise split leaks all of that from train into validation, so
    your validation score looks better than the model's true performance on
    an unseen match. Splitting by match is the honest evaluation and matches
    how the model would be used (predict for a *new* game).

    Returns
    -------
    ``(train_df, val_df, test_df)`` -- row order within each frame is
    preserved from the input; indexes are reset.
    """
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must be in [0, 1) and sum to < 1")

    matches = np.array(sorted(df[match_col].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(matches)

    n = len(matches)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_ids = set(matches[:n_test].tolist())
    val_ids = set(matches[n_test : n_test + n_val].tolist())

    is_test = df[match_col].isin(test_ids)
    is_val = df[match_col].isin(val_ids)

    train_df = df[~is_test & ~is_val].reset_index(drop=True)
    val_df = df[is_val].reset_index(drop=True)
    test_df = df[is_test].reset_index(drop=True)
    return train_df, val_df, test_df


# --- Categorical encoding for embeddings -----------------------------------
def build_category_maps(
    df: pd.DataFrame, cols: Iterable[str] = CATEGORICAL_COLS
) -> dict[str, dict]:
    """Map each column's raw values to contiguous codes ``1..k``.

    Code ``0`` is reserved for "unknown / missing", so an ``nn.Embedding`` for
    a column needs ``num_embeddings = len(map) + 1``. Build the maps on the
    training split only, then apply them to val/test so unseen categories
    correctly fall through to ``0``.
    """
    maps: dict[str, dict] = {}
    for col in cols:
        values = sorted(v for v in df[col].dropna().unique())
        maps[col] = {value: code for code, value in enumerate(values, start=1)}
    return maps


def encode_categoricals(df: pd.DataFrame, maps: dict[str, dict]) -> pd.DataFrame:
    """Return a copy of ``df`` with an added ``{col}_idx`` int column per map."""
    out = df.copy()
    for col, mapping in maps.items():
        out[f"{col}_idx"] = out[col].map(mapping).fillna(0).astype("int64")
    return out


def embedding_sizes(maps: dict[str, dict]) -> dict[str, int]:
    """``num_embeddings`` (including the reserved 0 slot) for each column."""
    return {col: len(mapping) + 1 for col, mapping in maps.items()}
