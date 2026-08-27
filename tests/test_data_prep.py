"""Tests for ``src/data_prep.py``.

Unlike the weekN tests, these cover code that is *already implemented* (data
engineering plumbing), so they pass on a fresh clone. They guard against
regressions while you build on top of the shot table.
"""

from __future__ import annotations

import json
import math

import pytest

from src import data_prep as dp


# --- geometry --------------------------------------------------------------
def test_distance_to_goal():
    assert dp.distance_to_goal(114.0, 40.0) == pytest.approx(6.0)
    assert dp.distance_to_goal(120.0, 40.0) == pytest.approx(0.0)
    assert dp.distance_to_goal(108.0, 44.0) == pytest.approx(math.hypot(12.0, 4.0))


def test_angle_to_goal_ordering():
    central_close = dp.angle_to_goal(114.0, 40.0)
    penalty_spot = dp.angle_to_goal(108.0, 40.0)
    wide_and_far = dp.angle_to_goal(114.0, 10.0)
    assert central_close > penalty_spot > wide_and_far > 0.0
    assert penalty_spot == pytest.approx(math.acos(0.8), abs=1e-6)


def test_angle_to_goal_is_clamped_on_goal_line():
    # exactly on a post -> degenerate, must not raise
    assert dp.angle_to_goal(120.0, 36.0) == 0.0


# --- shot extraction ------------------------------------------------------
def _fake_shot_event(outcome="Goal", with_location=True):
    event = {
        "id": "abc-123",
        "index": 42,
        "period": 1,
        "minute": 10,
        "second": 5,
        "type": {"id": 16, "name": "Shot"},
        "team": {"id": 1, "name": "A"},
        "possession_team": {"id": 1, "name": "A"},
        "player": {"id": 99, "name": "Striker"},
        "position": {"id": 23, "name": "Center Forward"},
        "play_pattern": {"id": 1, "name": "Regular Play"},
        "under_pressure": True,
        "shot": {
            "statsbomb_xg": 0.35,
            "outcome": {"id": 97, "name": outcome},
            "type": {"id": 87, "name": "Open Play"},
            "body_part": {"id": 40, "name": "Right Foot"},
            "technique": {"id": 93, "name": "Normal"},
            "first_time": True,
        },
    }
    if with_location:
        event["location"] = [110.0, 40.0]
    return event


def test_iter_shot_events_filters_non_shots():
    events = [
        {"type": {"name": "Pass"}},
        _fake_shot_event(),
        {"type": {"name": "Pressure"}},
    ]
    shots = list(dp.iter_shot_events(events))
    assert len(shots) == 1


def test_shot_event_to_row_fields():
    row = dp.shot_event_to_row(_fake_shot_event(outcome="Goal"), match_id=7)
    assert row["match_id"] == 7
    assert row["goal"] == 1
    assert row["under_pressure"] == 1
    assert row["first_time"] == 1
    assert row["is_penalty"] == 0
    assert row["body_part_id"] == 40
    assert row["play_pattern_id"] == 1
    assert row["distance_to_goal"] == pytest.approx(10.0)
    assert row["statsbomb_xg"] == pytest.approx(0.35)

    miss = dp.shot_event_to_row(_fake_shot_event(outcome="Off T"), match_id=7)
    assert miss["goal"] == 0


def test_shot_event_to_row_without_location_is_none():
    assert dp.shot_event_to_row(_fake_shot_event(with_location=False), match_id=1) is None


def test_build_shot_dataframe_from_disk(tmp_path):
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    for match_id in (100, 101):
        payload = [
            {"type": {"name": "Pass"}},
            _fake_shot_event(outcome="Goal"),
            _fake_shot_event(outcome="Saved"),
        ]
        (events_dir / f"{match_id}.json").write_text(json.dumps(payload))

    df = dp.build_shot_dataframe([100, 101], raw_dir=tmp_path)
    assert len(df) == 4
    assert set(df["match_id"]) == {100, 101}
    assert df["goal"].sum() == 2
    for col in dp.CATEGORICAL_COLS:
        assert col in df.columns


# --- splitting ----------------------------------------------------------
def _synthetic_df(n_matches=20, rows_per_match=10):
    import pandas as pd

    records = []
    for match_id in range(n_matches):
        for _ in range(rows_per_match):
            records.append({"match_id": match_id, "goal": 0, "body_part_id": 40})
    return pd.DataFrame(records)


def test_split_by_match_is_disjoint_and_complete():
    df = _synthetic_df()
    train, val, test = dp.split_by_match(df, val_frac=0.2, test_frac=0.2, seed=0)

    train_m = set(train["match_id"])
    val_m = set(val["match_id"])
    test_m = set(test["match_id"])

    assert train_m and val_m and test_m
    assert train_m.isdisjoint(val_m)
    assert train_m.isdisjoint(test_m)
    assert val_m.isdisjoint(test_m)
    assert train_m | val_m | test_m == set(df["match_id"])
    assert len(train) + len(val) + len(test) == len(df)


def test_split_by_match_is_seed_stable():
    df = _synthetic_df()
    a = dp.split_by_match(df, seed=123)[0]["match_id"].tolist()
    b = dp.split_by_match(df, seed=123)[0]["match_id"].tolist()
    assert a == b


def test_split_by_match_rejects_bad_fractions():
    df = _synthetic_df()
    with pytest.raises(ValueError):
        dp.split_by_match(df, val_frac=0.6, test_frac=0.6)


# --- categorical encoding ------------------------------------------------
def test_category_maps_reserve_zero_for_unknown():
    df = _synthetic_df()
    df.loc[0, "body_part_id"] = 38
    maps = dp.build_category_maps(df, cols=["body_part_id"])
    assert 0 not in maps["body_part_id"].values()
    assert min(maps["body_part_id"].values()) == 1

    encoded = dp.encode_categoricals(df, {"body_part_id": maps["body_part_id"]})
    assert "body_part_id_idx" in encoded.columns
    # a value absent from the map encodes to 0
    df2 = df.copy()
    df2["body_part_id"] = 999
    encoded2 = dp.encode_categoricals(df2, {"body_part_id": maps["body_part_id"]})
    assert (encoded2["body_part_id_idx"] == 0).all()
