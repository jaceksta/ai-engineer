"""Download and cache a subset of StatsBomb Open Data.

StatsBomb Open Data lives in a public GitHub repo:
    https://github.com/statsbomb/open-data

Raw files are served from raw.githubusercontent.com with this layout
(verified against the live repo):

    data/competitions.json
    data/matches/{competition_id}/{season_id}.json
    data/events/{match_id}.json
    data/lineups/{match_id}.json
    data/three-sixty/{match_id}.json   # selected matches only, not fetched here

This script mirrors the pieces you need into ``data/raw/`` with the same
sub-paths, so ``src/data_prep.py`` can read them straight off disk.

Defaults to the 2018 FIFA World Cup (competition_id=43, season_id=3): 64
matches, ~1700 shots -- plenty for the exercises and small enough for a
laptop.

Usage
-----
    uv run python data/download_statsbomb.py                 # default subset
    uv run python data/download_statsbomb.py --matches 10    # first 10 matches
    uv run python data/download_statsbomb.py --competition-id 11 --season-id 90
    uv run python data/download_statsbomb.py --force         # re-download everything

Attribution: StatsBomb ask that any published analysis based on this data
names StatsBomb as the source. Keep that in mind for your write-ups.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

RAW_BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
DEFAULT_RAW_DIR = Path(__file__).resolve().parent / "raw"

DEFAULT_COMPETITION_ID = 43  # FIFA World Cup
DEFAULT_SEASON_ID = 3  # 2018

# Be a polite guest on a free public host.
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.1


class DownloadError(RuntimeError):
    """Raised for any unrecoverable problem fetching the data."""


def _proxy_hint() -> str:
    return (
        "\nIf you are on a corporate network you almost certainly need a proxy.\n"
        "Copy .env.example to .env and set HTTP_PROXY / HTTPS_PROXY, then re-run.\n"
        "(requests picks these up from the environment automatically.)"
    )


def _get(session: requests.Session, url: str) -> bytes:
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.ProxyError as exc:
        raise DownloadError(f"Proxy rejected the request for {url}: {exc}{_proxy_hint()}") from exc
    except requests.exceptions.SSLError as exc:
        raise DownloadError(
            f"TLS/SSL error fetching {url}: {exc}\n"
            "A corporate TLS-inspection proxy can cause this; you may need to point "
            "REQUESTS_CA_BUNDLE at your company's CA certificate." + _proxy_hint()
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise DownloadError(f"Could not connect to {url}: {exc}{_proxy_hint()}") from exc
    except requests.exceptions.RequestException as exc:
        raise DownloadError(f"Request failed for {url}: {exc}") from exc

    if response.status_code == 404:
        raise DownloadError(
            f"404 Not Found: {url}\n"
            "Check the competition_id / season_id / match_id -- not every "
            "combination exists in the open data."
        )
    if response.status_code != 200:
        raise DownloadError(f"HTTP {response.status_code} for {url}")
    return response.content


def _fetch_to_file(session: requests.Session, rel_path: str, raw_dir: Path, force: bool) -> Path:
    """Download ``RAW_BASE/rel_path`` to ``raw_dir/rel_path`` unless cached."""
    dest = raw_dir / rel_path
    if dest.exists() and not force:
        return dest

    content = _get(session, f"{RAW_BASE}/{rel_path}")
    # Validate it parses as JSON before writing, so a truncated download does
    # not leave a poisoned cache file.
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        raise DownloadError(f"{rel_path} did not parse as JSON: {exc}") from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    time.sleep(SLEEP_BETWEEN_REQUESTS)
    return dest


def _load_match_ids(matches_path: Path, limit: int | None) -> list[int]:
    matches = json.loads(matches_path.read_text(encoding="utf-8"))
    match_ids = sorted(int(m["match_id"]) for m in matches)
    if limit is not None:
        match_ids = match_ids[:limit]
    return match_ids


def download(
    competition_id: int,
    season_id: int,
    raw_dir: Path,
    limit: int | None = None,
    match_ids: list[int] | None = None,
    with_lineups: bool = True,
    force: bool = False,
) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "ai-engineer-study-repo/0.1 (learning project)"})

    print("competitions.json ...")
    _fetch_to_file(session, "competitions.json", raw_dir, force)

    if match_ids is None:
        print(f"matches/{competition_id}/{season_id}.json ...")
        matches_path = _fetch_to_file(
            session, f"matches/{competition_id}/{season_id}.json", raw_dir, force
        )
        match_ids = _load_match_ids(matches_path, limit)

    print(f"{len(match_ids)} matches -> events" + (" + lineups" if with_lineups else ""))
    downloaded = 0
    for match_id in tqdm(match_ids, unit="match"):
        _fetch_to_file(session, f"events/{match_id}.json", raw_dir, force)
        if with_lineups:
            _fetch_to_file(session, f"lineups/{match_id}.json", raw_dir, force)
        downloaded += 1

    print(f"\nDone. {downloaded} matches cached under {raw_dir}")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--competition-id", type=int, default=DEFAULT_COMPETITION_ID)
    parser.add_argument("--season-id", type=int, default=DEFAULT_SEASON_ID)
    parser.add_argument(
        "--matches",
        type=int,
        default=None,
        metavar="N",
        help="Only download the first N matches of the season (default: all).",
    )
    parser.add_argument(
        "--match-ids",
        type=int,
        nargs="+",
        default=None,
        help="Explicit match IDs to download (skips the matches listing).",
    )
    parser.add_argument("--no-lineups", action="store_true", help="Skip lineup files.")
    parser.add_argument(
        "--force", action="store_true", help="Re-download files that are already cached."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Where to cache files (default: {DEFAULT_RAW_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # pick up HTTP_PROXY / HTTPS_PROXY from .env if present
    args = _parse_args(argv)
    try:
        download(
            competition_id=args.competition_id,
            season_id=args.season_id,
            raw_dir=args.raw_dir,
            limit=args.matches,
            match_ids=args.match_ids,
            with_lineups=not args.no_lineups,
            force=args.force,
        )
    except DownloadError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
