#!/usr/bin/env python3
"""List top-N largest misfit station pairs from window_chi files.

This follows the same measurement logic in utils/misift_histogram.py:
- Keep rows where tr_chi (col 28) != 0 OR am_chi (col 29) != 0
- Use mt_dt (col 12) when mt_dt != 0 else use xc_dt (col 14)
- Use mt_dlna (col 13) when mt_dt != 0 else use xc_dlna (col 15)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

import numpy as np
import pandas as pd


COL_FLAB = 0
COL_STA = 1
COL_NET = 2
COL_CMP = 3
COL_MT_DT = 12
COL_MT_DLNA = 13
COL_XC_DT = 14
COL_XC_DLNA = 15
COL_TR_CHI = 28
COL_AM_CHI = 29


def find_window_chi_files(measure_dir: Path) -> List[Path]:
    """Find window_chi files under MEASURE_{dataset}.

    Priority path is */adjoints/*/window_chi (current workflow style).
    Fallback is recursive search for any file named window_chi.
    """
    prioritized = sorted(measure_dir.glob("adjoints/*/window_chi"))
    if prioritized:
        return prioritized
    return sorted(measure_dir.rglob("window_chi"))


def load_rows(window_chi: Path) -> pd.DataFrame:
    df = pd.read_csv(window_chi, sep=r"\s+", header=None)

    # same filtering as generate_misfit_array
    df = df[(df.iloc[:, COL_TR_CHI] != 0) | (df.iloc[:, COL_AM_CHI] != 0)].copy()
    if df.empty:
        return df

    mt_dt = df.iloc[:, COL_MT_DT].to_numpy()
    mt_dlna = df.iloc[:, COL_MT_DLNA].to_numpy()
    xc_dt = df.iloc[:, COL_XC_DT].to_numpy()
    xc_dlna = df.iloc[:, COL_XC_DLNA].to_numpy()

    df["dt"] = np.where(mt_dt != 0, mt_dt, xc_dt)
    df["dlna"] = np.where(mt_dt != 0, mt_dlna, xc_dlna)
    df["chi"] = df.iloc[:, COL_TR_CHI]

    # event id is parent folder for .../adjoints/{event}/window_chi
    # for other layouts, use immediate parent as best effort.
    if window_chi.parent.name != "adjoints":
        event_id = window_chi.parent.name
    else:
        event_id = "UNKNOWN"

    # if path looks like .../adjoints/{event}/window_chi, use {event}
    if window_chi.parent.parent.name == "adjoints":
        event_id = window_chi.parent.name

    df["event"] = event_id
    df["window_chi_file"] = str(window_chi)
    df["station_component"] = df.iloc[:, COL_FLAB].astype(str)
    df["station"] = df.iloc[:, COL_STA].astype(str)
    df["network"] = df.iloc[:, COL_NET].astype(str)
    df["component"] = df.iloc[:, COL_CMP].astype(str)

    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Find top-N largest misfit station pairs from TOMO/<model>/MEASURE_<dataset>."
        )
    )
    parser.add_argument("--tomo-dir", type=Path, default=Path("TOMO"), help="TOMO root path")
    parser.add_argument("--model", required=True, help="Model name, e.g. m000")
    parser.add_argument("--dataset", required=True, help="Dataset tag used in MEASURE_{dataset}")
    parser.add_argument("--top", type=int, default=10, help="Top N rows to show (default: 10)")
    parser.add_argument(
        "--per-window",
        action="store_true",
        help="Rank each window directly (default groups by event+station_component and keeps max chi per pair)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for saving the top-N result",
    )

    args = parser.parse_args()

    measure_dir = args.tomo_dir / args.model / f"MEASURE_{args.dataset}"
    if not measure_dir.exists():
        print(f"[ERROR] Measure directory not found: {measure_dir}", file=sys.stderr)
        return 1

    files = find_window_chi_files(measure_dir)
    if not files:
        print(f"[ERROR] No window_chi found under: {measure_dir}", file=sys.stderr)
        return 1

    all_rows = []
    for wf in files:
        try:
            df = load_rows(wf)
        except Exception as exc:  # keep simple and robust for batch scanning
            print(f"[WARN] Skip unreadable file: {wf} ({exc})", file=sys.stderr)
            continue
        if not df.empty:
            all_rows.append(df)

    if not all_rows:
        print("[INFO] No valid misfit rows after filtering.")
        return 0

    merged = pd.concat(all_rows, ignore_index=True)
    merged = merged.sort_values("chi", ascending=False)
    if args.per_window:
        ranked = merged
    else:
        # Keep one row per event-station pair (max chi), then rank pairs.
        ranked = merged.drop_duplicates(subset=["event", "station_component"], keep="first")
    top_df = ranked.head(args.top).copy()
    top_df.insert(0, "rank", np.arange(1, len(top_df) + 1))

    cols = [
        "rank",
        "chi",
        "dt",
        "dlna",
        "event",
        "station_component",
        "station",
        "network",
        "component",
        "window_chi_file",
    ]
    top_view = top_df.loc[:, cols]

    print(f"Measure dir: {measure_dir}")
    print(f"window_chi files scanned: {len(files)}")
    print(f"valid rows after filter: {len(merged)}")
    if args.per_window:
        print(f"top {len(top_view)} windows by chi (col 28):")
    else:
        print(f"unique event-station pairs: {len(ranked)}")
        print(f"top {len(top_view)} station pairs by chi (col 28):")
    print(top_view.to_string(index=False, justify="left"))

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        top_view.to_csv(args.output_csv, index=False)
        print(f"Saved CSV: {args.output_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
