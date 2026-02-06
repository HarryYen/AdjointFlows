#!/usr/bin/env python3
import os
import sys
from types import SimpleNamespace


def load_events(event_file):
    events = []
    with open(event_file, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            events.append(stripped.split()[0])
    return events


def read_window_counts(windows_file):
    with open(windows_file, "r") as f:
        lines = f.readlines()
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError("Empty MEASUREMENT.WINDOWS file.")
    try:
        pair_count = int(lines[0].strip().split()[0])
    except ValueError as exc:
        raise ValueError("Invalid pair count on first line.") from exc

    idx = 1
    parsed_pairs = 0
    windows_total = 0
    while idx < len(lines):
        if idx + 2 >= len(lines):
            raise ValueError("Incomplete pair header in MEASUREMENT.WINDOWS.")
        try:
            win_count = int(lines[idx + 2].strip().split()[0])
        except ValueError as exc:
            raise ValueError(f"Invalid window count line: {lines[idx + 2].strip()}") from exc
        windows_total += win_count
        idx += 3 + win_count
        parsed_pairs += 1

    if parsed_pairs != pair_count:
        raise ValueError(
            f"Pair count mismatch: header {pair_count}, parsed {parsed_pairs}."
        )
    return pair_count, windows_total


def normalize_model_dir(model_arg):
    if model_arg is None:
        return None
    model_arg = str(model_arg).strip()
    if not model_arg:
        return None
    if model_arg.isdigit():
        return f"m{int(model_arg):03d}"
    return model_arg


def resolve_windows_root(args):
    if args.windows_root:
        return args.windows_root
    model_dir = normalize_model_dir(args.model)
    if not model_dir:
        raise ValueError("--model is required unless --windows-root is provided.")
    if args.measure_dir:
        measure_dir = args.measure_dir
    elif args.dataset:
        measure_dir = f"MEASURE_{args.dataset}"
    else:
        measure_dir = "MEASURE"
    return os.path.join(args.tomo_dir, model_dir, measure_dir, args.windows_subdir)


def main():
    # User settings (edit these).
    event_file = "DATA/evlst/cmt_evt_test.txt"
    windows_root = None
    tomo_dir = "TOMO"
    model = 16
    dataset = "EQ_10_30s"
    measure_dir = None
    windows_subdir = "windows"
    windows_file = "MEASUREMENT.WINDOWS"
    per_event = False

    if not os.path.isfile(event_file):
        print(f"Event file not found: {event_file}", file=sys.stderr)
        return 2

    try:
        windows_root = resolve_windows_root(
            SimpleNamespace(
                windows_root=windows_root,
                tomo_dir=tomo_dir,
                model=model,
                dataset=dataset,
                measure_dir=measure_dir,
                windows_subdir=windows_subdir,
            )
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    events = load_events(event_file)
    if not events:
        print("No events found in event file.", file=sys.stderr)
        return 2

    total_pairs = 0
    total_windows = 0
    missing = []
    invalid = []
    found = 0

    for event in events:
        win_path = os.path.join(windows_root, event, windows_file)
        if not os.path.isfile(win_path):
            missing.append(event)
            continue
        try:
            pair_count, window_count = read_window_counts(win_path)
        except ValueError as exc:
            invalid.append((event, str(exc)))
            continue
        found += 1
        total_pairs += pair_count
        total_windows += window_count
        if per_event:
            print(f"{event}\tpairs={pair_count}\twindows={window_count}")

    print(f"windows_root: {windows_root}")
    print(f"events_in_list: {len(events)}")
    print(f"windows_files_found: {found}")
    print(f"total_pairs: {total_pairs}")
    print(f"total_windows: {total_windows}")
    if missing:
        print(f"missing_files: {len(missing)}")
        for event in missing:
            print(f"missing\t{event}")
    if invalid:
        print(f"invalid_files: {len(invalid)}")
        for event, reason in invalid:
            print(f"invalid\t{event}\t{reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
