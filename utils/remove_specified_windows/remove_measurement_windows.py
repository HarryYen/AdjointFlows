#!/usr/bin/env python3
import argparse
import json
import os
import re
from collections import defaultdict


PAIR_RE = re.compile(r"DATA/([^./]+)\.([^./]+)\.([^./]+)\.([^./]+)\.")


def parse_pairs(lines):
    if not lines:
        raise ValueError("Empty MEASUREMENT.WINDOWS file.")
    try:
        total_pairs = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError("Invalid pair count on first line.") from exc

    pairs = []
    idx = 1
    while idx < len(lines):
        if idx + 2 >= len(lines):
            raise ValueError("Incomplete pair header in MEASUREMENT.WINDOWS.")
        data_line = lines[idx]
        syn_line = lines[idx + 1]
        try:
            win_count = int(lines[idx + 2].strip())
        except ValueError as exc:
            raise ValueError(f"Invalid window count line: {lines[idx + 2].strip()}") from exc
        win_start = idx + 3
        win_end = win_start + win_count
        if win_end > len(lines):
            raise ValueError("Window count exceeds file length.")
        windows = lines[win_start:win_end]
        pairs.append(
            {
                "data_line": data_line,
                "syn_line": syn_line,
                "windows": windows,
            }
        )
        idx = win_end

    if total_pairs != len(pairs):
        raise ValueError(
            f"Pair count mismatch: header {total_pairs}, parsed {len(pairs)}."
        )
    return pairs


def extract_pair_key(data_line):
    match = PAIR_RE.search(data_line)
    if not match:
        return None
    event, network, station, component = match.groups()
    return event, station, network, component


def apply_deletions(pairs, deletions):
    removed = 0
    for pair in pairs:
        key = extract_pair_key(pair["data_line"])
        if not key or key not in deletions:
            continue
        indices = deletions[key]
        if not indices:
            continue
        # Delete in descending order so indices stay valid.
        for idx in sorted(indices, reverse=True):
            if idx < 0 or idx >= len(pair["windows"]):
                print(
                    f"Warning: window_index {idx + 1} out of range for {key}."
                )
                continue
            del pair["windows"][idx]
            removed += 1
    # Remove empty pairs.
    pairs[:] = [p for p in pairs if p["windows"]]
    return removed


def write_pairs(path, pairs):
    lines = [f"{len(pairs)}\n"]
    for pair in pairs:
        lines.append(pair["data_line"] if pair["data_line"].endswith("\n") else pair["data_line"] + "\n")
        lines.append(pair["syn_line"] if pair["syn_line"].endswith("\n") else pair["syn_line"] + "\n")
        lines.append(f"{len(pair['windows']):11d}\n")
        for win_line in pair["windows"]:
            lines.append(win_line if win_line.endswith("\n") else win_line + "\n")
    with open(path, "w") as f:
        f.writelines(lines)


def load_operations(config):
    windows_root = config.get("windows_root")
    ops = config.get("operations", [])
    if not ops:
        ops = [config]
    normalized = []
    for op in ops:
        root = op.get("windows_root", windows_root)
        if not root:
            raise ValueError("windows_root is required.")
        window_index = op.get("window_index")
        if window_index is None:
            raise ValueError("window_index is required.")
        if isinstance(window_index, int):
            indices = [window_index]
        else:
            indices = list(window_index)
        normalized.append(
            {
                "windows_root": root,
                "event": op["event"],
                "station": op["station"],
                "network": op["network"],
                "component": op["component"],
                "indices": indices,
            }
        )
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config.")
    parser.add_argument("--dry-run", action="store_true", help="Print changes only.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    ops = load_operations(config)
    ops_by_file = defaultdict(list)
    for op in ops:
        windows_file = os.path.join(
            op["windows_root"], op["event"], "MEASUREMENT.WINDOWS"
        )
        key = (op["event"], op["station"], op["network"], op["component"])
        indices = [i - 1 for i in op["indices"]]
        ops_by_file[windows_file].append((key, indices))

    for windows_file, deletions in ops_by_file.items():
        if not os.path.isfile(windows_file):
            raise FileNotFoundError(f"MEASUREMENT.WINDOWS not found: {windows_file}")
        with open(windows_file, "r") as f:
            lines = f.readlines()
        pairs = parse_pairs(lines)
        delete_map = defaultdict(set)
        for key, indices in deletions:
            for idx in indices:
                delete_map[key].add(idx)
        removed = apply_deletions(pairs, delete_map)
        if args.dry_run:
            print(f"{windows_file}: would remove {removed} windows.")
            continue
        write_pairs(windows_file, pairs)
        print(f"{windows_file}: removed {removed} windows.")


if __name__ == "__main__":
    main()
