#!/usr/bin/env python3
"""Validate and summarize adjointflows/dataset.yaml."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "adjointflows"))

from tools.dataset_loader import load_dataset_config, get_by_path


def validate_datasets(data: dict) -> list[str]:
    """Validate datasets structure and return dataset names.

    Args:
        data: Parsed dataset configuration.

    Returns:
        List of dataset names.
    """
    datasets = data.get("datasets", [])
    if not isinstance(datasets, list):
        raise ValueError("'datasets' must be a list.")
    if not datasets:
        raise ValueError("No datasets found under 'datasets'.")

    names = []
    for index, entry in enumerate(datasets, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset #{index} must be a mapping.")
        name = entry.get("name")
        if not name:
            raise ValueError(f"Dataset #{index} missing required 'name'.")
        names.append(str(name))
    if len(set(names)) != len(names):
        raise ValueError("Duplicate dataset names found.")
    return names


def deep_merge(base, override):
    """Deep-merge two dictionaries with override taking precedence."""
    if not isinstance(base, dict):
        return override
    if not isinstance(override, dict):
        return override
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def main() -> int:
    """Run dataset.yaml validation."""
    parser = argparse.ArgumentParser(description="Validate and summarize dataset.yaml")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to dataset.yaml (default: adjointflows/dataset.yaml)",
    )
    parser.add_argument(
        "--show-merged",
        action="store_true",
        help="Print merged dataset configs with defaults applied.",
    )
    args = parser.parse_args()

    try:
        if args.path is None:
            repo_root = Path(__file__).resolve().parents[1]
            data = load_dataset_config(str(repo_root / "adjointflows"))
        else:
            data = load_dataset_config(str(args.path.parent))
        names = validate_datasets(data)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    if args.path is None:
        repo_root = Path(__file__).resolve().parents[1]
        dataset_path = repo_root / "adjointflows" / "dataset.yaml"
    else:
        dataset_path = args.path
    datasets = data.get("datasets", [])
    default_settings = get_by_path(data, "defaults.seismogram.tbeg")


    defaults = data.get("defaults", {})
    print("merged_datasets:")
    for dataset in datasets:
        merged = deep_merge(defaults, dataset)
        print(merged)
        print(get_by_path(merged, "seismogram.tbeg"))
        sys.exit(0)
        # print(json.dumps(merged, indent=2, ensure_ascii=True))

    print(default_settings)
    # print(f"OK: {dataset_path}")
    # print(f"datasets: {len(names)}")
    # for dataset in datasets:
    #     merge_data = deep_merge(defaults, dataset)
    #     print(merge_data)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# 
