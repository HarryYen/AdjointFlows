#!/usr/bin/env python3
"""Validate and summarize adjointflows/dataset.yaml."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def load_dataset_config(path: Path) -> dict:
    """Load dataset.yaml and return the parsed content.

    Args:
        path: Path to dataset.yaml.

    Returns:
        Parsed YAML content as a dictionary.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Dataset file must contain a mapping at the top level.")
    return data


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


def main() -> int:
    """Run dataset.yaml validation."""
    parser = argparse.ArgumentParser(description="Validate and summarize dataset.yaml")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Path to dataset.yaml (default: adjointflows/dataset.yaml)",
    )
    args = parser.parse_args()

    if args.path is None:
        repo_root = Path(__file__).resolve().parents[1]
        dataset_path = repo_root / "adjointflows" / "dataset.yaml"
    else:
        dataset_path = args.path

    try:
        data = load_dataset_config(dataset_path)
        names = validate_datasets(data)
    except (FileNotFoundError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"OK: {dataset_path}")
    print(f"datasets: {len(names)}")
    for name in names:
        print(f"- {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
