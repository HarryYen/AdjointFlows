#!/usr/bin/env python3
"""Prepare fake observed waveforms for checkerboard synthetic tests.

This script copies synthetic SAC files produced from a perturbed model into a
DATA-like directory tree so the existing inversion workflow can treat them as
"observed" waveforms.

Default use:
    1. Edit the USER PARAMETERS block below.
    2. Run: python scripts/prepare_checkerboard_data.py

Optional fallback:
    python scripts/prepare_checkerboard_data.py --help
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


OBSERVED_COMPONENT_MAP = {
    "E": "HHE",
    "N": "HHN",
    "Z": "HHZ",
}


# --------------------
# USER PARAMETERS
# --------------------
# Edit these values directly when preparing a new checkerboard dataset.
DEFAULT_CONFIG = {
    "source_syn_dir": "TOMO/m999/SYN_EQ_TRUE",
    "event_list": "DATA/evlst/fwi_new_cat_version4.txt",
    "output_data_dir": "DATA/wav_EQ_checkerboard",
    "template_data_dir": "DATA/wav_EQ",
    "station_list": "DATA/stlst/sta_new_remove_western.txt",
    "network": "TW",
    "synthetic_component": "semv",
    "clean_output": True,
    "strict": False,
    "dry_run": True,
}


@dataclass(frozen=True)
class EventInfo:
    name: str
    date: str
    time: str

    @property
    def origin(self) -> datetime:
        return datetime.strptime(f"{self.date} {self.time}", "%Y/%m/%d %H:%M:%S.%f")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy synthetic SAC files into a DATA/wav_EQ-like directory layout "
            "for checkerboard synthetic tests."
        )
    )
    parser.add_argument("--source-syn-dir", help="Source synthetic event tree, e.g. TOMO/m900/SYN_EQ_TRUE")
    parser.add_argument("--event-list", help="Event list used to generate the synthetics")
    parser.add_argument("--output-data-dir", help="Output observed-style data directory, e.g. DATA/wav_EQ_checkerboard")
    parser.add_argument(
        "--template-data-dir",
        help=(
            "Optional existing observed waveform tree used only to reuse exact "
            "filenames when available, e.g. DATA/wav_EQ"
        ),
    )
    parser.add_argument(
        "--station-list",
        help="Optional station list for coverage checks; format: sta lon lat elev",
    )
    parser.add_argument(
        "--network",
        default=None,
        help="Network code used in synthetic filenames (default: TW)",
    )
    parser.add_argument(
        "--synthetic-component",
        default=None,
        help="Synthetic component suffix used in SYN files (default: semv)",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        default=None,
        help="Delete the output directory before copying files",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=None,
        help="Exit with an error if any expected source waveform is missing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Print what would be copied without writing files",
    )
    return parser.parse_args()


def build_runtime_args(cli_args: argparse.Namespace) -> SimpleNamespace:
    """Merge CLI values onto the in-script defaults.

    The script is designed to work out of the box with the USER PARAMETERS
    block above. CLI remains available as an override path when needed.
    """
    runtime = dict(DEFAULT_CONFIG)
    for key, value in vars(cli_args).items():
        if value is not None:
            runtime[key] = value
    return SimpleNamespace(**runtime)


def load_event_list(path: Path) -> list[EventInfo]:
    events: list[EventInfo] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=" ", skipinitialspace=True)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            compact = [item for item in row if item]
            if len(compact) < 3:
                continue
            events.append(EventInfo(name=compact[0], date=compact[1], time=compact[2]))
    return events


def load_station_names(path: Path | None) -> list[str]:
    if path is None:
        return []
    stations: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            stations.append(stripped.split()[0])
    return stations


def build_default_filename(event: EventInfo, station: str, component_letter: str) -> str:
    origin = event.origin
    observed_component = OBSERVED_COMPONENT_MAP[component_letter]
    return (
        f"{station}.{observed_component}."
        f"{origin.year:04d}.{origin.timetuple().tm_yday:03d}."
        f"{origin.hour:02d}.{origin.minute:02d}.sac"
    )


def find_template_name(
    template_event_dir: Path | None,
    station: str,
    component_letter: str,
) -> str | None:
    if template_event_dir is None or not template_event_dir.is_dir():
        return None

    observed_component = OBSERVED_COMPONENT_MAP[component_letter]
    candidates = sorted(template_event_dir.glob(f"{station}.{observed_component}.*.sac"))
    if candidates:
        return candidates[0].name
    return None


def find_source_waveform(
    source_event_dir: Path,
    network: str,
    station: str,
    component_letter: str,
    synthetic_component: str,
) -> Path | None:
    component_candidates = [
        f"BX{component_letter}",
        f"BH{component_letter}",
    ]
    suffix_candidates = [
        f"{synthetic_component}.convolved.sac",
        f"{synthetic_component}.sac",
    ]

    for channel in component_candidates:
        for suffix in suffix_candidates:
            candidate = source_event_dir / f"{network}.{station}.{channel}.{suffix}"
            if candidate.is_file():
                return candidate
    return None


def discover_station_names(
    source_event_dir: Path,
    requested_stations: list[str],
    network: str,
) -> list[str]:
    if requested_stations:
        return requested_stations

    stations = set()
    prefix = f"{network}."
    for path in source_event_dir.iterdir():
        if not path.is_file():
            continue
        name = path.name
        if not name.startswith(prefix):
            continue
        parts = name.split(".")
        if len(parts) < 4:
            continue
        stations.add(parts[1])
    return sorted(stations)


def ensure_output_dir(path: Path, clean_output: bool, dry_run: bool) -> None:
    if clean_output and path.exists() and not dry_run:
        shutil.rmtree(path)
    if not dry_run:
        path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    cli_args = parse_args()
    args = build_runtime_args(cli_args)

    source_syn_dir = Path(args.source_syn_dir).resolve()
    event_list_path = Path(args.event_list).resolve()
    output_data_dir = Path(args.output_data_dir).resolve()
    template_data_dir = Path(args.template_data_dir).resolve() if args.template_data_dir else None
    station_list_path = Path(args.station_list).resolve() if args.station_list else None

    if not source_syn_dir.is_dir():
        raise FileNotFoundError(f"Source synthetic directory not found: {source_syn_dir}")
    if not event_list_path.is_file():
        raise FileNotFoundError(f"Event list not found: {event_list_path}")
    if template_data_dir is not None and not template_data_dir.is_dir():
        raise FileNotFoundError(f"Template data directory not found: {template_data_dir}")
    if station_list_path is not None and not station_list_path.is_file():
        raise FileNotFoundError(f"Station list not found: {station_list_path}")

    events = load_event_list(event_list_path)
    station_names = load_station_names(station_list_path)

    ensure_output_dir(output_data_dir, clean_output=args.clean_output, dry_run=args.dry_run)

    copied_count = 0
    missing_sources: list[str] = []
    empty_events: list[str] = []

    for event in events:
        source_event_dir = source_syn_dir / event.name
        if not source_event_dir.is_dir():
            empty_events.append(event.name)
            continue

        event_station_names = discover_station_names(
            source_event_dir=source_event_dir,
            requested_stations=station_names,
            network=args.network,
        )
        if not event_station_names:
            empty_events.append(event.name)
            continue

        output_event_dir = output_data_dir / event.name
        template_event_dir = (template_data_dir / event.name) if template_data_dir else None
        if not args.dry_run:
            output_event_dir.mkdir(parents=True, exist_ok=True)

        for station in event_station_names:
            for component_letter in ("E", "N", "Z"):
                source_file = find_source_waveform(
                    source_event_dir=source_event_dir,
                    network=args.network,
                    station=station,
                    component_letter=component_letter,
                    synthetic_component=args.synthetic_component,
                )
                if source_file is None:
                    missing_sources.append(f"{event.name}:{station}:{component_letter}")
                    continue

                output_name = find_template_name(
                    template_event_dir=template_event_dir,
                    station=station,
                    component_letter=component_letter,
                )
                if output_name is None:
                    output_name = build_default_filename(
                        event=event,
                        station=station,
                        component_letter=component_letter,
                    )

                destination = output_event_dir / output_name
                if args.dry_run:
                    print(f"DRY-RUN copy {source_file} -> {destination}")
                else:
                    shutil.copy2(source_file, destination)
                copied_count += 1

    print(f"Copied waveform files: {copied_count}")
    print(f"Events in event list: {len(events)}")
    print(f"Events without usable synthetic folders: {len(empty_events)}")
    if empty_events:
        preview = ", ".join(empty_events[:10])
        print(f"Empty/missing events preview: {preview}")

    print(f"Missing station-component waveforms: {len(missing_sources)}")
    if missing_sources:
        preview = ", ".join(missing_sources[:10])
        print(f"Missing waveform preview: {preview}")

    if args.strict and (missing_sources or empty_events):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
