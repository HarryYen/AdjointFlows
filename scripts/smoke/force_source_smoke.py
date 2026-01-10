#!/usr/bin/env python3
"""
Smoke test for force-source generation.

This writes specfem3d/DATA/FORCESOLUTION and a dummy CMTSOLUTION
based on one virtual station entry, without running SPECFEM.
"""
from pathlib import Path
import argparse
import sys


def parse_station_list(path, station_name=None):
    entries = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 4:
                raise ValueError(f"Invalid station line: {stripped}")
            name, lon, lat, elev = parts[0], parts[1], parts[2], parts[3]
            entries.append({
                "name": name,
                "lon": float(lon),
                "lat": float(lat),
                "elev": float(elev),
            })

    if not entries:
        raise ValueError(f"No stations found in {path}")

    if station_name:
        for entry in entries:
            if entry["name"] == station_name:
                return entry
        raise ValueError(f"Station '{station_name}' not found in {path}")

    return entries[0]


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.append(str(root / "adjointflows"))

    parser = argparse.ArgumentParser(description="Smoke test for force-source generation")
    parser.add_argument(
        "--config",
        default=str(root / "adjointflows" / "config.yaml"),
        help="Path to adjointflows config.yaml",
    )
    parser.add_argument(
        "--station-list",
        default=str(root / "codex_demo" / "sta_demo.txt"),
        help="Virtual station list (sta lon lat elev)",
    )
    parser.add_argument(
        "--station-name",
        default=None,
        help="Optional station name to select (defaults to first line)",
    )
    parser.add_argument(
        "--depth-km",
        type=float,
        default=None,
        help="Override source.force.depth_km for this test",
    )
    args = parser.parse_args()

    from tools import ConfigManager
    from kernel.forward import ForwardGenerator

    cfg = ConfigManager(args.config)
    cfg.load()

    if cfg.config is None:
        raise RuntimeError("Config did not load")

    cfg.config.setdefault("source", {})["type"] = "force"
    cfg.config.setdefault("source", {}).setdefault("force", {})["auto_set_par_file"] = 0
    if args.depth_km is not None:
        cfg.config["source"].setdefault("force", {})["depth_km"] = args.depth_km

    fg = ForwardGenerator(current_model_num=0, config=cfg)
    source_info = parse_station_list(args.station_list, args.station_name)
    fg.write_source_files(source_info)

    force_path = root / "specfem3d" / "DATA" / "FORCESOLUTION"
    cmt_path = root / "specfem3d" / "DATA" / "CMTSOLUTION"

    print("--- FORCESOLUTION ---")
    print(force_path.read_text())
    print("--- CMTSOLUTION (dummy) ---")
    print(cmt_path.read_text())


if __name__ == "__main__":
    main()
