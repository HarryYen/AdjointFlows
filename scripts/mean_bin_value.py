import argparse
import os
import sys

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute the mean value of a SPECFEM-style binary kernel file."
    )
    parser.add_argument("bin_path", help="Path to the .bin file.")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Binary data type.",
    )
    parser.add_argument(
        "--keep-padding",
        action="store_true",
        help="Include the first and last padding values in the statistics.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    bin_path = os.path.abspath(args.bin_path)
    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"Binary file not found: {bin_path}")

    dtype = np.float32 if args.dtype == "float32" else np.float64
    values = np.fromfile(bin_path, dtype=dtype)
    if values.size == 0:
        raise ValueError(f"Binary file is empty: {bin_path}")

    data = values if args.keep_padding or values.size < 3 else values[1:-1]
    if data.size == 0:
        raise ValueError(f"No payload values found in: {bin_path}")

    print(f"file: {bin_path}")
    print(f"dtype: {args.dtype}")
    print(f"count: {data.size}")
    print(f"mean: {float(np.mean(data)):.6e}")
    print(f"mean_abs: {float(np.mean(np.abs(data))):.6e}")
    print(f"min: {float(np.min(data)):.6e}")
    print(f"max: {float(np.max(data)):.6e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
