"""Small helpers for summing SPECFEM rank-wise kernel binaries.

Edit another script or an interactive session to call ``sum_rank_kernel_files``
with an explicit list of files. This module intentionally has no CLI parser.
"""

from pathlib import Path

import numpy as np
import sys


# =============================================================================
# User settings: edit this block directly, then run this file.
# =============================================================================
BASE_DIR = Path("/home/harry/Work/AdjointFlows_mesh_2_no_smoothed")
TOMO_DIR = BASE_DIR / "TOMO"

# Example: [26, 27, 28]. Keep one unified NPROC group per run.
MODEL_NUM_LIST = [22, 23, 24, 25]

# Example input directory:
# TOMO/m026/KERNEL_EGF_5_12s/PRECOND/proc000000_alpha_kernel_smooth.bin
KERNEL_DIR_NAME = "KERNEL"
KERNEL_SUBDIR = "PRECOND"

KERNEL_NAMES = [
    "alpha_kernel_smooth",
    "beta_kernel_smooth",
    "rho_kernel_smooth",
]

OUTPUT_DIR = TOMO_DIR / "MASK_KERNEL_SUM" / KERNEL_DIR_NAME / KERNEL_SUBDIR

# Set to None to detect from the first model directory. Or set explicitly, e.g. 1, 2, 4.
NPROC = None

# "abs" is usually better for resolution/illumination masks because signed
# kernels can cancel each other. Use "raw" if you need signed sums.
SUM_MODE = "abs"

DTYPE = np.float32


def make_rank_kernel_name(rank, kernel_name):
    """Return a SPECFEM rank-wise kernel filename."""
    return f"proc{rank:06d}_{kernel_name}.bin"


def detect_nproc_from_kernel_dir(kernel_dir, kernel_name):
    """Detect NPROC from contiguous rank files in one kernel directory.

    Args:
        kernel_dir (str | Path): Directory containing rank-wise kernel files.
        kernel_name (str): Kernel name without the ``proc000000_`` prefix and
            without the ``.bin`` suffix, for example
            ``"alpha_kernel_smooth"``.

    Returns:
        int: Number of contiguous rank files found from rank 0.
    """
    kernel_dir = Path(kernel_dir)
    if not kernel_dir.is_dir():
        raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")

    nproc = 0
    while (kernel_dir / make_rank_kernel_name(nproc, kernel_name)).is_file():
        nproc += 1

    if nproc == 0:
        expected_file = kernel_dir / make_rank_kernel_name(0, kernel_name)
        raise FileNotFoundError(f"No rank kernel files found. Expected: {expected_file}")

    return nproc


def check_kernel_dirs_have_same_nproc(kernel_dirs, kernel_name, nproc):
    """Verify all kernel directories contain the same contiguous ranks."""
    missing_files = []
    extra_files = []

    for kernel_dir in kernel_dirs:
        kernel_dir = Path(kernel_dir)
        for rank in range(nproc):
            kernel_file = kernel_dir / make_rank_kernel_name(rank, kernel_name)
            if not kernel_file.is_file():
                missing_files.append(kernel_file)

        next_rank_file = kernel_dir / make_rank_kernel_name(nproc, kernel_name)
        if next_rank_file.is_file():
            extra_files.append(next_rank_file)

    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing expected rank files:\n{missing_text}")

    if extra_files:
        extra_text = "\n".join(str(path) for path in extra_files)
        raise ValueError(
            f"Found rank files beyond detected nproc={nproc}. "
            f"Kernel dirs may use different NPROC:\n{extra_text}"
        )


def read_kernel_bin(kernel_file, dtype=np.float32):
    """Read one SPECFEM kernel binary and return payload plus padding value.

    SPECFEM binary kernels in this workflow store one padding value at the
    beginning and one at the end. The returned array excludes those padding
    values, matching ``adjointflows.tools.matrix_utils.read_bin``.
    """
    kernel_file = Path(kernel_file)
    if not kernel_file.is_file():
        raise FileNotFoundError(f"Kernel file not found: {kernel_file}")

    values = np.fromfile(kernel_file, dtype=dtype)
    if values.size < 3:
        raise ValueError(f"Kernel file is too small to contain payload: {kernel_file}")

    padding_value = values[0]
    end_padding_value = values[-1]
    if not np.isclose(padding_value, end_padding_value):
        raise ValueError(
            f"Padding values differ in {kernel_file}: "
            f"{padding_value} != {end_padding_value}"
        )

    return values[1:-1], padding_value


def write_kernel_bin(kernel_values, output_file, padding_value, dtype=np.float32):
    """Write payload values back to a SPECFEM-style padded binary file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    kernel_values = np.asarray(kernel_values, dtype=dtype)
    padded_values = np.pad(
        kernel_values,
        (1, 1),
        mode="constant",
        constant_values=padding_value,
    )
    padded_values.astype(dtype, copy=False).tofile(output_file)
    return output_file


def get_kernel_contribution(kernel_values, mode):
    """Return the contribution used in the summation."""
    if mode == "raw":
        return kernel_values
    if mode == "abs":
        return np.abs(kernel_values)
    raise ValueError(f"Unsupported sum mode: {mode}. Use 'raw' or 'abs'.")


def sum_rank_kernel_files(kernel_files, output_file, dtype=np.float32, mode="abs"):
    """Sum same-rank, same-scalar kernel binaries and write one summed file.

    Args:
        kernel_files (list[str | Path]): Explicit list of input kernel files.
            These should all be the same MPI rank and same scalar, for example
            ``proc000000_alpha_kernel_smooth.bin`` from different datasets.
        output_file (str | Path): Destination summed kernel binary.
        dtype: NumPy dtype used by the SPECFEM binary files.
        mode (str): ``"abs"`` sums absolute values; ``"raw"`` sums signed values.

    Returns:
        Path: Path to the written summed kernel file.
    """
    if not kernel_files:
        raise ValueError("kernel_files must contain at least one file.")

    summed_kernel = None
    padding_value = None
    expected_size = None

    for kernel_file in kernel_files:
        kernel_values, file_padding = read_kernel_bin(kernel_file, dtype=dtype)

        if expected_size is None:
            expected_size = kernel_values.size
            padding_value = file_padding
            summed_kernel = np.zeros_like(kernel_values, dtype=dtype)
        elif kernel_values.size != expected_size:
            raise ValueError(
                f"Kernel size mismatch for {kernel_file}: "
                f"{kernel_values.size} != {expected_size}"
            )

        contribution = get_kernel_contribution(kernel_values, mode)
        summed_kernel += contribution.astype(dtype, copy=False)

    return write_kernel_bin(summed_kernel, output_file, padding_value, dtype=dtype)


def sum_kernel_dirs(
    kernel_dirs,
    output_dir,
    kernel_name,
    nproc=None,
    dtype=np.float32,
    mode="abs",
):
    """Sum one kernel name across directories using a unified NPROC.

    Args:
        kernel_dirs (list[str | Path]): Directories containing rank-wise kernel
            files from the same mesh partitioning.
        output_dir (str | Path): Directory where summed rank files are written.
        kernel_name (str): Kernel name without ``proc000000_`` and ``.bin``.
            Example: ``"alpha_kernel_smooth"``.
        nproc (int | None): Unified NPROC. If ``None``, detect it from the
            first kernel directory.
        dtype: NumPy dtype used by the SPECFEM binary files.
        mode (str): ``"abs"`` sums absolute values; ``"raw"`` sums signed values.

    Returns:
        list[Path]: Written summed rank kernel files.
    """
    if not kernel_dirs:
        raise ValueError("kernel_dirs must contain at least one directory.")

    kernel_dirs = [Path(kernel_dir) for kernel_dir in kernel_dirs]
    output_dir = Path(output_dir)

    if nproc is None:
        nproc = detect_nproc_from_kernel_dir(kernel_dirs[0], kernel_name)
    if nproc < 1:
        raise ValueError(f"nproc must be >= 1, got {nproc}")

    check_kernel_dirs_have_same_nproc(kernel_dirs, kernel_name, nproc)

    output_files = []
    for rank in range(nproc):
        rank_file_name = make_rank_kernel_name(rank, kernel_name)
        input_files = [kernel_dir / rank_file_name for kernel_dir in kernel_dirs]
        output_file = output_dir / rank_file_name
        output_files.append(
            sum_rank_kernel_files(input_files, output_file, dtype=dtype, mode=mode)
        )

    return output_files


def build_kernel_dirs(tomo_dir, model_num_list, kernel_dir_name, kernel_subdir):
    """Build kernel directories from model numbers and config names."""
    if not model_num_list:
        raise ValueError("MODEL_NUM_LIST is empty. Edit the config block before running.")

    return [
        Path(tomo_dir) / f"m{model_num:03d}" / kernel_dir_name / kernel_subdir
        for model_num in model_num_list
    ]


def main():
    """Sum configured kernel names across model kernel directories."""
    kernel_dirs = build_kernel_dirs(
        tomo_dir=TOMO_DIR,
        model_num_list=MODEL_NUM_LIST,
        kernel_dir_name=KERNEL_DIR_NAME,
        kernel_subdir=KERNEL_SUBDIR,
    )

    print("Kernel directories:")
    for kernel_dir in kernel_dirs:
        print(f"  {kernel_dir}")
    print(f"Output directory: {OUTPUT_DIR}")


    for kernel_name in KERNEL_NAMES:
        nproc = NPROC
        if nproc is None:
            nproc = detect_nproc_from_kernel_dir(kernel_dirs[0], kernel_name)

        print(f"Summing {kernel_name} with nproc={nproc}, mode={SUM_MODE}")
        
        output_files = sum_kernel_dirs(
            kernel_dirs=kernel_dirs,
            output_dir=OUTPUT_DIR,
            kernel_name=kernel_name,
            nproc=nproc,
            dtype=DTYPE,
            mode=SUM_MODE,
        )
        for output_file in output_files:
            print(f"  wrote {output_file}")


if __name__ == "__main__":
    main()
