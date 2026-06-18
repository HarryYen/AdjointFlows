"""Project summed SPECFEM kernel binaries to a regular-grid gradient.xyz file.

Edit the config block directly, then run this file. This is the middle step:

summed rank-wise .bin -> VTU -> regular lon/lat/depth xyz
"""

from pathlib import Path
import os
import subprocess
import sys


# =============================================================================
# User settings: edit this block directly, then run this file.
# =============================================================================
BASE_DIR = Path("/home/harry/Work/AdjointFlows_mesh_2_no_smoothed")
TOMO_DIR = BASE_DIR / "TOMO"

# A model with the same mesh/partitioning as the summed kernels. Its
# DATABASES_MPI directory provides the mesh geometry for projection.
MODEL_NUM_FOR_MESH = 22

KERNEL_DIR = TOMO_DIR / "MASK_KERNEL_SUM" / "KERNEL" / "PRECOND"
OUTPUT_DIR = KERNEL_DIR
VTU_OUTPUT_DIR = OUTPUT_DIR / "VTU"
OUTPUT_XYZ_NAME = "gradient.xyz"

KERNEL_NAMES = {
    "alpha": "alpha_kernel_smooth",
    "beta": "beta_kernel_smooth",
    "rho": "rho_kernel_smooth",
}

LON_RANGE = [119.0, 123.0]
LAT_RANGE = [21.0, 26.0]
DEP_RANGE = [0.0, 200.0]
LON_INTERVAL = 0.02
LAT_INTERVAL = 0.02
DEP_INTERVAL = 2.0

UTM_ZONE = 50
IS_NORTH_HEMISPHERE = True


ADJOINTFLOWS_DIR = BASE_DIR / "adjointflows"
if str(ADJOINTFLOWS_DIR) not in sys.path:
    sys.path.append(str(ADJOINTFLOWS_DIR))

from tools import ConfigManager, GLOBAL_PARAMS  # noqa: E402
from visualizer import TomographyVisualizer  # noqa: E402


def convert_kernel_to_vtu(visualizer, kernel_name, vtu_output_dir):
    """Convert one summed kernel to VTU under vtu_output_dir."""
    vtu_output_dir = Path(vtu_output_dir)
    vtu_output_dir.mkdir(parents=True, exist_ok=True)

    target_file = vtu_output_dir / f"{kernel_name}.vtu"
    if target_file.exists():
        target_file.unlink()

    current_dir = Path.cwd()
    os.chdir(visualizer.specfem_dir)
    try:
        subprocess.run(
            [
                "./bin/xcombine_vol_data_vtu",
                "0",
                f"{visualizer.nproc - 1}",
                kernel_name,
                str(visualizer.kernels_dir),
                str(vtu_output_dir),
                "0",
            ],
            check=True,
        )
    finally:
        os.chdir(current_dir)


def project_kernel(visualizer, kernel_name, vtu_output_dir):
    """Convert one kernel to VTU and project it to the configured regular grid."""
    convert_kernel_to_vtu(visualizer, kernel_name, vtu_output_dir)

    original_databases_dir = visualizer.databases_dir
    visualizer.databases_dir = str(vtu_output_dir)
    try:
        values, perturbation = visualizer.project_gll_to_regular(
            kernel_name=kernel_name,
            utm_zone=UTM_ZONE,
            is_north_hemisphere=IS_NORTH_HEMISPHERE,
        )
    finally:
        visualizer.databases_dir = original_databases_dir
    return values, perturbation


def main():
    """Project configured summed kernels and write gradient.xyz."""
    kernel_dir = Path(KERNEL_DIR)
    if not kernel_dir.is_dir():
        raise FileNotFoundError(f"Kernel directory not found: {kernel_dir}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    vtu_output_dir = Path(VTU_OUTPUT_DIR)
    vtu_output_dir.mkdir(parents=True, exist_ok=True)

    current_dir = Path.cwd()
    os.chdir(ADJOINTFLOWS_DIR)
    try:
        config = ConfigManager("config.yaml")
        config.load()

        visualizer = TomographyVisualizer(
            config=config,
            global_params=GLOBAL_PARAMS,
            specified_model_num=MODEL_NUM_FOR_MESH,
        )
        visualizer.load_params()
        visualizer.kernels_dir = str(kernel_dir)
        visualizer.output_dir = str(output_dir)
        visualizer.setup_spatial_range(
            lon_range=LON_RANGE,
            lat_range=LAT_RANGE,
            dep_range=DEP_RANGE,
            lon_interval=LON_INTERVAL,
            lat_interval=LAT_INTERVAL,
            dep_interval=DEP_INTERVAL,
        )

        print(f"Projecting summed kernels from {kernel_dir}")
        print(f"Writing VTU files to {vtu_output_dir}")
        alpha_abs, alpha_pert = project_kernel(
            visualizer, KERNEL_NAMES["alpha"], vtu_output_dir
        )
        beta_abs, beta_pert = project_kernel(
            visualizer, KERNEL_NAMES["beta"], vtu_output_dir
        )
        rho_abs, rho_pert = project_kernel(
            visualizer, KERNEL_NAMES["rho"], vtu_output_dir
        )

        visualizer.output_kernel_txt_file(
            output_file_name=OUTPUT_XYZ_NAME,
            v1_abs=alpha_abs,
            v1_pert=alpha_pert,
            v2_abs=beta_abs,
            v2_pert=beta_pert,
            v3_abs=rho_abs,
            v3_pert=rho_pert,
        )
    finally:
        os.chdir(current_dir)

    print(f"Wrote {output_dir / OUTPUT_XYZ_NAME}")


if __name__ == "__main__":
    main()
