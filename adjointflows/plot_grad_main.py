from visualizer import TomographyVisualizer
from tools import ConfigManager
from tools import GLOBAL_PARAMS
import numpy as np
import os


def normalize_grad(arr):
    max_abs = np.nanmax(np.abs(arr))
    if not np.isfinite(max_abs) or max_abs == 0:
        return arr
    return arr / max_abs

def resolve_kernels_dir(tomo_dir, kernel_source, dataset_name, kernel_subdir):
    kernel_source = kernel_source.lower()
    kernel_subdir = kernel_subdir.upper()
    if kernel_subdir not in ("PRECOND", "SMOOTH"):
        raise ValueError(f"Unsupported kernel_subdir: {kernel_subdir}")
    if kernel_source in ("combined", "kernel_combined"):
        kernel_base_dir = "KERNEL_COMBINED"
    elif kernel_source in ("dataset", "kernel_dataset"):
        if not dataset_name:
            raise ValueError("dataset_name is required when kernel_source is 'dataset'.")
        kernel_base_dir = f"KERNEL_{dataset_name}"
    else:
        raise ValueError(f"Unknown kernel_source: {kernel_source}")

    return os.path.join(tomo_dir, kernel_base_dir, kernel_subdir)

def resolve_output_dir(tomo_dir, kernel_source, dataset_name, kernel_subdir):
    kernel_source = kernel_source.lower()
    kernel_subdir = kernel_subdir.lower()
    if kernel_source in ("combined", "kernel_combined"):
        output_tag = "combined"
    elif kernel_source in ("dataset", "kernel_dataset"):
        if not dataset_name:
            raise ValueError("dataset_name is required when kernel_source is 'dataset'.")
        output_tag = f"dataset_{dataset_name}"
    else:
        raise ValueError(f"Unknown kernel_source: {kernel_source}")

    return os.path.join(tomo_dir, "OUTPUT", "GRADIENTS", output_tag, kernel_subdir)

def main():
    # ---------------------------
    # Setting up the parameters
    # ---------------------------

    model_n = 0
    kernel_source = "combined"  # "combined" or "dataset"
    dataset_name = None  # required when kernel_source == "dataset"
    kernel_subdir = "PRECOND"  # PRECOND or SMOOTH
    lon_range = [119.0, 123.0]
    lat_range = [21.0, 26.0]
    dep_range = [0., 200]
    lon_interval = 0.08
    lat_interval = 0.08
    dep_interval = 5
    # ---------------------------
    config = ConfigManager('config.yaml')
    config.load()
    
    visualizer = TomographyVisualizer(config=config, 
                                      global_params=GLOBAL_PARAMS, 
                                      specified_model_num=model_n)
    visualizer.load_params()
    visualizer.kernels_dir = resolve_kernels_dir(
        visualizer.tomo_dir, kernel_source, dataset_name, kernel_subdir
    )
    if not os.path.isdir(visualizer.kernels_dir):
        raise FileNotFoundError(f"Kernel directory not found: {visualizer.kernels_dir}")
    visualizer.output_dir = resolve_output_dir(
        visualizer.tomo_dir, kernel_source, dataset_name, kernel_subdir
    )
    os.makedirs(visualizer.output_dir, exist_ok=True)
    visualizer.setup_spatial_range(lon_range=lon_range, lat_range=lat_range, dep_range=dep_range,
                                   lon_interval=lon_interval, lat_interval=lat_interval, dep_interval=dep_interval)
    visualizer.from_bin_to_vtu_gradient(kernel_name='alpha_kernel_smooth')
    visualizer.from_bin_to_vtu_gradient(kernel_name='beta_kernel_smooth')
    visualizer.from_bin_to_vtu_gradient(kernel_name='rho_kernel_smooth')
    vp_abs, vp_pert = visualizer.project_gll_to_regular(kernel_name='alpha_kernel_smooth',
                                                        utm_zone=50,
                                                        is_north_hemisphere=True)
    vs_abs, vs_pert = visualizer.project_gll_to_regular(kernel_name='beta_kernel_smooth',
                                                        utm_zone=50,
                                                        is_north_hemisphere=True)
    rho_abs, rho_pert = visualizer.project_gll_to_regular(kernel_name='rho_kernel_smooth',
                                                          utm_zone=50,
                                                          is_north_hemisphere=True)
    # model unit transformation (from m to km)


    vp_abs = normalize_grad(vp_abs)
    vs_abs = normalize_grad(vs_abs)
    rho_abs = normalize_grad(rho_abs)

    visualizer.output_kernel_txt_file(output_file_name='gradient.xyz',
                               v1_abs=vp_abs, v1_pert=vp_pert, 
                               v2_abs=vs_abs, v2_pert=vs_pert,
                               v3_abs=rho_abs, v3_pert=rho_pert) 

    visualizer.plot_horizontal_slices_3x3_gradient()
if __name__ == '__main__':
    main()
