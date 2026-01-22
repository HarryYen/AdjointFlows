from visualizer import TomographyVisualizer
from tools import ConfigManager
from tools import GLOBAL_PARAMS
import numpy as np


def normalize_grad(arr):
    max_abs = np.nanmax(np.abs(arr))
    if not np.isfinite(max_abs) or max_abs == 0:
        return arr
    return arr / max_abs

def main():
    # ---------------------------
    # Setting up the parameters
    # ---------------------------

    model_n = 0
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
