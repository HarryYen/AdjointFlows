from visualizer import TomographyVisualizer
from vertical_slices_grad_diff import plot_vertical_slices_grad_diff
from horizontal_slices_grad_diff import plot_horizontal_slices_grad_diff
from tools import ConfigManager
from tools import GLOBAL_PARAMS
import sys


def main():
    # ---------------------------
    # Setting up the parameters
    # ---------------------------

    model_n = 28
    model_ref_n = 26
    lon_range = [119.0, 123.0]
    lat_range = [21.0, 26.0]
    dep_range = [0., 200]
    lon_interval = 0.08
    lat_interval = 0.08
    dep_interval = 5
    # ---------------------------
    config = ConfigManager('config.yaml')
    config.load()
    
    base_dir = GLOBAL_PARAMS.get('base_dir')
    input_dir = f'{base_dir}/TOMO/m{model_n:03d}/OUTPUT'
    ref_dir = f'{base_dir}/TOMO/m{model_ref_n:03d}/OUTPUT'
    
    visualizer = TomographyVisualizer(config=config, 
                                      global_params=GLOBAL_PARAMS, 
                                      specified_model_num=model_n)
    visualizer.load_params()
    visualizer.setup_spatial_range(lon_range=lon_range, lat_range=lat_range, dep_range=dep_range,
                                   lon_interval=lon_interval, lat_interval=lat_interval, dep_interval=dep_interval)
    # --------------------------------
    # Reading current gradient
    # --------------------------------

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
    
    
    # # --------------------------------
    # # Reading reference gradient
    # # --------------------------------
    visualizer_ref = TomographyVisualizer(config=config, 
                                      global_params=GLOBAL_PARAMS, 
                                      specified_model_num=model_ref_n)

    visualizer_ref.load_params()
    visualizer_ref.setup_spatial_range(lon_range=lon_range, lat_range=lat_range, dep_range=dep_range,
                                   lon_interval=lon_interval, lat_interval=lat_interval, dep_interval=dep_interval)
    visualizer_ref.from_bin_to_vtu_gradient(kernel_name='alpha_kernel_smooth')
    visualizer_ref.from_bin_to_vtu_gradient(kernel_name='beta_kernel_smooth')
    visualizer_ref.from_bin_to_vtu_gradient(kernel_name='rho_kernel_smooth')
    
    vp_abs_ref, vp_pert_ref = visualizer_ref.project_gll_to_regular(kernel_name='alpha_kernel_smooth',
                                                        utm_zone=50,
                                                        is_north_hemisphere=True)
    vs_abs_ref, vs_pert_ref = visualizer_ref.project_gll_to_regular(kernel_name='beta_kernel_smooth',
                                                        utm_zone=50,
                                                        is_north_hemisphere=True)
    rho_abs_ref, rho_pert_ref = visualizer_ref.project_gll_to_regular(kernel_name='rho_kernel_smooth',
                                                          utm_zone=50,
                                                          is_north_hemisphere=True)
    # model unit transformation (from m to km)

    vp_diff = vp_abs - vp_abs_ref
    vs_diff = vs_abs - vs_abs_ref
    rho_diff = rho_abs - rho_abs_ref
    
    visualizer.output_kernel_txt_file(output_file_name='gradient_difference.xyz',
                               v1_abs=vp_diff, v1_pert=vp_pert_ref, 
                               v2_abs=vs_diff, v2_pert=vs_pert_ref,
                               v3_abs=rho_diff, v3_pert=rho_pert_ref) 

    # visualizer.plot_horizontal_slices_3x3_grad_diff()
    plot_horizontal_slices_grad_diff(grad_diff_file=f'{input_dir}/gradient_difference.xyz',
                                   model_peturb_file=f'{input_dir}/model.xyz',
                                   model_ref_file=f'{ref_dir}/model.xyz',
                                   output_dir=input_dir)
    
    # plot_vertical_slices_grad_diff(grad_diff_file=f'{input_dir}/gradient_difference.xyz',
    #                                model_peturb_file=f'{input_dir}/model.xyz',
    #                                model_ref_file=f'{ref_dir}/model.xyz',
    #                                output_dir=input_dir)
if __name__ == '__main__':
    main()
