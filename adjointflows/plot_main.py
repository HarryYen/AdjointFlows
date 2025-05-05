from visualizer import TomographyVisualizer
from tools import ConfigManager
from tools import GLOBAL_PARAMS
import sys


def main():
    # ---------------------------
    # Setting up the parameters
    # ---------------------------
    plot_config = ConfigManager('visualizer/plot_config.yaml')
    plot_config.load()
    
    do_preprocess = plot_config.get('work_flow.do_preprocess')
    plot_horizontal_slices = plot_config.get('work_flow.plot_horizontal_slices')
    plot_vertical_slices = plot_config.get('work_flow.plot_vertical_slices')

    model_n = plot_config.get('pre_processing.model_num')
    lon_range = plot_config.get('pre_processing.lon_range')
    lat_range = plot_config.get('pre_processing.lat_range')
    dep_range = plot_config.get('pre_processing.dep_range')
    lon_interval = plot_config.get('pre_processing.lon_interval')
    lat_interval = plot_config.get('pre_processing.lat_interval')
    dep_interval = plot_config.get('pre_processing.dep_interval')
    
    # ---------------------------
    config = ConfigManager('config.yaml')
    config.load()
    
    visualizer = TomographyVisualizer(config=config, 
                                      global_params=GLOBAL_PARAMS, 
                                      specified_model_num=model_n)
    visualizer.load_params()
    visualizer.setup_spatial_range(lon_range=lon_range, lat_range=lat_range, dep_range=dep_range,
                                   lon_interval=lon_interval, lat_interval=lat_interval, dep_interval=dep_interval)
    
    if do_preprocess:
        visualizer.from_bin_to_vtu_model(model_name='vp')
        visualizer.from_bin_to_vtu_model(model_name='vs')
        visualizer.from_bin_to_vtu_model(model_name='rho')
        vp_abs, vp_pert = visualizer.project_gll_to_regular(kernel_name='vp',
                                                            utm_zone=50,
                                                            is_north_hemisphere=True)
        vs_abs, vs_pert = visualizer.project_gll_to_regular(kernel_name='vs',
                                                            utm_zone=50,
                                                            is_north_hemisphere=True)
        rho_abs, rho_pert = visualizer.project_gll_to_regular(kernel_name='rho',
                                                            utm_zone=50,
                                                            is_north_hemisphere=True)
        # model unit transformation (from m to km)
        vp_abs = vp_abs / 1000.
        vs_abs = vs_abs / 1000.
        rho_abs = rho_abs / 1000.
        visualizer.output_model_txt_file(output_file_name='model.xyz',
                                v1_abs=vp_abs, v1_pert=vp_pert, 
                                v2_abs=vs_abs, v2_pert=vs_pert,
                                v3_abs=rho_abs, v3_pert=rho_pert) 

    if plot_horizontal_slices:
        visualizer.plot_horizontal_slices_3x3_pert()
        visualizer.plot_horizontal_slices_3x3_abs()
    if plot_vertical_slices:
        visualizer.plot_vertical_profile_pert()
        visualizer.plot_vertical_profile_abs()

    # visualizer.plot_horizontal_slices_3x3_updated(model_ref_num=0)
    # visualizer.plot_horizontal_slices_3x3_updated(model_ref_num=5)
if __name__ == '__main__':
    main()