from visualizer import TomographyVisualizer
from tools import ConfigManager
from tools import GLOBAL_PARAMS

def main():
    # ---------------------------
    # Setting up the parameters
    # ---------------------------

    model_n = 100
    lon_range = [119.0, 123.0]
    lat_range = [21.0, 26.0]
    dep_range = [0., 200]
    lon_interval = 0.08
    lat_interval = 0.08
    dep_interval = 5
    max_dist = 2000 # in meters
    # ---------------------------
    config = ConfigManager('config.yaml')
    config.load()
    
    visualizer = TomographyVisualizer(config=config, 
                                      global_params=GLOBAL_PARAMS, 
                                      specified_model_num=model_n)
    visualizer.load_params()
    visualizer.setup_spatial_range(lon_range=lon_range, lat_range=lat_range, dep_range=dep_range,
                                   lon_interval=lon_interval, lat_interval=lat_interval, dep_interval=dep_interval)
    visualizer.from_bin_to_vtu_model(model_name='vp')
    visualizer.from_bin_to_vtu_model(model_name='vs')
    visualizer.from_bin_to_vtu_model(model_name='rho')
    vp_abs, vp_pert = visualizer.project_gll_to_regular(kernel_name='vp',
                                                        utm_zone=50,
                                                        is_north_hemisphere=True,
                                                        max_distance=max_dist)
    vs_abs, vs_pert = visualizer.project_gll_to_regular(kernel_name='vs',
                                                        utm_zone=50,
                                                        is_north_hemisphere=True,
                                                        max_distance=max_dist)
    rho_abs, rho_pert = visualizer.project_gll_to_regular(kernel_name='rho',
                                                          utm_zone=50,
                                                          is_north_hemisphere=True,
                                                          max_distance=max_dist)
    ## model unit transformation (from m to km)
    vp_abs = vp_abs / 1000.
    vs_abs = vs_abs / 1000.
    rho_abs = rho_abs / 1000.
    visualizer.output_txt_file(output_file_name='model.xyz',
                               v1_abs=vp_abs, v1_pert=vp_pert, 
                               v2_abs=vs_abs, v2_pert=vs_pert,
                               v3_abs=rho_abs, v3_pert=rho_pert) 

    visualizer.plot_horizontal_slices_3x3_pert()
    visualizer.plot_horizontal_slices_3x3_abs()
if __name__ == '__main__':
    main()