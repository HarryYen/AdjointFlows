from plotting_modules import find_minmax_from_xyz_file, find_nxnynz_from_xyz_file, interp_2d_in_specific_dep, to_xarray_2d 
from scipy.interpolate import RegularGridInterpolator
from tools.job_utils import check_dir_exists
import pygmt
import pandas as pd
import numpy as np
import sys
import os
import xarray as xr


def plot_horizontal_slices_pert(map_region, base_dir, input_dir, output_dir):    

    
    scalar_list = ['dvp','dvs','drho']
    depth_list = [6, 10, 15, 20, 30, 50, 80, 120, 150]
    vp_range = [-20, 20]
    vs_range = [-30, 30]
    rho_range = [-20, 20]



    input_file = os.path.join(input_dir, 'model.xyz')
    nx, ny, nz = find_nxnynz_from_xyz_file(input_file)
    lon_min, lon_max, lat_min, lat_max, dep_min, dep_max = find_minmax_from_xyz_file(input_file)

    all_arr_flat = np.loadtxt(input_file, skiprows=5)
    grid_arr_flat = all_arr_flat[:,0:3]
    lon_arr, lat_arr, dep_arr = grid_arr_flat[:,0], grid_arr_flat[:,1], grid_arr_flat[:,2]
    lon_arr_uniq, lat_arr_uniq, dep_arr_uniq = np.unique(lon_arr), np.unique(lat_arr), np.unique(dep_arr)
    # vp_arr_flat, vs_arr_flat, rho_arr_flat = all_arr_flat[:,3], all_arr_flat[:,4], all_arr_flat[:,5]
    dvp_arr_flat, dvs_arr_flat, drho_arr_flat = all_arr_flat[:,6], all_arr_flat[:,7], all_arr_flat[:,8]
    dvp_arr = dvp_arr_flat.reshape(nz, nx, ny)
    dvs_arr = dvs_arr_flat.reshape(nz, nx, ny)
    drho_arr = drho_arr_flat.reshape(nz, nx, ny)
    grid_arr = grid_arr_flat.reshape(nz, nx, ny, 3)


    cpt_range_list = [vp_range, vs_range, rho_range]
    model_arr_list = [dvp_arr, dvs_arr, drho_arr]
    grd_range = [lon_min, lon_max, lat_min, lat_max]
    for ii, scalar in enumerate(scalar_list):
        print(f'======>{scalar}')
        fig = pygmt.Figure()
        pygmt.makecpt(cmap=f'{base_dir}/adjointflows/visualizer/Vp_ptb.cpt', series=cpt_range_list[ii], reverse=False)
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")

        
        with fig.subplot(nrows=3, ncols=3, figsize=("17c", "17c"), margins='0.005c', frame=['a', 'WSne']):
            for i in range(3): 
                for j in range(3): 
                    index = i * 3 + j
                    dep = depth_list[index]
                    print(f'====>{dep}km')
                    model_arr = model_arr_list[ii]
                    selected_df = interp_2d_in_specific_dep(lon_arr_uniq, lat_arr_uniq, dep_arr_uniq, model_arr, dep)

                    pygmt.xyz2grd(data=selected_df, 
                                outgrid='tmp.grd', 
                                region=grd_range, 
                                spacing=f'{nx}+n/{ny}+n',
                                verbose='q')
                    pygmt.grdsample(grid='tmp.grd', spacing=0.04, 
                                    region=grd_range, outgrid='tmp_fine.grd',
                                    verbose='q')

                    with fig.set_panel(panel=index):  # sets the current panel
                        fig.grdimage(grid='tmp_fine.grd', cmap=True, region=map_region, projection='M?', frame=True)
                        fig.coast(shorelines=True)
                        fig.text(text=f"dep: {depth_list[index]:3d}km", font="12p,Helvetica-Bold", position="BR", frame=True)

            check_dir_exists(output_dir)
            pygmt.config(FONT_ANNOT_PRIMARY='20p,Helvetica')
            pygmt.config(FONT_LABEL='20p,Helvetica')
            fig.colorbar(frame=f'a10f10+l{scalar}(%)')
            fig.savefig(f'{output_dir}/{scalar}.png', dpi=1000)


def plot_horizontal_slices_abs(map_region, base_dir, input_dir, output_dir):    

    
    scalar_list = ['vp','vs','rho']
    unit_list = ['km/s', 'km/s', 'g/cm^3']
    depth_list = [6, 10, 15, 20, 30, 50, 80, 120, 150]



    input_file = os.path.join(input_dir, 'model.xyz')
    nx, ny, nz = find_nxnynz_from_xyz_file(input_file)
    lon_min, lon_max, lat_min, lat_max, dep_min, dep_max = find_minmax_from_xyz_file(input_file)

    all_arr_flat = np.loadtxt(input_file, skiprows=5)
    grid_arr_flat = all_arr_flat[:,0:3]
    lon_arr, lat_arr, dep_arr = grid_arr_flat[:,0], grid_arr_flat[:,1], grid_arr_flat[:,2]
    lon_arr_uniq, lat_arr_uniq, dep_arr_uniq = np.unique(lon_arr), np.unique(lat_arr), np.unique(dep_arr)
    vp_arr_flat, vs_arr_flat, rho_arr_flat = all_arr_flat[:,3], all_arr_flat[:,4], all_arr_flat[:,5]
    # dvp_arr_flat, dvs_arr_flat, drho_arr_flat = all_arr_flat[:,6], all_arr_flat[:,7], all_arr_flat[:,8]
    vp_arr = vp_arr_flat.reshape(nz, nx, ny)
    vs_arr = vs_arr_flat.reshape(nz, nx, ny)
    rho_arr = rho_arr_flat.reshape(nz, nx, ny)
    # grid_arr = grid_arr_flat.reshape(nz, nx, ny, 3)


    # cpt_range_list = [vp_range, vs_range, rho_range]
    model_arr_list = [vp_arr, vs_arr, rho_arr]
    grd_range = [lon_min, lon_max, lat_min, lat_max]
    for ii, scalar in enumerate(scalar_list):
        print(f'======>{scalar}')
        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")
        with fig.subplot(nrows=3, ncols=3, figsize=("28c", "25c"), margins='0.01c', frame=['a', 'WSne']):
            for i in range(3): 
                for j in range(3): 
                    index = i * 3 + j
                    dep = depth_list[index]
                    print(f'====>{dep}km')
                    model_arr = model_arr_list[ii]
                    selected_df = interp_2d_in_specific_dep(lon_arr_uniq, lat_arr_uniq, dep_arr_uniq, model_arr, dep)
                    
                    cpt_range = [selected_df['scalar'].min() - 0.3, selected_df['scalar'].max() + 0.3, 0.025]
                    pygmt.xyz2grd(data=selected_df, 
                                outgrid='tmp.grd', 
                                region=grd_range, 
                                spacing=f'{nx}+n/{ny}+n',
                                verbose='q')
                    pygmt.grdsample(grid='tmp.grd', spacing=0.02, 
                                    region=grd_range, outgrid='tmp_fine.grd',
                                    verbose='q')

                    with fig.set_panel(panel=index):  # sets the current panel
                        pygmt.makecpt(cmap='vik', series=cpt_range, reverse=True)
                        fig.grdimage(grid='tmp_fine.grd', cmap=True, region=map_region, projection='M?', frame=True)
                        fig.coast(shorelines=True)
                        fig.text(text=f"dep: {depth_list[index]:3d}km", font="12p,Helvetica-Bold", position="BR", frame=True)
                        pygmt.config(FONT_ANNOT_PRIMARY='20p,Helvetica')
                        pygmt.config(FONT_LABEL='20p,Helvetica')
                        fig.colorbar(frame=f'a0.5f0.5+l{scalar}({unit_list[ii]})', position='JBC+w3.5c/0.3c+v+o3.5c/-3.5c')
            check_dir_exists(output_dir)

            
            fig.savefig(f'{output_dir}/{scalar}.png', dpi=1000)