#%%
from plotting_modules import find_minmax_from_xyz_file, find_nxnynz_from_xyz_file, interp_2d_in_specific_dep, to_xarray_2d 
from scipy.interpolate import RegularGridInterpolator
# from tools.job_utils import check_dir_exists
import pygmt
import pandas as pd
import numpy as np
import sys
import os
import xarray as xr
import yaml


def plot_horizontal_slices_abs(map_region):    

    # ----------------------------------------------------------------------
    # Load the configuration
    # ----------------------------------------------------------------------
    scalar_list = ['vp', 'vs', 'rho']
    unit_list = ['km/s', 'km/s', 'g/cm^3']
    cmap = 'roma'
    reverse_cmap = False
    dep = 100
    tomo_dir = '/home/harry/Work/AdjointFlows/TOMO'
    vp_range = [8.0, 9.5, 0.05]
    vs_range = [4.4, 5.0, 0.05]
    rho_range = [2.0, 3.5, 0.05]
    model_num =  0
    # -----------------------------------------------------------------------
    input_dir = os.path.join(tomo_dir, f'm{model_num:03d}', 'OUTPUT')
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


    cpt_range_list = [vp_range, vs_range, rho_range]
    model_arr_list = [vp_arr, vs_arr, rho_arr]
    grd_range = [lon_min, lon_max, lat_min, lat_max]
    for ii, scalar in enumerate(scalar_list):
        print(f'======>{scalar}')
        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")
        # fig.basemap(region=map_region, projection='M10c', frame=True)
        print(f'====>{dep}km')
        model_arr = model_arr_list[ii]
        selected_df = interp_2d_in_specific_dep(lon_arr_uniq, lat_arr_uniq, dep_arr_uniq, model_arr, dep)
        cpt_range = cpt_range_list[ii]
        # cpt_range = [selected_df['scalar'].min() - 0.3, selected_df['scalar'].max() + 0.3, 0.025]
        pygmt.xyz2grd(data=selected_df, 
                    outgrid='tmp.grd', 
                    region=grd_range, 
                    spacing=f'{nx}+n/{ny}+n',
                    verbose='q')
        pygmt.grdsample(grid='tmp.grd', spacing=0.02, 
                        region=grd_range, outgrid='tmp_fine.grd',
                        verbose='q')

  
        pygmt.makecpt(cmap=cmap, series=cpt_range, reverse=reverse_cmap)
        fig.grdimage(grid='tmp_fine.grd', cmap=True, region=map_region, projection='M10c', frame=True)
        fig.coast(shorelines=True)
        fig.text(text=f"dep: {dep:3d}km", font="20p,Helvetica-Bold", position="BR", frame=True)
        pygmt.config(FONT_ANNOT_PRIMARY='20p,Helvetica')
        pygmt.config(FONT_LABEL='20p,Helvetica')
        fig.colorbar(frame=f'a0.5f0.5+l{scalar}({unit_list[ii]})', position='JBC+w3.5c/0.3c+v+o5.5c/-4.5c')
        # check_dir_exists(output_dir)

        fig.show()
        # fig.savefig(f'{output_dir}/{scalar}.png', dpi=300)


if __name__ == '__main__':

    plot_horizontal_slices_abs(map_region = [119.0, 123.0, 21.0, 26.0])
# %%
