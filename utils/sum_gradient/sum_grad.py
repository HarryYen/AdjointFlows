#%%
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd
import pygmt
import os
import sys


# -----------------------------------------------
# Functions 
# -----------------------------------------------
def interp_2d_in_specific_dep(lon_uniq, lat_uniq, dep_uniq, arr, target_dep):

    interpolator = RegularGridInterpolator((dep_uniq, lon_uniq, lat_uniq), arr)
    lon_grid, lat_grid = np.meshgrid(lon_uniq, lat_uniq)
    points = np.array([np.full(lon_grid.size, target_dep), lon_grid.ravel(), lat_grid.ravel()]).T
    interpolated_values = interpolator(points)
    df = pd.DataFrame({
    'lon': lon_grid.ravel(),
    'lat': lat_grid.ravel(),
    'scalar': interpolated_values
    })
    
    return df


# -----------------------------------------------
# Parameters
# -----------------------------------------------

work_dir_each_set = [
    '/home/harry/Work/Other_AdjointFlows/AdjointFlows_no_smoothed/TOMO',
    '/home/harry/Work/AdjointFlows_mesh_2_no_smoothed/TOMO'
]
beg_model_each_set = [
    [0, 13],
    [14, 26]
]
nx, ny, nz = 51, 64, 41
grd_range = [119, 123.0, 21., 26.]
map_region = [119, 123., 21.5, 26.]
adjust_factor = 6e-11
# cpt_range = [0, 5e-10]
cpt_range = [-6, 6]
# -----------------------------------------------
scalar_list =  ['alpha', 'beta', 'rho']
depth_list = [6, 10, 15, 20, 30, 50, 80, 120, 150]

alpha_arr_list, beta_arr_list, rho_arr_list = [], [], []
for set_index, work_dir in enumerate(work_dir_each_set):
    model_num_list = np.arange(beg_model_each_set[set_index][0], beg_model_each_set[set_index][1]+1)
    for model_n in model_num_list:
        print(f'Processing model m{model_n:03d} in {work_dir}')

        input_dir = os.path.join(work_dir, f'm{model_n:03d}/OUTPUT')
        input_file = os.path.join(input_dir, 'gradient.xyz')
        all_arr_flat = np.loadtxt(input_file, skiprows=5)
        alpha_arr_flat, beta_arr_flat, rho_arr_flat = all_arr_flat[:,3], all_arr_flat[:,4], all_arr_flat[:,5]
        alpha_arr_list.append(np.abs(alpha_arr_flat))
        beta_arr_list.append(np.abs(beta_arr_flat))
        rho_arr_list.append(np.abs(rho_arr_flat))

# sum up all the model
alpha_arr_flat = np.sum(np.array(alpha_arr_list), axis=0)
beta_arr_flat = np.sum(np.array(beta_arr_list), axis=0)
rho_arr_flat = np.sum(np.array(rho_arr_list), axis=0)

# adjust the values
alpha_arr_flat = alpha_arr_flat / adjust_factor
beta_arr_flat = beta_arr_flat / adjust_factor
rho_arr_flat = rho_arr_flat / adjust_factor


# take nature log
alpha_arr_flat = np.log(alpha_arr_flat)
beta_arr_flat = np.log(beta_arr_flat)
rho_arr_flat = np.log(rho_arr_flat)


# Postprocessing
alpha_arr = alpha_arr_flat.reshape(nz, nx, ny)
beta_arr = beta_arr_flat.reshape(nz, nx, ny)
rho_arr = rho_arr_flat.reshape(nz, nx, ny)

# MASK
alpha_arr[alpha_arr < 0] = np.nan
beta_arr[beta_arr < 0] = np.nan

model_arr_list = [alpha_arr, beta_arr, rho_arr]
# location information
grid_arr_flat = all_arr_flat[:,0:3]
lon_arr, lat_arr, dep_arr = grid_arr_flat[:,0], grid_arr_flat[:,1], grid_arr_flat[:,2]
lon_arr_uniq, lat_arr_uniq, dep_arr_uniq = np.unique(lon_arr), np.unique(lat_arr), np.unique(dep_arr)

# print the gradient min/max
print('Alpha min/max:', np.nanmin(alpha_arr_flat), np.nanmax(alpha_arr_flat))
print('Beta min/max:', np.nanmin(beta_arr_flat), np.nanmax(beta_arr_flat))


for ii, scalar in enumerate(scalar_list):
        print(f'======>{scalar}')
        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")
        with fig.subplot(nrows=3, ncols=3, figsize=("25c", "25c"), margins='0.01c', frame=['a', 'WSne']):
            for i in range(3): 
                for j in range(3): 
                    index = i * 3 + j
                    dep = depth_list[index]
                    print(f'====>{dep}km')
                    model_arr = model_arr_list[ii]
                    selected_df = interp_2d_in_specific_dep(lon_arr_uniq, lat_arr_uniq, dep_arr_uniq, model_arr, dep)
                    
                    max_value = np.nanmax(model_arr)
                    min_value = np.nanmin(model_arr)

                    # cpt_min = min_value
                    # cpt_max = max_value
                    # cpt_unit_vec = (cpt_max - cpt_min) / 100
                    # cpt_range = [cpt_min, cpt_max, cpt_unit_vec]

                    pygmt.xyz2grd(data=selected_df, 
                                outgrid='tmp.grd', 
                                region=grd_range, 
                                spacing=f'{nx}+n/{ny}+n',
                                verbose='q')
                    pygmt.grdsample(grid='tmp.grd', spacing=0.02, 
                                    region=grd_range, outgrid='tmp_fine.grd',
                                    verbose='q')

                    with fig.set_panel(panel=index):  # sets the current panel
                        pygmt.makecpt(cmap='hot', series=cpt_range, continuous=True, reverse=False, background=True)
                        fig.grdimage(grid='tmp_fine.grd', cmap=True, region=map_region, projection='M?', frame=True)
                        fig.coast(shorelines='2p,gray')
                        fig.text(text=f"Depth: {depth_list[index]:3d}km", font="16p,Helvetica-Bold,white", position="BR", frame=True)
                        pygmt.config(FONT_ANNOT_PRIMARY='20p,Helvetica')
                        pygmt.config(FONT_LABEL='20p,Helvetica')
                        # pygmt.config(FORMAT_FLOAT_MAP="%.1e")
            fig.colorbar(frame=f'xa2f1+lln(K_{scalar}/{adjust_factor:.0E})', position='jBR+o0.1c')

            # fig.show()
            fig.savefig(f'summed_gradient_{scalar}.png', dpi=300)
# %%
