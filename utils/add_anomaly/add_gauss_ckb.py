#%%
from modules_anomaly import *
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import sys
import os




def degree_to_meter(dlon_deg, dlat_deg, lat0_deg):
    R = 6371000.0  # Earth radius (m)
    lat0_rad = np.deg2rad(lat0_deg)

    dy = dlat_deg * np.pi/180.0 * R
    dx = dlon_deg * np.pi/180.0 * R * np.cos(lat0_rad)
    return dx, dy


if __name__ == '__main__':
    
    # --------------------
    # Parameters
    # --------------------
    databases_dir = 'DATABASES_MPI'      
    vector = 'vs'
    nproc = 2
    output_dir = 'OUTPUT'
    
    lon_range_deg = [119, 123]
    lat_range_deg = [21, 26]
    dep_range_meter = [0, 200000] # downward positive (change to negative in the script)
    dlon_deg, dlat_deg = 0.5, 0.5
    dz_meter = 50000.0
    lat0_deg = 24.0
    edge_amplitude_ratio_gaussian = 0.15
    utm_zone = 50
    is_north_hemisphere = True
    
    # --------------------
    # main
    # --------------------
    
    # Convert to meters
    dx_meter, dy_meter = degree_to_meter(dlon_deg, dlat_deg, lat0_deg)
    
    # Suggest sigma from desired checkerboard spacing
    sx = suggest_sigma_from_dx(edge_amplitude_ratio_gaussian, dx_meter)
    sy = suggest_sigma_from_dx(edge_amplitude_ratio_gaussian, dy_meter)
    sz = suggest_sigma_from_dx(edge_amplitude_ratio_gaussian, dz_meter)

    lon_start = lon_range_deg[0] - dlon_deg
    lon_end = lon_range_deg[1] + dlon_deg
    lat_start = lat_range_deg[0] - dlat_deg
    lat_end = lat_range_deg[1] + dlat_deg
    dep_start = dep_range_meter[0]
    dep_end = dep_range_meter[1] + dz_meter


    # Construct checkerboard grid centers
    nx = int(np.ceil((lon_end - lon_start) / dlon_deg))
    ny = int(np.ceil((lat_end - lat_start) / dlat_deg))
    nz = int(np.ceil((dep_end - dep_start) / dz_meter))

    lon_centers = lon_start + (np.arange(nx) + 0.5) * dlon_deg
    lat_centers = lat_start + (np.arange(ny) + 0.5) * dlat_deg
    z_centers = -1 * (dep_range_meter[0] + (np.arange(nz) + 0.5) * dz_meter)  # downward positive to negative
    
    # from lon/lat to utm
    lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers, indexing='ij')
    x_grid, y_grid = lonlat_to_utm(lon_grid, lat_grid, utm_zone=utm_zone, is_north_hemisphere=is_north_hemisphere)

    # Read in global region min/max
    x_minmax_regions, y_minmax_regions, z_minmax_regions = [], [], []
    x_arr_list, y_arr_list, z_arr_list = [], [], []
    ibool_arr_list = []
    
    for rank in range(nproc):
        print('read min/max')
        ibool_file = os.path.join(databases_dir, f"proc{rank:06d}_ibool.bin")
        x_file = os.path.join(databases_dir, f"proc{rank:06d}_x.bin")
        y_file = os.path.join(databases_dir, f"proc{rank:06d}_y.bin")
        z_file = os.path.join(databases_dir, f"proc{rank:06d}_z.bin")

        ibool_arr, ibool_data_type = read_binary_int32(ibool_file)
        x_arr, _ = read_binary_float32(x_file)
        y_arr, _ = read_binary_float32(y_file)
        z_arr, _ = read_binary_float32(z_file)
        
        x_arr = x_arr[ibool_arr - 1]
        y_arr = y_arr[ibool_arr - 1]
        z_arr = z_arr[ibool_arr - 1]
        
        x_arr_list.append(x_arr)
        y_arr_list.append(y_arr)
        z_arr_list.append(z_arr)
        ibool_arr_list.append(ibool_arr)
        
        x_min, x_max = x_arr.min(), x_arr.max()
        y_min, y_max = y_arr.min(), y_arr.max()
        z_min, z_max = z_arr.min(), z_arr.max()

        print(y_min, y_max)
        x_minmax_regions.append([x_min, x_max])
        y_minmax_regions.append([y_min, y_max])
        z_minmax_regions.append([z_min, z_max])
        
    x_min_global, x_max_global = np.min(x_minmax_regions), np.max(x_minmax_regions) 
    y_min_global, y_max_global = np.min(y_minmax_regions), np.max(y_minmax_regions) 
    z_min_global, z_max_global = np.min(z_minmax_regions), np.max(z_minmax_regions)        
    
    # dx = suggest_dx_from_sigma(sx, mid_value=0.3)
    # dy = suggest_dx_from_sigma(sy, mid_value=0.3)
    # dz = suggest_dx_from_sigma(sz, mid_value=0.3)
        
    # nx = int((x_max_global - x_min_global) // dx)
    # ny = int((y_max_global - y_min_global) // dy)
    # nz = int((z_max_global - z_min_global) // dz)
    
    # x_centers = np.linspace(x_min_global + dx/2, x_min_global + dx/2 + nx * dx, nx)
    # y_centers = np.linspace(y_min_global + dy/2, y_min_global + dy/2 + ny * dy, ny)
    # z_centers = np.linspace(z_min_global + dz/2, z_min_global + dz/2 + nz * dz, nz)
    
    # print(x_centers)
    # print(y_centers)
    # print(z_centers)
    # sys.exit()
    
    
    for rank in range(nproc):
        print('processing rank:', rank)
        # ----------------------------------------------------
        # defind the file names
        # ----------------------------------------------------
        vector_file = os.path.join(databases_dir, f"proc{rank:06d}_{vector}.bin")

        # ----------------------------------------------------
        # Read files and construct arrays
        # ----------------------------------------------------
        vector_arr, vector_data_type = read_binary_float32(vector_file)  
        x_arr, y_arr, z_arr = x_arr_list[rank], y_arr_list[rank], z_arr_list[rank]
        lon_arr, lat_arr = utm_to_lonlat(x=x_arr, y=y_arr, utm_zone=utm_zone, is_north_hemisphere=is_north_hemisphere)

        # ----------------------------------------------------
        # Add anomaly
        # ----------------------------------------------------
    
        ckb_pert = build_gaussian_ckb(
            lon_arr=lon_arr, lat_arr=lat_arr, 
            x_arr=x_arr, y_arr=y_arr, z_arr=z_arr,
            lon_centers=lon_centers, lat_centers=lat_centers, z_centers=z_centers,
            x_grid=x_grid, y_grid=y_grid,
            dlon_deg=dlon_deg, dlat_deg=dlat_deg,
            utm_zone=utm_zone, is_north_hemisphere=is_north_hemisphere,
            dz=dz_meter, sigma=(sx, sy, sz), n_sigma=2, amplitude=0.01)
    
        vector_arr = vector_arr * (1.0 + ckb_pert)
        

        # ----------------------------------------------------
        # Output the file
        # ----------------------------------------------------
        os.makedirs(output_dir, exist_ok=True)
        output_binary_file(data = vector_arr, 
                           outfile = f'{output_dir}/{os.path.basename(vector_file)}', 
                           data_type_int = ibool_data_type, 
                           data_type_float = vector_data_type)
        
        



# %%
