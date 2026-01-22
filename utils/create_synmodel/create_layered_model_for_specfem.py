#%%
from scipy.ndimage import gaussian_filter
from scipy import interpolate
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def visualizing_and_check(arr, nx, ny, nz):
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(arr[nx//2, :, :].T, aspect='auto', cmap='jet')
    plt.subplot(1, 3, 2)
    plt.imshow(arr[:, ny//2, :].T, aspect='auto', cmap='jet')
    plt.subplot(1, 3, 3)
    plt.imshow(arr[:, :, nz//2], cmap='jet')
    fig.show()

def read_tomo_header(table):
    
    header_rows = 4
    lines_list = []
    with open(table, 'r') as f:
        for idx in range(header_rows):
            line = f.readline()
            lines_list.append(line)
            if idx == 1:
                values = line.strip().split()
                dx = round(float(values[0]), 3)
                dy = round(float(values[1]), 3)
                dz = round(float(values[2]), 3)
    return lines_list, dx, dy, dz

def ckb_len_to_freq(lon_len, lat_len, dep_len, dx, dy, dz):
        
        lon_freq = int(lon_len / dx)
        lat_freq = int(lat_len / dy)
        dep_freq = int(dep_len / dz)
        
        return lon_freq, lat_freq, dep_freq

def extract_vals_from_origin_file(table):
    
    df = pd.read_csv(table, sep=r'\s+', header=None, skiprows=4,
                     names=['lon', 'lat', 'dep', 'vp', 'vs', 'rho'])
    lon_uniq = df['lon'].unique()
    lat_uniq = df['lat'].unique()
    dep_uniq = df['dep'].unique()
    
    lon_index = df['lon'].map({v: i for i, v in enumerate(lon_uniq)}).to_numpy()
    lat_index = df['lat'].map({v: i for i, v in enumerate(lat_uniq)}).to_numpy()
    dep_index = df['dep'].map({v: i for i, v in enumerate(dep_uniq)}).to_numpy()
    
    lon_vals = df['lon'].to_numpy()
    lat_vals = df['lat'].to_numpy()
    dep_vals = df['dep'].to_numpy()
    
    index_list = [lon_index, lat_index, dep_index]
    vals_list = [lon_vals, lat_vals, dep_vals]    
    return index_list, vals_list

def read_tomo_array(table, scalar='vp'):
    
    # Read the table
    df = pd.read_csv(table, sep=r'\s+', header=None, skiprows=4,
                     names=['lon', 'lat', 'dep', 'vp', 'vs', 'rho'])
    lon_uniq = df['lon'].unique()
    lat_uniq = df['lat'].unique()
    dep_uniq = df['dep'].unique()
    
    lon_index = df['lon'].map({v: i for i, v in enumerate(lon_uniq)}).to_numpy()
    lat_index = df['lat'].map({v: i for i, v in enumerate(lat_uniq)}).to_numpy()
    dep_index = df['dep'].map({v: i for i, v in enumerate(dep_uniq)}).to_numpy()
    matrix = np.full((len(lon_uniq), len(lat_uniq), len(dep_uniq)), np.nan)

    
    matrix[lon_index, lat_index, dep_index] = df[scalar].to_numpy()
    
    return matrix

def generating_checkerboard(matrix, lon_freq, lat_freq, dep_freq, amplitude=0.2, power=1):
    global visualization_flag
    
    nx, ny, nz = matrix.shape
    
    lon_idx = np.arange(nx).reshape(-1, 1, 1)
    lat_idx = np.arange(ny).reshape(1, -1, 1)
    depth_idx = np.arange(nz).reshape(1, 1, -1)
    
    checkerboard_pattern = (np.sin(2 * np.pi * lon_idx / (2*lon_freq)) * \
                       np.sin(2 * np.pi * lat_idx / (2*lat_freq)) * \
                       np.sin(2 * np.pi * depth_idx / (2*dep_freq))) ** power
                       
    ckb_arr = origin_arr * (1 + amplitude * checkerboard_pattern)
    if visualization_flag:
        visualizing_and_check(checkerboard_pattern, nx, ny, nz)
    
    return ckb_arr

def calculating_minmax(arr):
    min_val, max_val = np.nanmin(arr), np.nanmax(arr)
    return min_val, max_val
    

def output_table(output_file, lines_list, minmax_list, index_list, vals_list, scalar_arr_list):
    
    header_info = ''.join(lines_list[:3])
    minmax_string = f' {minmax_list[0][0]:.3f} {minmax_list[0][1]:.3f} {minmax_list[1][0]:.3f} {minmax_list[1][1]:.3f} {minmax_list[2][0]:.3f} {minmax_list[2][1]:.3f}'
    header_info += minmax_string
    
    vp_arr, vs_arr, rho_arr = scalar_arr_list[0], scalar_arr_list[1], scalar_arr_list[2]
    lon_idx, lat_idx, dep_idx = index_list[0], index_list[1], index_list[2]
    vp_flat = vp_arr[lon_idx, lat_idx, dep_idx]
    vs_flat = vs_arr[lon_idx, lat_idx, dep_idx]
    rho_flat = rho_arr[lon_idx, lat_idx, dep_idx]
    
    output_data = np.column_stack((vals_list[0], vals_list[1], vals_list[2], vp_flat, vs_flat, rho_flat))
    np.savetxt(output_file, output_data, fmt='%.3f', header=header_info, comments='')

def output_table(xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz, nx, ny, nz, output_file, scalar_arr_list, vals_arr_list):
    
    header_info = f'{xmin:13.3f} {ymin:13.3f} {zmin:13.3f} {xmax:13.3f} {ymax:13.3f} {zmax:13.3f}\n' 
    header_info += f'{dx:10.3f} {dy:10.3f} {dz:10.3f}\n' 
    header_info += f'{nx:6d} {ny:6d} {nz:6d}\n'

    vp_arr, vs_arr, rho_arr = scalar_arr_list[0], scalar_arr_list[1], scalar_arr_list[2]
    vp_min, vp_max = np.nanmin(vp_arr), np.nanmax(vp_arr)
    vs_min, vs_max = np.nanmin(vs_arr), np.nanmax(vs_arr)
    rho_min, rho_max = np.nanmin(rho_arr), np.nanmax(rho_arr)

    minmax_string = f' {vp_min:10.3f} {vp_max:10.3f} {vs_min:10.3f} {vs_max:10.3f} {rho_min:10.3f} {rho_max:10.3f}'
    header_info += minmax_string
    
    
    output_data = np.column_stack((vals_arr_list[0], vals_arr_list[1], vals_arr_list[2], vp_arr, vs_arr, rho_arr))
    np.savetxt(output_file, output_data, fmt='%12.3f', header=header_info, comments='')
    


    
    

if __name__ == "__main__":
    """
    Create a 1D layered model from a file and output it in a specific format.
    """
    # ---------------PARAMETERS-----------------#
    layered_model = 'H14_1d.txt'
    output_file = 'tomo_1d.xyz'
    xmin, ymin, zmin = 700010.642, 2356665.767, -300000
    dx, dy, dz = 5000, 5000, 5000
    nx, ny, nz = 127, 111, 62
    # ------------------------------------------#
    
    layer_data = np.loadtxt(layered_model).T


    xmax = xmin + (nx - 1) * dx
    ymax = ymin + (ny - 1) * dy
    zmax = zmin + (nz - 1) * dz

    x_arr = np.arange(xmin, xmax + dx, dx)
    y_arr = np.arange(ymin, ymax + dy, dy)
    z_arr = np.arange(zmin, zmax + dz, dz)
    y_mesh, z_mesh, x_mesh = np.meshgrid(y_arr, z_arr, x_arr)
    x_flat = x_mesh.flatten()
    y_flat = y_mesh.flatten()
    z_flat = z_mesh.flatten()

    f_vp = interpolate.interp1d(layer_data[0], layer_data[1], kind='linear')
    f_vs = interpolate.interp1d(layer_data[0], layer_data[2], kind='linear')
    f_rho = interpolate.interp1d(layer_data[0], layer_data[3], kind='linear')
    vp_interp = f_vp(z_flat)
    vs_interp = f_vs(z_flat)
    rho_interp = f_rho(z_flat)
    scalar_arr_list = [vp_interp, vs_interp, rho_interp]
    vals_arr_list = [x_flat, y_flat, z_flat]

    output_table(xmin=xmin, xmax=xmax, 
                 ymin=ymin, ymax=ymax, 
                 zmin=zmin, zmax=zmax, 
                 dx=dx, dy=dy, dz=dz, 
                 nx=nx, ny=ny, nz=nz, 
                 output_file=output_file, scalar_arr_list=scalar_arr_list, vals_arr_list=vals_arr_list)
    
# %%
