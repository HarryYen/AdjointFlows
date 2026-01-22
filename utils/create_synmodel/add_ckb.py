#%%
from scipy.ndimage import gaussian_filter
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
    
    df = pd.read_csv(table, sep='\s+', header=None, skiprows=4,
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
    df = pd.read_csv(table, sep='\s+', header=None, skiprows=4,
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
    


    
    

if __name__ == "__main__":
    
    # ---------------PARAMETERS-----------------#
    tomo_file = 'tomo_1d.xyz'
    output_file = 'tomo_1d_ckb.xyz'
    lon_ckb_len, lat_ckb_len, dep_ckb_len = 50000, 50000, 20000 # lon, lat for degree, dep for km
    amplitude = 0.1 # e.g. 0.1 = +-10%
    gaussian_sigma = 1
    power = 1
    visualization_flag = True
    # ------------------------------------------#
    
    lines_list, dx, dy, dz = read_tomo_header(tomo_file)
    lon_freq, lat_freq, dep_freq = ckb_len_to_freq(lon_ckb_len, lat_ckb_len, dep_ckb_len, dx, dy, dz)
    index_list, vals_list = extract_vals_from_origin_file(tomo_file)
    
    
    scalar_list = ['vp', 'vs', 'rho']
    scalar_arr_list, minmax_list = [], []
    for scalar in scalar_list:
        origin_arr = read_tomo_array(tomo_file, scalar)
        ckb_arr = generating_checkerboard(origin_arr, lon_freq, lat_freq, dep_freq, amplitude, power)
        ckb_arr = gaussian_filter(ckb_arr, sigma=gaussian_sigma)
        minmax_tmp = calculating_minmax(ckb_arr)
        scalar_arr_list.append(ckb_arr)
        minmax_list.append(minmax_tmp)
        
    output_table(output_file, lines_list, minmax_list, index_list, vals_list, scalar_arr_list)
    
# %%
