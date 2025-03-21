from pyproj import Proj
from scipy.spatial import cKDTree
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import pandas as pd
import xarray as xr


def utm_to_lonlat(x, y, utm_zone=50, is_north_hemisphere=True):
    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', north=is_north_hemisphere)
    lon, lat = utm_proj(x, y, inverse=True)
    return lon, lat

def lonlat_to_utm(lon, lat, utm_zone=50, is_north_hemisphere=True):
    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', north=is_north_hemisphere)
    x, y = utm_proj(lon, lat)
    return x, y


def get_values_by_kdtree(query_lon, query_lat, query_dep, given_lon, given_lat, given_dep, data_arr, max_distance):
    """
    Interpolate the values by KDTree for producing regular table
    Becasue the specfem kernel is in GLL points, we need to interpolate the values to regular points
    
    Input (M points of known data, N points of query (unknown) data): 
        query_lon (np.array) : the longitude of the query points (N,)
        query_lat (np.array) : the latitude of the query points (N,)
        query_dep (np.array) : the depth of the query points (N,)
        given_lon (np.array) : the longitude of the given points (M,)
        given_lat (np.array) : the latitude of the given points (M,)
        given_dep (np.array) : the depth of the given points (M,)
        data_arr (np.array)  : the data array of the given points (M,)
    Output:
        interp_arr_reshaped (np.array) : the interpolated data array of the query (N,)
    """
    
    
    known_points = np.vstack([given_lon, given_lat, given_dep]).T
    tree = cKDTree(known_points)
    
    query_arr = np.vstack([query_lon, query_lat, query_dep]).T

    distances, indices = tree.query(query_arr, k=3)
    too_far = np.all(distances > max_distance, axis=1)
    # too_far = distances > max_distance
    neighbor_data = data_arr[indices]  # shape (num_query_points, k)
    weights = 1 / (distances + 1e-10) 
    weights /= weights.sum(axis=1, keepdims=True) 
    interp_arr = np.sum(neighbor_data * weights, axis=1)
    
    interp_arr[too_far] = np.nan
    interp_arr_reshaped = interp_arr.reshape(query_lon.shape)
    
    return interp_arr_reshaped

# ----------------------------------------------------------------------------------------------------
def find_minmax_from_xyz_file(input_file):
    with open(input_file, 'r') as f:
        for index, line in enumerate(f):
            if index == 0:
                values = line.strip().split()
                lon_min = float(values[0])
                lat_min = float(values[1])
                dep_min = float(values[2])
                lon_max = float(values[3])
                lat_max = float(values[4])
                dep_max = float(values[5])
    return lon_min, lon_max, lat_min, lat_max, dep_min, dep_max

def find_nxnynz_from_xyz_file(input_file):
    with open(input_file, 'r') as f:
        for index, line in enumerate(f):
            if index == 2:
                values = line.strip().split()
                nx = int(values[0])
                ny = int(values[1])
                nz = int(values[2])
    return nx, ny, nz

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

def to_xarray_2d(data, lon_arr_uniq, lat_arr_uniq, name):
    data_array = xr.DataArray(
    data, 
    dims=['lon', 'lat'],
    coords={'lon': lon_arr_uniq,  
            'lat': lat_arr_uniq},
    name=name 
)
    return data_array

