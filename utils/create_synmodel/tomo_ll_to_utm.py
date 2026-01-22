from pyproj import Proj
import pandas as pd
import numpy as np

def lonlat_to_utm(lon, lat, utm_zone=50, is_north_hemisphere=True):
    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', north=is_north_hemisphere)
    x, y = utm_proj(lon, lat)
    return x, y

if __name__ == "__main__":

    # ---------------------------------
    # Parameters
    # ---------------------------------
    given_tomo_file = 'tomo_1d.xyz'
    output_tomo_file = 'tomo_1d_utm.xyz'
    utm_zone = 50
    # ---------------------------------

    df = pd.read_csv(given_tomo_file, sep=r'\s+', skiprows=4,
                 names=['lon', 'lat', 'dep', 'vp', 'vs', 'rho'])
    lon_arr = df['lon'].values
    lat_arr = df['lat'].values
    dep_arr = df['dep'].values

    x_arr, y_arr = lonlat_to_utm(lon=lon_arr, lat=lat_arr,
                                 utm_zone=utm_zone, is_north_hemisphere=True)
    print(y_arr)
