from pyproj import Proj
import pandas as pd
import numpy as np

def lonlat_to_utm(lon, lat, utm_zone=50, is_north_hemisphere=True):
    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', north=is_north_hemisphere)
    x, y = utm_proj(lon, lat)
    return x, y

def utm_to_lonlat(x, y, utm_zone=50, is_north_hemisphere=True):
    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', north=is_north_hemisphere)
    lon, lat = utm_proj(x, y, inverse=True)
    return lon, lat

if __name__ == '__main__':
    
    use_utm_to_transform = 0
    utm_zone = 50
    is_north_hemisphere = True
    x, y = 120.5, 22.75
    
    if use_utm_to_transform:
        x_new, y_new = utm_to_lonlat(x, y, utm_zone=50, is_north_hemisphere=is_north_hemisphere)
    else:
        x_new, y_new = lonlat_to_utm(x, y, utm_zone=50, is_north_hemisphere=is_north_hemisphere)
    
    print(f'{x}, {y} -> {x_new}, {y_new}')

