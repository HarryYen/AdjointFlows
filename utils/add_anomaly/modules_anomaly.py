from pyproj import Proj
import numpy as np
import sys
import os

# ----------------------------------------------------------------------
# Small tools for UTM and lon/lat conversion 
# ----------------------------------------------------------------------
def lonlat_to_utm(lon, lat, utm_zone=50, is_north_hemisphere=True):
    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', north=is_north_hemisphere)
    x, y = utm_proj(lon, lat)
    return x, y

def utm_to_lonlat(x, y, utm_zone=50, is_north_hemisphere=True):
    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', north=is_north_hemisphere)
    lon, lat = utm_proj(x, y, inverse=True)
    
    return lon, lat


# ------------------------------------------------------------------
# Functions for reading and writing binary files
# ------------------------------------------------------------------
def detect_endian_single_record(path):
    size = os.path.getsize(path)
    with open(path, 'rb') as f:
        head_bytes = f.read(4)
        f.seek(size - 4)
        tail_bytes = f.read(4)

    head_le = np.frombuffer(head_bytes, dtype='<i4')[0]
    tail_le = np.frombuffer(tail_bytes, dtype='<i4')[0]
    head_be = np.frombuffer(head_bytes, dtype='>i4')[0]
    tail_be = np.frombuffer(tail_bytes, dtype='>i4')[0]

    if head_le == tail_le and (8 + head_le) == size:
        return '<', head_le
    if head_be == tail_be and (8 + head_be) == size:
        return '>', head_be
    return None, None

def read_binary_float32(input_file):
 
    endian, nbytes = detect_endian_single_record(input_file)
    if endian is None:
        raise ValueError("Not single record file.")

    i4 = endian + 'i4'
    f4 = endian + 'f4'
    data_type = f4
    with open(input_file, 'rb') as fi:
        # read header 
        head = np.fromfile(fi, dtype=i4, count=1)[0]
        if head != nbytes:
            raise ValueError("record length not match.")

        # read data
        count = nbytes // 4
        data = np.fromfile(fi, dtype=f4, count=count)
        tail = np.fromfile(fi, dtype=i4, count=1)[0]
        if tail != head or data.nbytes != nbytes:
            raise ValueError("record length not match.")
    
    return data, data_type 

def read_binary_int32(input_file):
 
    endian, nbytes = detect_endian_single_record(input_file)
    if endian is None:
        raise ValueError("Not single record file.")

    i4 = endian + 'i4'
    data_type = i4
    with open(input_file, 'rb') as fi:
        # read header 
        head = np.fromfile(fi, dtype=i4, count=1)[0]
        if head != nbytes:
            raise ValueError("record length not match.")

        # read data
        count = nbytes // 4
        data = np.fromfile(fi, dtype=i4, count=count)
        tail = np.fromfile(fi, dtype=i4, count=1)[0]
        if tail != head or data.nbytes != nbytes:
            raise ValueError("record length not match.")
    
    return data, data_type

def output_binary_file(data, outfile, data_type_int, data_type_float):
    i4 = data_type_int
    f4 = data_type_float
    with open(outfile, 'wb') as fo:
        head = np.array([data.nbytes], dtype=i4)
        tail = head

        head.tofile(fo)
        data.astype(f4, copy=False).tofile(fo)
        tail.tofile(fo)


# ------------------------------------------------------------------
# Functions for adding anomalies
# ------------------------------------------------------------------
def add_constant_perturb(data, constant):
    return data + constant

def add_gaussian_perturb(data, x_arr, y_arr, z_arr, center, amplitude, sigma):
    """
    Add Gaussian perturbation to 3D data array.
    Args:
        data: 3D numpy array of the original data.
        x_arr, y_arr, z_arr: 3D numpy arrays of the coordinates.
        center: tuple of (x0, y0, z0) for the center of the Gaussian.
        amplitude: Amplitude of the Gaussian perturbation.
        sigma: tuple of (sx, sy, sz) for the standard deviations in each direction.
    """
    # Check inputs
    if not (isinstance(center, (list, tuple)) and len(center) == 3):
        raise ValueError(f"'center' need to be 3-element tuple, e.g. (x0, y0, z0)")
    if not (isinstance(sigma, (list, tuple)) and len(sigma) == 3):
        raise ValueError(f"'sigma' need to be 3-element tuple, e.g. (sx, sy, sz)")
    # ----------------------------------
    x0, y0, z0 = center
    sx, sy, sz = sigma

    r2 = ((x_arr - x0)**2) / (2 * sx**2) + ((y_arr - y0)**2) / (2 * sy**2) + ((z_arr - z0)**2) / (2 * sz**2)
    gauss = np.exp(-r2)

    dv = data * amplitude * gauss
    data_new = data + dv

    return data_new

def build_spaced_gaussian_ckb(
    lon_arr, lat_arr, x_arr, y_arr, z_arr,
    lon_centers, lat_centers, z_centers,
    x_grid, y_grid,
    dlon_deg, dlat_deg, utm_zone, is_north_hemisphere, dz, sigma,
    n_sigma, amplitude):
    
    lon_min, lon_max = lon_arr.min(), lon_arr.max()
    lat_min, lat_max = lat_arr.min(), lat_arr.max()    
    z_min, z_max = z_arr.min(), z_arr.max()
    
    sx, sy, sz = sigma
    
    ix_valid = np.where(
        (lon_centers >= lon_min - dlon_deg/2) &
        (lon_centers <= lon_max + dlon_deg/2)
    )[0]

    iy_valid = np.where(
        (lat_centers >= lat_min - dlat_deg/2) &
        (lat_centers <= lat_max + dlat_deg/2)
    )[0]

    iz_valid = np.where(
        (z_centers >= z_min - dz/2) &
        (z_centers <= z_max + dz/2)
    )[0]

    # Pattern for -1 / 0 / +1 checkerboard
    # index: 0 -> -1, 1 -> 0, 2 -> +1
    pattern_values = np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    pert = np.zeros_like(lon_arr, dtype=np.float32)
    for ix in ix_valid:
        for iy in iy_valid:
       
            cx, cy = x_grid[ix, iy], y_grid[ix, iy]
            dx_all = x_arr - cx
            dy_all = y_arr - cy
            
            xy_mask = (
                (np.abs(dx_all) <= n_sigma * sx) &
                (np.abs(dy_all) <= n_sigma * sy)
            )
            if not np.any(xy_mask):
                continue
            
            idx_xy = np.where(xy_mask)[0]
            dx = dx_all[idx_xy]
            dy = dy_all[idx_xy]
            z_local_all = z_arr[idx_xy]
            
            base_r2 = (dx*dx) / (2.0 * sx*sx) + (dy*dy) / (2.0 * sy*sy)
            
            for iz in iz_valid:
                cz = z_centers[iz]

                dz_all = z_local_all - cz
                z_mask = np.abs(dz_all) <= n_sigma * sz
                
                if not np.any(z_mask):
                    continue
                
                idx = idx_xy[z_mask]
                dz_local = dz_all[z_mask]
                
                r2 = base_r2[z_mask] + (dz_local*dz_local) / (2.0 * sz*sz)
                gauss = np.exp(-r2)

                # Choose pattern index based on 3D index sum
                pattern_idx = (ix + iy + iz) % 4   # 0, 1, or 2
                sign = pattern_values[pattern_idx] # -1, 0, or +1

                pert[idx] += sign * amplitude * gauss

    return pert


def build_gaussian_ckb(
    lon_arr, lat_arr, x_arr, y_arr, z_arr,
    lon_centers, lat_centers, z_centers,
    x_grid, y_grid,
    dlon_deg, dlat_deg, utm_zone, is_north_hemisphere, dz, sigma,
    n_sigma, amplitude):
    
    lon_min, lon_max = lon_arr.min(), lon_arr.max()
    lat_min, lat_max = lat_arr.min(), lat_arr.max()    
    z_min, z_max = z_arr.min(), z_arr.max()
    
    sx, sy, sz = sigma
    
    ix_valid = np.where(
    (lon_centers >= lon_min - dlon_deg/2) &
    (lon_centers <= lon_max + dlon_deg/2))[0]

    iy_valid = np.where(
        (lat_centers >= lat_min - dlat_deg/2) &
        (lat_centers <= lat_max + dlat_deg/2))[0]

    iz_valid = np.where(
        (z_centers >= z_min - dz/2) &
        (z_centers <= z_max + dz/2))[0]

    # Sign for +/- checkerboard
    sign_x = ((np.arange(len(lon_centers)) % 2) * 2 - 1)
    sign_y = ((np.arange(len(lat_centers)) % 2) * 2 - 1)
    sign_z = ((np.arange(len(z_centers)) % 2) * 2 - 1)


    pert = np.zeros_like(lon_arr, dtype=np.float32)
    for ix in ix_valid:
        for iy in iy_valid:
       
            cx, cy = x_grid[ix, iy], y_grid[ix, iy]
            dx_all = x_arr - cx
            dy_all = y_arr - cy
            
            xy_mask = (
                (np.abs(dx_all) <= n_sigma * sx) &
                (np.abs(dy_all) <= n_sigma * sy)
            )
            if not np.any(xy_mask):
                continue
            
            idx_xy = np.where(xy_mask)[0]
            dx = dx_all[idx_xy]
            dy = dy_all[idx_xy]
            z_local_all = z_arr[idx_xy]
            
            base_r2 = (dx*dx) / (2.0 * sx*sx) + (dy*dy) / (2.0 * sy*sy)
            
            for iz in iz_valid:
                cz = z_centers[iz]

                dz_all = z_local_all - cz
                z_mask = np.abs(dz_all) <= n_sigma * sz
                
                if not np.any(z_mask):
                    continue
                
                idx = idx_xy[z_mask]
                dz_local = dz_all[z_mask]
                
                r2 = base_r2[z_mask] + (dz_local*dz_local) / (2.0 * sz*sz)
                gauss = np.exp(-r2)

                sign = sign_x[ix] * sign_y[iy] * sign_z[iz]
                pert[idx] += sign * amplitude * gauss

    return pert




def suggest_dx_from_sigma(sigma, mid_value=0.5):
    """
    Suggest checkerboard spacing dx for a given Gaussian sigma.

    We assume neighboring Gaussian centers are spaced by dx.
    At the midpoint between two centers (dx/2 from each),
    the Gaussian amplitude is required to be `mid_value`:

        mid_value = exp(-(dx/2)^2 / (2 * sigma^2))

    Solving for dx gives:

        dx = 2 * sigma * sqrt(-2 * ln(mid_value))

    Args:
        sigma (float): Standard deviation of the Gaussian (sx, sy, or sz).
        mid_value (float): Desired Gaussian value at the midpoint between centers,
                           0 < mid_value < 1. e.g. 0.5 ~ FWHM spacing.

    Returns:
        float: Suggested checkerboard spacing dx.
    """
    if not (0.0 < mid_value < 1.0):
        raise ValueError("mid_value must be between 0 and 1 (exclusive).")

    dx = 2.0 * sigma * np.sqrt(-2.0 * np.log(mid_value))
    return dx


def suggest_sigma_from_dx(edge_amplitude_ratio, dx):
    """
    Suggest Gaussian sigma for a given checkerboard spacing dx.

    We assume neighboring Gaussian centers are spaced by dx.
    At the midpoint between two centers (dx/2 from each),
    the Gaussian amplitude is required to be `edge_amplitude_ratio`:

        edge_amplitude_ratio = exp(-(dx/2)^2 / (2 * sigma^2))

    Solving for sigma gives:

        sigma = (dx/2) / sqrt(-2 * ln(edge_amplitude_ratio))

    Args:
        edge_amplitude_ratio (float): Desired Gaussian value at the midpoint between centers,
                                      0 < edge_amplitude_ratio < 1. e.g. 0.5 ~ FWHM spacing.
        dx (float): Checkerboard spacing.

    Returns:
        float: Suggested Gaussian sigma.
    """
    if not (0.0 < edge_amplitude_ratio < 1.0):
        raise ValueError("edge_amplitude_ratio must be between 0 and 1 (exclusive).")

    sigma = (dx / 2.0) / np.sqrt(-2.0 * np.log(edge_amplitude_ratio))
    return sigma