from plotting_modules import interp_2d_in_specific_dep, find_nxnynz_from_xyz_file, find_minmax_from_xyz_file
from tools.job_utils import check_dir_exists
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
import numpy as np
from pyproj import Proj
import pygmt
import sys
import os

def utm_to_lonlat(x, y, utm_zone=50, is_north_hemisphere=True):
    utm_proj = Proj(proj='utm', zone=utm_zone, ellps='WGS84', north=is_north_hemisphere)
    lon, lat = utm_proj(x, y, inverse=True)
    return lon, lat

def ellipse_xy(center, fullwidth_x, fullwidth_y, angle_deg=0, n=360):
    """
    Generate (x, y) coordinates outlining an ellipse using given full widths.

    Parameters
    ----------
    center : tuple of float
        The (x0, y0) coordinates of the ellipse center.
    fullwidth_x : float
        Full width (diameter) of the ellipse in the x-direction.
    fullwidth_y : float
        Full width (diameter) of the ellipse in the y-direction.
    angle_deg : float, optional
        Rotation angle of the ellipse major axis, in degrees. Default is 0.
    n : int, optional
        Number of discrete points along the ellipse perimeter. Default is 360.

    Returns
    -------
    x, y : ndarray
        Arrays of x and y coordinates representing the ellipse boundary.
    """
    x0, y0 = center
    a = fullwidth_x / 2.0  # Semi-major axis
    b = fullwidth_y / 2.0  # Semi-minor axis
    theta = np.linspace(0, 2 * np.pi, n, endpoint=True)
    X = a * np.cos(theta)
    Y = b * np.sin(theta)

    # Apply rotation
    ang = np.deg2rad(angle_deg)
    xr = X * np.cos(ang) - Y * np.sin(ang)
    yr = X * np.sin(ang) + Y * np.cos(ang)
    
    return x0 + xr, y0 + yr

def sigma_to_fullwidth(sigma):
    """
    Convert Gaussian sigma to full width at half maximum (FWHM).

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    float
        Full width at half maximum (FWHM).
    """
    return np.sqrt(8) * sigma

def plot_horizontal_slices_grad_diff(grad_diff_file, model_peturb_file, model_ref_file, output_dir):

    # -------------------------------------------
    # Parameters
    # -------------------------------------------
    map_region = [119, 123.5, 21.5, 26.]
    dep = 10 

    grad_scalar = 'beta'
    model_scalar = 'vs'
    normalize_grad_flag = True
    
    grad_range = [0, 1]
    model_range = [-0.3, 0.3]
    
    # parameter of the anomaly (PSF)
    utm_zone = 50
    is_north_hemisphere = True
    anomaly_x, anomaly_y = 9.5e+05, 2.7e+06  # in UTM (m)
    sigma_h, sigma_v = 3, 3  # in km
    gaussian_color = '#5DC200'
    pen_width = 0.7
    
    cmap_grad_diff = 'hot'
    reverse_cmap_grad_diff = True
    cbar_label_grad_diff = 'Normalized H@~d@~m'

    cmap_model_diff = 'polar'
    reverse_cmap_model_diff = True
    cbar_label_model_diff = f'{model_scalar} perturb (km/s)'
    
    # -------------------------------------------
    # Load model and gradient difference file
    # -------------------------------------------
    
    grad_diff_df = pd.read_csv(grad_diff_file, sep='\s+', skiprows=5, usecols=[0, 1, 2, 3, 4, 5],
                            names=['lon', 'lat', 'dep', 'alpha', 'beta', 'rho'])
    model_perturb_df = pd.read_csv(model_peturb_file, sep='\s+', skiprows=5, usecols=[0, 1, 2, 3, 4, 5],
                            names=['lon', 'lat', 'dep', 'vp', 'vs', 'rho'])
    model_ref_df = pd.read_csv(model_ref_file, sep='\s+', skiprows=5, usecols=[0, 1, 2, 3, 4, 5],
                            names=['lon', 'lat', 'dep', 'vp', 'vs', 'rho'])
    
    ### Get some grid info
    lon_min, lon_max, lat_min, lat_max, dep_min, dep_max = find_minmax_from_xyz_file(input_file=grad_diff_file)
    nx, ny, nz = find_nxnynz_from_xyz_file(input_file=grad_diff_file)
    
    grd_region = [lon_min, lon_max, lat_min, lat_max]
    
    lon_uniq = np.unique(grad_diff_df['lon'].values)
    lat_uniq = np.unique(grad_diff_df['lat'].values)
    dep_uniq = np.unique(grad_diff_df['dep'].values)
    
    ### Get model and gradient difference array
    model_diff_arr = (model_perturb_df[model_scalar].values - model_ref_df[model_scalar].values).reshape(nz, nx, ny)
    grad_diff_arr = grad_diff_df[grad_scalar].values.reshape(nz, nx, ny)
    
    # -------------------------------------------
    # Post-processing
    # -------------------------------------------
    # Calculate the ellipse for the gaussian range
    sigma_h = sigma_h * 1e3  # convert to meter
    sigma_v = sigma_v * 1e3  # convert to meter
    full_width_x, full_width_y = sigma_to_fullwidth(sigma_h), sigma_to_fullwidth(sigma_v)
    ex, ey = ellipse_xy(center=(anomaly_x, anomaly_y), 
                        fullwidth_x=full_width_x, fullwidth_y=full_width_y, 
                        angle_deg=0, n=360)
    elon, elat = utm_to_lonlat(ex, ey, utm_zone=utm_zone, is_north_hemisphere=is_north_hemisphere)
    
    ## Interpolate to specific depth
    model_diff_interp_df = interp_2d_in_specific_dep(lon_uniq, lat_uniq, dep_uniq, model_diff_arr, dep)
    grad_diff_interp_df = interp_2d_in_specific_dep(lon_uniq, lat_uniq, dep_uniq, grad_diff_arr, dep)
    
    ## Take absolute values and Normalize gradient difference
    grad_diff_interp_df['scalar'] = np.abs(grad_diff_interp_df['scalar'].values)
    if normalize_grad_flag:
        grad_max = grad_diff_interp_df['scalar'].max()
        grad_diff_interp_df['scalar'] = grad_diff_interp_df['scalar'].values / grad_max
        
    # -------------------------------------------
    # Plotting -- Model Difference
    # -------------------------------------------
    fig = pygmt.Figure()
    pygmt.config(FONT_ANNOT_PRIMARY='12p,Helvetica')
    pygmt.config(FONT_LABEL='16p,Helvetica')
    
    pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")

    model_diff_grd = pygmt.xyz2grd(data=model_diff_interp_df,  
                region=grd_region, 
                spacing=f'{nx}+n/{ny}+n',
                verbose='q')

    model_diff_grd_fine = pygmt.grdsample(grid=model_diff_grd, spacing=0.01, 
                    region=grd_region, interpolation='l',
                    verbose='q')

    pygmt.makecpt(cmap=cmap_model_diff, series=model_range, reverse=reverse_cmap_model_diff)
    
    fig.grdimage(grid=model_diff_grd_fine, cmap=True, region=grd_region, projection='M7c', frame=True)
    fig.plot(x=elon, y=elat, pen=f'{pen_width}p,{gaussian_color}', close=True)
    
    fig.coast(shorelines=True)
    
    fig.text(text=f"dep: {dep:3d}km", font="14p,Helvetica-Bold", position="BR", frame=True)
    
    model_range_length = model_range[1] - model_range[0]
    fig.colorbar(frame=f'a{model_range_length/2}f{model_range_length/2}+l{cbar_label_model_diff}', 
                 position='JBC+w3.5c/0.2c+h+o0c/1c')

    # -------------------------------------------
    # Plotting -- Gradient Difference
    # -------------------------------------------
    fig.shift_origin(xshift='9c')
    
    grad_diff_grd = pygmt.xyz2grd(data=grad_diff_interp_df,  
                region=grd_region, 
                spacing=f'{nx}+n/{ny}+n',
                verbose='q')

    grad_diff_grd_fine = pygmt.grdsample(grid=grad_diff_grd, spacing=0.01, 
                    region=grd_region, interpolation='l',
                    verbose='q')

    pygmt.makecpt(cmap=cmap_grad_diff, series=grad_range, reverse=reverse_cmap_grad_diff)
    
    fig.grdimage(grid=grad_diff_grd_fine, cmap=True, region=grd_region, projection='M7c', frame=True)
    fig.plot(x=elon, y=elat, pen=f'{pen_width}p,{gaussian_color}', close=True)
    
    fig.coast(shorelines=True)
    
    fig.text(text=f"dep: {dep:3d}km", font="14p,Helvetica-Bold", position="BR", frame=True)
    
    grad_range_length = grad_range[1] - grad_range[0]
    fig.colorbar(frame=f'a{grad_range_length/2}f{grad_range_length/2}+l{cbar_label_grad_diff}', 
                 position='JBC+w3.5c/0.2c+h+o0c/1c')
    
    output_dir_for_fig = os.path.join(output_dir, 'fig', 'horizontal_slices_grad_diff')
    check_dir_exists(output_dir_for_fig)
    fig.savefig(f'{output_dir_for_fig}/PSF_{grad_scalar}_{grad_scalar}_{dep}.png', dpi=300,
                transparent=True)