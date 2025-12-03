from plotting_modules import read_profile_input_file, find_minmax_from_df, create_target_grid, interp_from_array_to_profile, calculate_profile_division_points, grab_earthqaukes
from tools.job_utils import check_dir_exists
import pandas as pd
import numpy as np
import yaml
import pygmt
import sys
import os

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


def plot_vertical_slices_grad_diff(grad_diff_file, model_peturb_file, model_ref_file, output_dir):
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, 'plot_config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    # ----------------------------------------------------------------------
    # Load the configuration
    # ----------------------------------------------------------------------

    map_region = [119, 123.5, 21.5, 26.]
    dep_range  = [0, 200]
    dep_interval_for_interp = 2
    interval_for_profile = 2
    width_profile = [-5, 5]
    topo_range_for_plot = [-8000, 4000]
    neighbor_spacing = 0.1
    search_radius = 10

    mapview_flag = True
    grad_scalar = 'beta'
    model_scalar = 'vs'
    normalize_grad_flag = True
    
    grad_range = [0, 1]
    model_range = [-0.3, 0.3]
    
    # parameter of the anomaly (PSF)
    anomaly_x, anomaly_z = 0, 10  # in km
    sigma_h, sigma_v = 3, 3  # in km
    gaussian_color = '#5DC200'
    
    cmap_grad_diff = 'hot'
    reverse_cmap_grad_diff = True
    cbar_label_grad_diff = 'Normalized H@~d@~m'

    cmap_model_diff = 'polar'
    reverse_cmap_model_diff = True
    cbar_label_model_diff = f'{model_scalar} perturb (km/s)'

    
    topo_grd = '/home/harry/Work/AdjointFlows/adjointflows/visualizer/plotting_files/taiwan_topo.grd'
    profile_input_file = '/home/harry/Work/AdjointFlows/adjointflows/visualizer/plotting_files/profile_grad_diff.input'
    # ----------------------------------------------------------------------
    # Main 
    # ----------------------------------------------------------------------
    profile_info_df = read_profile_input_file(file=profile_input_file)

    print(f'plot the model read from {grad_diff_file}')
    grad_diff_df = pd.read_csv(grad_diff_file, sep='\s+', skiprows=5, usecols=[0, 1, 2, 3, 4, 5],
                            names=['lon', 'lat', 'dep', 'alpha', 'beta', 'rho'])
    model_perturb_df = pd.read_csv(model_peturb_file, sep='\s+', skiprows=5, usecols=[0, 1, 2, 3, 4, 5],
                            names=['lon', 'lat', 'dep', 'vp', 'vs', 'rho'])
    model_ref_df = pd.read_csv(model_ref_file, sep='\s+', skiprows=5, usecols=[0, 1, 2, 3, 4, 5],
                            names=['lon', 'lat', 'dep', 'vp', 'vs', 'rho'])
    
    # Calculate the ellipse for the gaussian range
    full_width_x, full_width_z = sigma_to_fullwidth(sigma_h), sigma_to_fullwidth(sigma_v)
    ex, ez = ellipse_xy(center=(anomaly_x, anomaly_z), 
                        fullwidth_x=full_width_x, fullwidth_y=full_width_z, 
                        angle_deg=0, n=360)
    

    for profile_index, profile_info in profile_info_df.iterrows():
        profile_index = profile_index + 1
        val = str(profile_info['profile_name'])
        try:
            letter_index = float(val)
            letter_index = int(letter_index)
        except ValueError:
            letter_index = val


        azi_profile = profile_info['angle']
        len_profile = [profile_info['lmin'], profile_info['lmax']]
        center = [profile_info['clon'], profile_info['clat']]
        print(f'Profile {profile_index}: center: {center}, angle:{azi_profile}, length:{len_profile}')
        
        # ---------------------------------------------------------------------------       
        # Preprocessing for the gradient difference
        # 1. Constructing the query grid
        # 2. Interpolating onto the specified plane
        # 3. Using GMT function to make the uneven points onto regular grid (neighbor or surface)
        # ---------------------------------------------------------------------------
        profile_range = [len_profile[0], len_profile[1], dep_range[0], dep_range[1]]
        grad_diff_filter_df = grad_diff_df[['lon', 'lat', 'dep', grad_scalar]]
        

        grad_diff_pro_arr = pygmt.project(
            data = grad_diff_filter_df,
            center = center,
            azimuth = azi_profile,
            length = [len_profile[0]*1.1, len_profile[1]*1.1],
            unit = True,
            width = width_profile,
        )
        
        pro_line = pygmt.project(
            data = grad_diff_filter_df,
            center = center,
            azimuth = azi_profile,
            length = len_profile,
            unit = True,
            generate = interval_for_profile,
        )

        dep_arr = np.linspace(dep_range[0], dep_range[1], num=int((dep_range[1] - dep_range[0]) / dep_interval_for_interp) + 1)
        
        target_grid = create_target_grid(pro_line.r.values, pro_line.s.values, dep_arr)
        profile_points = np.array([grad_diff_pro_arr[0].values, grad_diff_pro_arr[1].values, grad_diff_pro_arr[2].values]).T
        grad_diff_profile_values = grad_diff_pro_arr[3].values
        
        grad_diff_interp = interp_from_array_to_profile(profile_points, grad_diff_profile_values, target_grid)
        
        x_grid, z_grid = np.meshgrid(pro_line.p.values, dep_arr, indexing='ij')
        x_grid_flat = x_grid.flatten()
        z_grid_flat = z_grid.flatten()
        
        divided_points_index_list = calculate_profile_division_points(len(pro_line), 4)

        # pro_surf_arr = pygmt.surface(
        #     x = x_grid_flat,
        #     y = z_grid_flat,
        #     z = grad_diff_interp,
        #     region = [len_profile[0], len_profile[1], dep_range[0], dep_range[1]],
        #     spacing = surface_spacing,
        # )
        grad_diff_surf_arr = pygmt.nearneighbor(
            x=x_grid_flat, y=z_grid_flat, z=grad_diff_interp,
            region=[len_profile[0], len_profile[1], dep_range[0], dep_range[1]],
            spacing=neighbor_spacing,
            search_radius=search_radius, 
        )       
        grad_diff_surf_arr = np.abs(grad_diff_surf_arr)
        if normalize_grad_flag:
            grad_diff_surf_arr = grad_diff_surf_arr / np.max(grad_diff_surf_arr)

        # ---------------------------------------------------------------------------       
        # Preprocessing for the model difference
        # 0. The query grids are the same as above
        # 1. Interpolating onto the specified plane
        # 2. Using GMT function to make the uneven points onto regular grid (neighbor or surface)
        # ---------------------------------------------------------------------------
        model_perturb_filter_df = model_perturb_df[['lon', 'lat', 'dep', model_scalar]]
        model_ref_filter_df = model_ref_df[['lon', 'lat', 'dep', model_scalar]]
        
        model_diff_filter_df = model_perturb_filter_df.copy()
        model_diff_filter_df[model_scalar] = model_perturb_filter_df[model_scalar] - model_ref_filter_df[model_scalar]
        
        model_diff_pro_arr = pygmt.project(
            data = model_diff_filter_df,
            center = center,
            azimuth = azi_profile,
            length = [len_profile[0]*1.1, len_profile[1]*1.1],
            unit = True,
            width = width_profile,
        )
        
        model_diff_profile_values = model_diff_pro_arr[3].values
        
        model_diff_interp = interp_from_array_to_profile(profile_points, model_diff_profile_values, target_grid)


        model_diff_surf_arr = pygmt.nearneighbor(
            x=x_grid_flat, y=z_grid_flat, z=model_diff_interp,
            region=[len_profile[0], len_profile[1], dep_range[0], dep_range[1]],
            spacing=neighbor_spacing,
            search_radius=search_radius, 
        )

        # -------------------------------------------
        # ------------ GMT config settings ----------
        # -------------------------------------------
        fig = pygmt.Figure()
        pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")
        pygmt.config(FONT='30p')
        pygmt.config(FONT_LABEL='36p')
        pygmt.config(MAP_FRAME_PEN='4p')

        # ---------------------------------------------------
        # Plotting for the mapview of the profile location
        # ---------------------------------------------------
        if mapview_flag:
            fig.basemap(region=map_region, projection='M4.5i', frame=["neWS","a2f1"])

            gradient_data = pygmt.grdgradient(
                grid      = topo_grd,
                azimuth   = [45, 135], 
                normalize = 'e0.7'
            )

            pygmt.makecpt(
                cmap = 'gray',
                series = topo_range_for_plot, 
            )

            fig.grdimage(
                region     = map_region, 
                grid       = topo_grd, 
                shading    = gradient_data,
                cmap = True,
                transparency = 60,
            )
            fig.coast(shorelines='1.5p', resolution='h')
            fig.plot(x = [pro_line.r.iloc[0], pro_line.r.iloc[-1]],
                    y = [pro_line.s.iloc[0], pro_line.s.iloc[-1]],
                    pen = '4p,black',
                    style = 'f-2/0.75c',
            )
            fig.plot(
                x = pro_line.r.values[divided_points_index_list],
                y = pro_line.s.values[divided_points_index_list],
                pen = '1p,black',
                style = 'c0.3c',
                fill = '#8c8cea')
            
            fig.text(
                x = [pro_line.r.iloc[0], pro_line.r.iloc[-1]],
                y = [pro_line.s.iloc[0], pro_line.s.iloc[-1]],
                text = [letter_index, f"{letter_index}'"],
                font = '32p,black',
                justify = 'MC',
                fill = '#fff683'
            )

            fig.shift_origin(xshift='6.7i')
        # ---------------------------------------------------
        # Plotting for the vertical slice of model difference
        # ---------------------------------------------------
        pygmt.makecpt(cmap=cmap_model_diff, series=model_range, reverse=reverse_cmap_model_diff)
        fig.basemap(region=profile_range, projection='X12i/-8i', frame=['WSne', 'a40f20', 'x+lDistance (km)', 'y+lDepth (km)'])
        fig.grdimage(grid=model_diff_surf_arr, cmap=True)
        
        # plot the gaussian range
        fig.plot(x=ex, y=ez, pen=f"3.5p,{gaussian_color}")
        
        fig.shift_origin(yshift='8.5i')
        topo_track = pygmt.grdtrack(
                        grid = topo_grd,
                        points = pro_line[['r', 's', 'p']],
                        newcolname = 'topo',
        )
        
        fig.basemap(region=[profile_range[0], profile_range[1], -1.2, 1.2],
                    projection='X12i/0.8i', frame=['WE'])

        dist_arr = np.array(topo_track.p)
        topo_arr = np.array(topo_track.topo) / np.max(np.abs(np.array(topo_track.topo)))
        
        dist_arr = np.concatenate(([dist_arr[0]], dist_arr, [dist_arr[-1]]))
        topo_arr = np.concatenate(([0], topo_arr, [0]))
        
        
        fig.plot(x = dist_arr[topo_arr <= 0], y = topo_arr[topo_arr <= 0], pen='1.5p,black',
            fill='#bcdaff', close=f"+y0")
        fig.plot(x = dist_arr[topo_arr >= 0], y = topo_arr[topo_arr >= 0], pen='1.5p,black',
            fill='#fce2bb', close=f"+y0")

        fig.plot(x = dist_arr[divided_points_index_list], y = np.zeros(len(divided_points_index_list)), 
                pen='1p,black', style='c0.5c', fill='#8c8cea')
        
        model_range_length = model_range[1] - model_range[0]
        fig.colorbar(frame=[f'a{model_range_length/4}f{model_range_length/4}', f'x+l{cbar_label_model_diff}'], position='JMR+o-40c/-4c+w10c/0.5c+ml', cmap=True)

        
        offset = np.abs(len_profile[1] - len_profile[0]) * 0.025
        fig.text(x = [profile_range[0] + offset, profile_range[1] - offset], y = [-1, -1],
                text = [letter_index, f"{letter_index}'"], font='45p,black', justify='MC', fill='#f6fc67', no_clip=True)      
        
        
    
        fig.shift_origin(yshift='-8.5i')
        fig.shift_origin(xshift='14i')
        # ---------------------------------------------------
        # Plotting for the vertical slice of gradient difference
        # ---------------------------------------------------

        pygmt.makecpt(cmap=cmap_grad_diff, series=grad_range, reverse=reverse_cmap_grad_diff)
        fig.basemap(region=profile_range, projection='X12i/-8i', frame=['WSne', 'a40f20', 'x+lDistance (km)', 'y+lDepth (km)'])
        fig.grdimage(grid=grad_diff_surf_arr, cmap=True)
        # plot the gaussian range
        fig.plot(x=ex, y=ez, pen=f"3.5p,{gaussian_color}")
        
        fig.shift_origin(yshift='8.5i')
        topo_track = pygmt.grdtrack(
                        grid = topo_grd,
                        points = pro_line[['r', 's', 'p']],
                        newcolname = 'topo',
        )
        
        fig.basemap(region=[profile_range[0], profile_range[1], -1.2, 1.2],
                    projection='X12i/0.8i', frame=['WE'])

        
        fig.plot(x = dist_arr[topo_arr <= 0], y = topo_arr[topo_arr <= 0], pen='1.5p,black',
            fill='#bcdaff', close=f"+y0")
        fig.plot(x = dist_arr[topo_arr >= 0], y = topo_arr[topo_arr >= 0], pen='1.5p,black',
            fill='#fce2bb', close=f"+y0")

        fig.plot(x = dist_arr[divided_points_index_list], y = np.zeros(len(divided_points_index_list)), 
                pen='1p,black', style='c0.5c', fill='#8c8cea')
        
        grad_range_length = grad_range[1] - grad_range[0]
        fig.colorbar(frame=[f'a{grad_range_length/2}f{grad_range_length/2}', f'x+l{cbar_label_grad_diff}'], position='JMR+o-80c/-4c+w10c/0.5c+ml', cmap=True)

        
        offset = np.abs(len_profile[1] - len_profile[0]) * 0.025
        fig.text(x = [profile_range[0] + offset, profile_range[1] - offset], y = [-1, -1],
                text = [letter_index, f"{letter_index}'"], font='45p,black', justify='MC', fill='#f6fc67', no_clip=True)
        
        output_dir_for_fig = os.path.join(output_dir, 'fig', 'vertical', 'grad_diff')
        check_dir_exists(output_dir_for_fig)
        fig.savefig(f'{output_dir_for_fig}/profile_{grad_scalar}_{model_scalar}_{int(azi_profile)}_{letter_index}.png', dpi=300,
                    transparent=True)
