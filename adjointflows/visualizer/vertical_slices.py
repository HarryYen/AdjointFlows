from plotting_modules import read_profile_input_file, find_minmax_from_df, create_target_grid, interp_from_array_to_profile, calculate_profile_division_points, grab_earthqaukes

import pandas as pd
import numpy as np
import yaml
import pygmt
import sys
import os

def plot_eq(fig, eq_df, center, azi_profile, len_profile, width_profile):
    projected_eq = pygmt.project(
        data = eq_df,
        center = center,
        azimuth = azi_profile,
        length = [len_profile[0]*1.1, len_profile[1]*1.1],
        unit = True,
        width = width_profile,
    )
    try:
        fig.plot(
            x = projected_eq[3].values,
            y = projected_eq[2].values,
            style = 'c0.4c',
            fill = 'gray',
            pen = '0.1p,black'
        )
    except KeyError:
        pass
    
    return fig

def check_if_output_dir_exist(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    

def plot_vertical_slices_pert(input_dir, output_dir):
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, 'plot_config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    # ----------------------------------------------------------------------
    # Load the configuration
    # ----------------------------------------------------------------------
    
    config_vert = config['vertical_slice']
    map_region = config_vert['general_map']['map_region']
    dep_range  = config_vert['general_map']['dep_range']
    dep_interval_for_interp = config_vert['general_map']['dep_interval_for_interp']
    interval_for_profile = config_vert['general_map']['interval_for_profile']
    width_profile = config_vert['general_map']['width_profile']
    width_for_projecting_eq = config_vert['general_map']['width_for_projecting_eq']
    topo_range_for_plot = config_vert['general_map']['topo_range_for_plot']
    surface_spacing = config_vert['general_map']['surface_spacing']

    mapview_flag = config_vert['general_flag']['plot_mapview']
    plot_eq_flag = config_vert['general_flag']['plot_eq']
    
    scalar_list = config_vert['fine_tune_perturb']['scalar_list']
    scalar_range = config_vert['fine_tune_perturb']['scalar_range']
    cmap = config_vert['fine_tune_perturb']['cmap']
    reverse_cmap = config_vert['fine_tune_perturb']['reverse_cmap']
    
    topo_grd = config_vert['general_file']['topo_grd']
    eq_file = config_vert['general_file']['eq_file']
    profile_input_file = config_vert['general_file']['profile_input']
    # ----------------------------------------------------------------------
    # Main 
    # ----------------------------------------------------------------------
    profile_info_df = read_profile_input_file(file=profile_input_file)
    input_file = f'{input_dir}/model.xyz'
    print(f'plot the model read from {input_file}')
    xyz_df = pd.read_csv(input_file, sep='\s+', skiprows=5, 
                            names=['lon', 'lat', 'dep', 'vp', 'vs', 'rho', 'dvp', 'dvs', 'drho'])

    for scalar in scalar_list:
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
            # GMT pre-processing
            # ---------------------------------------------------------------------------
            profile_range = [len_profile[0], len_profile[1], dep_range[0], dep_range[1]]
            selected_df = xyz_df[['lon','lat','dep',scalar]]

            pro_arr = pygmt.project(
                data = selected_df,
                center = center,
                azimuth = azi_profile,
                length = [len_profile[0]*1.1, len_profile[1]*1.1],
                unit = True,
                width = width_profile,
            )

            pro_line = pygmt.project(
                data = selected_df,
                center = center,
                azimuth = azi_profile,
                length = len_profile,
                unit = True,
                generate = interval_for_profile,
            )
            
            dep_arr = np.linspace(dep_range[0], dep_range[1], num=int((dep_range[1] - dep_range[0]) / dep_interval_for_interp) + 1)
            
            target_grid = create_target_grid(pro_line.r.values, pro_line.s.values, dep_arr)
            profile_points = np.array([pro_arr[0].values, pro_arr[1].values, pro_arr[2].values]).T
            profile_values = pro_arr[3].values
            pro_arr_interp = interp_from_array_to_profile(profile_points, profile_values, target_grid)

            x_grid, z_grid = np.meshgrid(pro_line.p.values, dep_arr, indexing='ij')
            x_grid_flat = x_grid.flatten()
            z_grid_flat = z_grid.flatten()
            
            divided_points_index_list = calculate_profile_division_points(len(pro_line), 4)
            
            pro_surf_arr = pygmt.surface(
                x = x_grid_flat,
                y = z_grid_flat,
                z = pro_arr_interp,
                region = [len_profile[0], len_profile[1], dep_range[0], dep_range[1]],
                spacing = surface_spacing,
            )
            

            # -------------------------------------------
            # ------------ GMT config settings ----------
            # -------------------------------------------
            fig = pygmt.Figure()
            pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")
            pygmt.config(FONT='30p')
            pygmt.config(FONT_LABEL='36p')
            pygmt.config(MAP_FRAME_PEN='4p')

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

            pygmt.makecpt(cmap=cmap, series=scalar_range, reverse=reverse_cmap)
            fig.basemap(region=profile_range, projection='X12i/-8i', frame=['WSne', 'a40f20', 'x+lDistance (km)', 'y+lDepth (km)'])
            fig.grdimage(grid=pro_surf_arr, cmap=True)

            # plotting eq
            if plot_eq_flag:
                eq_df = grab_earthqaukes(eq_file)
                fig = plot_eq(fig, eq_df, center, azi_profile, len_profile, width_for_projecting_eq)


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
            
            fig.colorbar(frame=['a10f5', f'x+l{scalar} (%)'], position='JMR+o-35c/-4c+w10c/0.5c+ml', cmap=True)

            
            offset = np.abs(len_profile[1] - len_profile[0]) * 0.025
            fig.text(x = [profile_range[0] + offset, profile_range[1] - offset], y = [-1, -1],
                    text = [letter_index, f"{letter_index}'"], font='45p,black', justify='MC', fill='#f6fc67', no_clip=True)
            
            output_dir_for_fig = os.path.join(output_dir, 'fig', 'vertical', 'pert')
            check_if_output_dir_exist(output_dir=output_dir_for_fig)
            fig.savefig(f'{output_dir_for_fig}/profile_{scalar}_{int(azi_profile)}_{profile_index}.png', dpi=300,
                        transparent=True)



def plot_vertical_slices_abs(input_dir, output_dir):
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, 'plot_config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    # ----------------------------------------------------------------------
    # Load the configuration
    # ----------------------------------------------------------------------
    
    config_vert = config['vertical_slice']
    map_region = config_vert['general_map']['map_region']
    dep_range  = config_vert['general_map']['dep_range']
    dep_interval_for_interp = config_vert['general_map']['dep_interval_for_interp']
    interval_for_profile = config_vert['general_map']['interval_for_profile']
    width_profile = config_vert['general_map']['width_profile']
    width_for_projecting_eq = config_vert['general_map']['width_for_projecting_eq']
    topo_range_for_plot = config_vert['general_map']['topo_range_for_plot']
    surface_spacing = config_vert['general_map']['surface_spacing']

    mapview_flag = config_vert['general_flag']['plot_mapview']
    plot_eq_flag = config_vert['general_flag']['plot_eq']
    
    scalar_list = config_vert['fine_tune_abs']['scalar_list']
    scalar_range_list = config_vert['fine_tune_abs']['scalar_range_list']
    cmap = config_vert['fine_tune_abs']['cmap']
    reverse_cmap = config_vert['fine_tune_abs']['reverse_cmap']
    cbar_label_list = config_vert['fine_tune_abs']['cbar_label_list']
    plot_contour = config_vert['fine_tune_abs']['contour']['plot_contour']
    line_interval = config_vert['fine_tune_abs']['contour']['line_interval']
    annotation_interval = config_vert['fine_tune_abs']['contour']['annotation_interval']
    
    topo_grd = config_vert['general_file']['topo_grd']
    eq_file = config_vert['general_file']['eq_file']
    profile_input_file = config_vert['general_file']['profile_input']
    # ----------------------------------------------------------------------
    # Main 
    # ----------------------------------------------------------------------
    profile_info_df = read_profile_input_file(file=profile_input_file)
    input_file = f'{input_dir}/model.xyz'
    print(f'plot the model read from {input_file}')
    xyz_df = pd.read_csv(input_file, sep='\s+', skiprows=5, 
                            names=['lon', 'lat', 'dep', 'vp', 'vs', 'rho', 'dvp', 'dvs', 'drho'])

    for scalar_index, scalar in enumerate(scalar_list):
        scalar_range = scalar_range_list[scalar_index]
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
            # GMT pre-processing
            # ---------------------------------------------------------------------------
            profile_range = [len_profile[0], len_profile[1], dep_range[0], dep_range[1]]
            selected_df = xyz_df[['lon','lat','dep',scalar]]

            pro_arr = pygmt.project(
                data = selected_df,
                center = center,
                azimuth = azi_profile,
                length = [len_profile[0]*1.1, len_profile[1]*1.1],
                unit = True,
                width = width_profile,
            )

            pro_line = pygmt.project(
                data = selected_df,
                center = center,
                azimuth = azi_profile,
                length = len_profile,
                unit = True,
                generate = interval_for_profile,
            )
            
            dep_arr = np.linspace(dep_range[0], dep_range[1], num=int((dep_range[1] - dep_range[0]) / dep_interval_for_interp) + 1)
            
            target_grid = create_target_grid(pro_line.r.values, pro_line.s.values, dep_arr)
            profile_points = np.array([pro_arr[0].values, pro_arr[1].values, pro_arr[2].values]).T
            profile_values = pro_arr[3].values
            pro_arr_interp = interp_from_array_to_profile(profile_points, profile_values, target_grid)

            x_grid, z_grid = np.meshgrid(pro_line.p.values, dep_arr, indexing='ij')
            x_grid_flat = x_grid.flatten()
            z_grid_flat = z_grid.flatten()
            
            divided_points_index_list = calculate_profile_division_points(len(pro_line), 4)
            
            pro_surf_arr = pygmt.surface(
                x = x_grid_flat,
                y = z_grid_flat,
                z = pro_arr_interp,
                region = [len_profile[0], len_profile[1], dep_range[0], dep_range[1]],
                spacing = surface_spacing,
            )
            

            # -------------------------------------------
            # ------------ GMT config settings ----------
            # -------------------------------------------
            fig = pygmt.Figure()
            pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")
            pygmt.config(FONT='30p')
            pygmt.config(FONT_LABEL='36p')
            pygmt.config(MAP_FRAME_PEN='4p')

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

            pygmt.makecpt(cmap=cmap, series=scalar_range, reverse=reverse_cmap)
            fig.basemap(region=profile_range, projection='X12i/-8i', frame=['WSne', 'a40f20', 'x+lDistance (km)', 'y+lDepth (km)'])
            fig.grdimage(grid=pro_surf_arr, cmap=True)
            
            if plot_contour:
                surf_x = pro_surf_arr.coords['x']
                surf_y = pro_surf_arr.coords['y']
                surf_x_mesh, surf_y_mesh = np.meshgrid(surf_x, surf_y)
                surf_x_flat, surf_y_flat = surf_x_mesh.flatten(), surf_y_mesh.flatten()
                surf_z_flat = pro_surf_arr.values.flatten()
                fig.contour(x = surf_x_flat, y = surf_y_flat, z = surf_z_flat, pen='0.25p,black', levels=line_interval, annotation=annotation_interval,
                            region=profile_range)
                
            # plotting eq
            if plot_eq_flag:
                eq_df = grab_earthqaukes(eq_file)
                fig = plot_eq(fig, eq_df, center, azi_profile, len_profile, width_for_projecting_eq)


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
            
            fig.colorbar(frame=['a1f1', f'x+l{cbar_label_list[scalar_index]}'], position='JMR+o-35c/-4c+w10c/0.5c+ml', cmap=True)

            
            offset = np.abs(len_profile[1] - len_profile[0]) * 0.025
            fig.text(x = [profile_range[0] + offset, profile_range[1] - offset], y = [-1, -1],
                    text = [letter_index, f"{letter_index}'"], font='45p,black', justify='MC', fill='#f6fc67', no_clip=True)
            
            output_dir_for_fig = os.path.join(output_dir, 'fig', 'vertical', 'abs')
            check_if_output_dir_exist(output_dir=output_dir_for_fig)
            fig.savefig(f'{output_dir_for_fig}/profile_{scalar}_{int(azi_profile)}_{profile_index}.png', dpi=300,
                        transparent=True)




def plot_vertical_slices_updated(input_dir, input_dir_ref, output_dir, model_n, model_ref_n):
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, 'plot_config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    # ----------------------------------------------------------------------
    # Load the configuration
    # ----------------------------------------------------------------------
    
    config_vert = config['vertical_slice']
    map_region = config_vert['general_map']['map_region']
    dep_range  = config_vert['general_map']['dep_range']
    dep_interval_for_interp = config_vert['general_map']['dep_interval_for_interp']
    interval_for_profile = config_vert['general_map']['interval_for_profile']
    width_profile = config_vert['general_map']['width_profile']
    width_for_projecting_eq = config_vert['general_map']['width_for_projecting_eq']
    topo_range_for_plot = config_vert['general_map']['topo_range_for_plot']
    surface_spacing = config_vert['general_map']['surface_spacing']

    mapview_flag = config_vert['general_flag']['plot_mapview']
    plot_eq_flag = config_vert['general_flag']['plot_eq']
    
    scalar_list = config_vert['fine_tune_updated']['scalar_list']
    scalar_range_list = config_vert['fine_tune_updated']['scalar_range_list']
    cmap = config_vert['fine_tune_updated']['cmap']
    reverse_cmap = config_vert['fine_tune_updated']['reverse_cmap']
    cbar_label_list = config_vert['fine_tune_updated']['cbar_label_list']
    plot_contour = config_vert['fine_tune_updated']['contour']['plot_contour']
    line_interval = config_vert['fine_tune_updated']['contour']['line_interval']
    annotation_interval = config_vert['fine_tune_updated']['contour']['annotation_interval']
    
    topo_grd = config_vert['general_file']['topo_grd']
    eq_file = config_vert['general_file']['eq_file']
    profile_input_file = config_vert['general_file']['profile_input']
    # ----------------------------------------------------------------------
    # Main 
    # ----------------------------------------------------------------------
    profile_info_df = read_profile_input_file(file=profile_input_file)
    input_file = f'{input_dir}/model.xyz'
    input_file_ref = f'{input_dir_ref}/model.xyz'
    
    print(f'plot the model difference read from {input_file} and {input_file_ref}')
    xyz_df = pd.read_csv(input_file, sep='\s+', skiprows=5, 
                            names=['lon', 'lat', 'dep', 'vp', 'vs', 'rho', 'dvp', 'dvs', 'drho'])
    xyz_ref = pd.read_csv(input_file_ref, sep='\s+', skiprows=5,
                            names=['lon', 'lat', 'dep', 'vp', 'vs', 'rho', 'dvp', 'dvs', 'drho'])
    
    # Calculate the difference
    log_diff = np.log(xyz_df[['vp', 'vs', 'rho']].values / xyz_ref[['vp', 'vs', 'rho']].values) * 1E+02
    result_df = xyz_df[['lon', 'lat', 'dep']].copy()
    result_df[['vp', 'vs', 'rho']] = log_diff
    
    
    
    for scalar_index, scalar in enumerate(scalar_list):
        scalar_range = scalar_range_list[scalar_index]
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
            # GMT pre-processing
            # ---------------------------------------------------------------------------
            profile_range = [len_profile[0], len_profile[1], dep_range[0], dep_range[1]]
            selected_df = result_df[['lon','lat','dep',scalar]]

            pro_arr = pygmt.project(
                data = selected_df,
                center = center,
                azimuth = azi_profile,
                length = [len_profile[0]*1.1, len_profile[1]*1.1],
                unit = True,
                width = width_profile,
            )

            pro_line = pygmt.project(
                data = selected_df,
                center = center,
                azimuth = azi_profile,
                length = len_profile,
                unit = True,
                generate = interval_for_profile,
            )
            
            dep_arr = np.linspace(dep_range[0], dep_range[1], num=int((dep_range[1] - dep_range[0]) / dep_interval_for_interp) + 1)
            
            target_grid = create_target_grid(pro_line.r.values, pro_line.s.values, dep_arr)
            profile_points = np.array([pro_arr[0].values, pro_arr[1].values, pro_arr[2].values]).T
            profile_values = pro_arr[3].values
            pro_arr_interp = interp_from_array_to_profile(profile_points, profile_values, target_grid)

            x_grid, z_grid = np.meshgrid(pro_line.p.values, dep_arr, indexing='ij')
            x_grid_flat = x_grid.flatten()
            z_grid_flat = z_grid.flatten()
            
            divided_points_index_list = calculate_profile_division_points(len(pro_line), 4)
            
            pro_surf_arr = pygmt.surface(
                x = x_grid_flat,
                y = z_grid_flat,
                z = pro_arr_interp,
                region = [len_profile[0], len_profile[1], dep_range[0], dep_range[1]],
                spacing = surface_spacing,
            )
            

            # -------------------------------------------
            # ------------ GMT config settings ----------
            # -------------------------------------------
            fig = pygmt.Figure()
            pygmt.config(FORMAT_GEO_MAP="ddd.x", MAP_FRAME_TYPE="plain")
            pygmt.config(FONT='30p')
            pygmt.config(FONT_LABEL='36p')
            pygmt.config(MAP_FRAME_PEN='4p')

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

            pygmt.makecpt(cmap=cmap, series=scalar_range, reverse=reverse_cmap)
            fig.basemap(region=profile_range, projection='X12i/-8i', frame=['WSne', 'a40f20', 'x+lDistance (km)', 'y+lDepth (km)'])
            fig.grdimage(grid=pro_surf_arr, cmap=True)
            
            if plot_contour:
                surf_x = pro_surf_arr.coords['x']
                surf_y = pro_surf_arr.coords['y']
                surf_x_mesh, surf_y_mesh = np.meshgrid(surf_x, surf_y)
                surf_x_flat, surf_y_flat = surf_x_mesh.flatten(), surf_y_mesh.flatten()
                surf_z_flat = pro_surf_arr.values.flatten()
                fig.contour(x = surf_x_flat, y = surf_y_flat, z = surf_z_flat, pen='0.25p,black', levels=line_interval, annotation=annotation_interval,
                            region=profile_range)
                
            # plotting eq
            if plot_eq_flag:
                eq_df = grab_earthqaukes(eq_file)
                fig = plot_eq(fig, eq_df, center, azi_profile, len_profile, width_for_projecting_eq)


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
            
            fig.colorbar(frame=['a10f5', f'x+l{cbar_label_list[scalar_index]}'], position='JMR+o-35c/-4c+w10c/0.5c+ml', cmap=True)

            
            offset = np.abs(len_profile[1] - len_profile[0]) * 0.025
            fig.text(x = [profile_range[0] + offset, profile_range[1] - offset], y = [-1, -1],
                    text = [letter_index, f"{letter_index}'"], font='45p,black', justify='MC', fill='#f6fc67', no_clip=True)
            
            output_dir_for_fig = os.path.join(output_dir, 'fig', 'vertical', 'update')
            check_if_output_dir_exist(output_dir=output_dir_for_fig)
            fig.savefig(f'{output_dir_for_fig}/profile_{scalar}_m{model_n:03d}_m{model_ref_n:03d}_{int(azi_profile)}_{profile_index}.png', dpi=300,
                        transparent=True)



if __name__ == '__main__':
    plot_vertical_slices_pert(input_dir='/home/harry/Work/AdjointFlows/TOMO/m000/OUTPUT',
                              output_dir='/home/harry/Work/AdjointFlows/TOMO/m000/OUTPUT')
    # plot_vertical_slices_pert(input_dir='/home/harry/Work/AdjointFlows/TOMO_lastest_20250414/m005/OUTPUT',
    #                           output_dir='/home/harry/Work/AdjointFlows/TOMO_lastest_20250414/m005/OUTPUT')
    # plot_vertical_slices_abs(input_dir='/home/harry/Work/AdjointFlows/TOMO/m006/OUTPUT',
                            #   output_dir='/home/harry/Work/AdjointFlows/TOMO/m006/OUTPUT')
    # plot_vertical_slices_updated(input_dir='/home/harry/Work/AdjointFlows/TOMO/m006/OUTPUT',
    #                              input_dir_ref = '/home/harry/Work/AdjointFlows/TOMO/m000/OUTPUT',
    #                              output_dir='/home/harry/Work/AdjointFlows/TOMO/m006/OUTPUT',
    #                              model_n=6,
    #                              model_ref_n=0)