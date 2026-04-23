from plotting_modules import (
    read_profile_input_file,
    create_target_grid,
    interp_from_array_to_profile,
    calculate_profile_division_points,
    grab_earthqaukes,
)

import os

import numpy as np
import pandas as pd
import pygmt
import yaml


MODEL_COLUMNS = ['lon', 'lat', 'dep', 'vp', 'vs', 'rho', 'dvp', 'dvs', 'drho']
COLORBAR_POSITION = 'JMR+o1.5c/-12.c+w12c/1.c+ml'


def load_vertical_slice_config():
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, 'plot_config.yaml'), 'r') as file:
        config = yaml.safe_load(file)
    return config['vertical_slice']


def read_model_xyz(input_dir):
    input_file = os.path.join(input_dir, 'model.xyz')
    print(f'plot the model read from {input_file}')
    return pd.read_csv(
        input_file,
        sep=r'\s+',
        skiprows=5,
        names=MODEL_COLUMNS,
    )


def check_if_output_dir_exist(output_dir):
    os.makedirs(output_dir, exist_ok=True)


def parse_profile_info(profile_info):
    val = str(profile_info['profile_name'])
    try:
        letter_index = int(float(val))
    except ValueError:
        letter_index = val

    azi_profile = profile_info['angle']
    len_profile = [profile_info['lmin'], profile_info['lmax']]
    center = [profile_info['clon'], profile_info['clat']]
    return letter_index, azi_profile, len_profile, center


def create_vertical_slice_figure():
    fig = pygmt.Figure()
    pygmt.config(FORMAT_GEO_MAP='ddd.x', MAP_FRAME_TYPE='plain')
    pygmt.config(FONT='30p')
    pygmt.config(FONT_LABEL='36p')
    pygmt.config(MAP_FRAME_PEN='4p')
    return fig


def build_vertical_profile_surface(
    data_df,
    scalar,
    center,
    azi_profile,
    len_profile,
    dep_range,
    width_profile,
    interval_for_profile,
    dep_interval_for_interp,
    surface_spacing,
):
    profile_range = [len_profile[0], len_profile[1], dep_range[0], dep_range[1]]
    selected_df = data_df[['lon', 'lat', 'dep', scalar]]

    pro_arr = pygmt.project(
        data=selected_df,
        center=center,
        azimuth=azi_profile,
        length=[len_profile[0] * 1.1, len_profile[1] * 1.1],
        unit=True,
        width=width_profile,
    )

    pro_line = pygmt.project(
        data=selected_df,
        center=center,
        azimuth=azi_profile,
        length=len_profile,
        unit=True,
        generate=interval_for_profile,
    )

    dep_arr = np.linspace(
        dep_range[0],
        dep_range[1],
        num=int((dep_range[1] - dep_range[0]) / dep_interval_for_interp) + 1,
    )

    target_grid = create_target_grid(pro_line.r.values, pro_line.s.values, dep_arr)
    profile_points = np.array([pro_arr[0].values, pro_arr[1].values, pro_arr[2].values]).T
    profile_values = pro_arr[3].values
    pro_arr_interp = interp_from_array_to_profile(profile_points, profile_values, target_grid)

    x_grid, z_grid = np.meshgrid(pro_line.p.values, dep_arr, indexing='ij')
    divided_points_index_list = calculate_profile_division_points(len(pro_line), 4)

    pro_surf_arr = pygmt.surface(
        x=x_grid.flatten(),
        y=z_grid.flatten(),
        z=pro_arr_interp,
        region=profile_range,
        spacing=surface_spacing,
    )
    return profile_range, pro_line, pro_surf_arr, divided_points_index_list


def plot_eq(fig, eq_df, center, azi_profile, len_profile, width_profile):
    projected_eq = pygmt.project(
        data=eq_df,
        center=center,
        azimuth=azi_profile,
        length=[len_profile[0] * 1.1, len_profile[1] * 1.1],
        unit=True,
        width=width_profile,
    )
    try:
        fig.plot(
            x=projected_eq[3].values,
            y=projected_eq[2].values,
            style='c0.05c',
            pen='0.005p,black',
        )
    except KeyError:
        pass

    return fig


def plot_mapview_panel(
    fig,
    pro_line,
    divided_points_index_list,
    letter_index,
    map_region,
    topo_grd,
    topo_range_for_plot,
):
    fig.basemap(region=map_region, projection='M4.5i', frame=['neWS', 'a2f1'])

    gradient_data = pygmt.grdgradient(
        grid=topo_grd,
        azimuth=[45, 135],
        normalize='e0.7',
    )

    pygmt.makecpt(
        cmap='gray',
        series=topo_range_for_plot,
    )

    fig.grdimage(
        region=map_region,
        grid=topo_grd,
        shading=gradient_data,
        cmap=True,
        transparency=60,
    )
    fig.coast(shorelines='1.5p', resolution='h')
    fig.plot(
        x=[pro_line.r.iloc[0], pro_line.r.iloc[-1]],
        y=[pro_line.s.iloc[0], pro_line.s.iloc[-1]],
        pen='4p,black',
        style='f-2/0.75c',
    )
    fig.plot(
        x=pro_line.r.values[divided_points_index_list],
        y=pro_line.s.values[divided_points_index_list],
        pen='1p,black',
        style='c0.3c',
        fill='#8c8cea',
    )

    fig.text(
        x=[pro_line.r.iloc[0], pro_line.r.iloc[-1]],
        y=[pro_line.s.iloc[0], pro_line.s.iloc[-1]],
        text=[letter_index, f"{letter_index}'"],
        font='32p,black',
        justify='MC',
        fill='#fff683',
    )

    fig.shift_origin(xshift='6.7i')


def plot_profile_panel(
    fig,
    pro_surf_arr,
    profile_range,
    cmap,
    scalar_range,
    reverse_cmap,
    contour_config=None,
    cpt_background=False,
):
    if cpt_background:
        pygmt.makecpt(cmap=cmap, series=scalar_range, reverse=reverse_cmap, background=True)
    else:
        pygmt.makecpt(cmap=cmap, series=scalar_range, reverse=reverse_cmap)

    fig.basemap(
        region=profile_range,
        projection='x0.06i/-0.06i',
        frame=['WSne', 'a40f20', 'x+lDistance (km)', 'y+lDepth (km)'],
    )
    fig.grdimage(grid=pro_surf_arr, cmap=True)

    if contour_config is not None and contour_config['plot_contour']:
        plot_profile_contour(
            fig,
            pro_surf_arr,
            profile_range,
            contour_config['line_interval'],
            contour_config['annotation_interval'],
        )


def plot_profile_contour(fig, pro_surf_arr, profile_range, line_interval, annotation_interval):
    surf_x = pro_surf_arr.coords['x']
    surf_y = pro_surf_arr.coords['y']
    surf_x_mesh, surf_y_mesh = np.meshgrid(surf_x, surf_y)
    fig.contour(
        x=surf_x_mesh.flatten(),
        y=surf_y_mesh.flatten(),
        z=pro_surf_arr.values.flatten(),
        pen='0.25p,black',
        levels=line_interval,
        annotation=annotation_interval,
        region=profile_range,
    )


def plot_topography_panel(fig, topo_grd, pro_line, profile_range, divided_points_index_list):
    fig.shift_origin(yshift='7i')
    topo_track = pygmt.grdtrack(
        grid=topo_grd,
        points=pro_line[['r', 's', 'p']],
        newcolname='topo',
    )

    fig.basemap(
        region=[profile_range[0], profile_range[1], -1.2, 1.2],
        projection='x0.06i/0.4i',
        frame=['WE'],
    )

    dist_arr = np.array(topo_track.p)
    topo_arr = np.array(topo_track.topo)
    topo_arr = topo_arr / np.max(np.abs(topo_arr))

    dist_arr = np.concatenate(([dist_arr[0]], dist_arr, [dist_arr[-1]]))
    topo_arr = np.concatenate(([0], topo_arr, [0]))

    fig.plot(
        x=dist_arr[topo_arr <= 0],
        y=topo_arr[topo_arr <= 0],
        pen='1.5p,black',
        fill='#bcdaff',
        close='+y0',
    )
    fig.plot(
        x=dist_arr[topo_arr >= 0],
        y=topo_arr[topo_arr >= 0],
        pen='1.5p,black',
        fill='#fce2bb',
        close='+y0',
    )
    fig.plot(
        x=dist_arr[divided_points_index_list],
        y=np.zeros(len(divided_points_index_list)),
        pen='1p,black',
        style='c0.5c',
        fill='#8c8cea',
    )


def add_profile_endpoint_labels(fig, profile_range, len_profile, letter_index):
    offset = np.abs(len_profile[1] - len_profile[0]) * 0.025
    fig.text(
        x=[profile_range[0] + offset, profile_range[1] - offset],
        y=[-1, -1],
        text=[letter_index, f"{letter_index}'"],
        font='45p,black',
        justify='MC',
        fill='#f6fc67',
        no_clip=True,
    )


def save_vertical_slice(fig, output_dir, output_subdir, filename):
    output_dir_for_fig = os.path.join(output_dir, 'fig', 'vertical', output_subdir)
    check_if_output_dir_exist(output_dir=output_dir_for_fig)
    fig.savefig(
        os.path.join(output_dir_for_fig, filename),
        dpi=300,
        transparent=True,
    )


def plot_single_vertical_slice(
    data_df,
    scalar,
    scalar_range,
    cbar_frame,
    output_dir,
    output_subdir,
    filename,
    profile_info,
    profile_index,
    config_vert,
    contour_config=None,
    eq_df=None,
    cpt_background=False,
):
    general_map = config_vert['general_map']
    general_flag = config_vert['general_flag']
    general_file = config_vert['general_file']

    letter_index, azi_profile, len_profile, center = parse_profile_info(profile_info)
    print(f'Profile {profile_index}: center: {center}, angle:{azi_profile}, length:{len_profile}')

    profile_range, pro_line, pro_surf_arr, divided_points_index_list = build_vertical_profile_surface(
        data_df=data_df,
        scalar=scalar,
        center=center,
        azi_profile=azi_profile,
        len_profile=len_profile,
        dep_range=general_map['dep_range'],
        width_profile=general_map['width_profile'],
        interval_for_profile=general_map['interval_for_profile'],
        dep_interval_for_interp=general_map['dep_interval_for_interp'],
        surface_spacing=general_map['surface_spacing'],
    )

    fig = create_vertical_slice_figure()

    if general_flag['plot_mapview']:
        plot_mapview_panel(
            fig=fig,
            pro_line=pro_line,
            divided_points_index_list=divided_points_index_list,
            letter_index=letter_index,
            map_region=general_map['map_region'],
            topo_grd=general_file['topo_grd'],
            topo_range_for_plot=general_map['topo_range_for_plot'],
        )

    plot_profile_panel(
        fig=fig,
        pro_surf_arr=pro_surf_arr,
        profile_range=profile_range,
        cmap=cbar_frame['cmap'],
        scalar_range=scalar_range,
        reverse_cmap=cbar_frame['reverse_cmap'],
        contour_config=contour_config,
        cpt_background=cpt_background,
    )

    if general_flag['plot_eq'] and eq_df is not None:
        plot_eq(
            fig=fig,
            eq_df=eq_df,
            center=center,
            azi_profile=azi_profile,
            len_profile=len_profile,
            width_profile=general_map['width_for_projecting_eq'],
        )

    plot_topography_panel(
        fig=fig,
        topo_grd=general_file['topo_grd'],
        pro_line=pro_line,
        profile_range=profile_range,
        divided_points_index_list=divided_points_index_list,
    )
    fig.colorbar(
        frame=cbar_frame['frame'],
        position=COLORBAR_POSITION,
        cmap=True,
    )
    add_profile_endpoint_labels(fig, profile_range, len_profile, letter_index)
    save_vertical_slice(fig, output_dir, output_subdir, filename)


def get_profile_and_eq_data(config_vert):
    general_file = config_vert['general_file']
    profile_info_df = read_profile_input_file(file=general_file['profile_input'])
    eq_df = None
    if config_vert['general_flag']['plot_eq']:
        eq_df = grab_earthqaukes(general_file['eq_file'])
    return profile_info_df, eq_df


def get_contour_config(fine_tune_config):
    contour_config = fine_tune_config.get('contour')
    if contour_config is None:
        return None
    return {
        'plot_contour': contour_config['plot_contour'],
        'line_interval': contour_config['line_interval'],
        'annotation_interval': contour_config['annotation_interval'],
    }


def plot_vertical_slices_pert(input_dir, output_dir):
    config_vert = load_vertical_slice_config()
    fine_tune = config_vert['fine_tune_perturb']
    profile_info_df, eq_df = get_profile_and_eq_data(config_vert)
    xyz_df = read_model_xyz(input_dir)

    for scalar in fine_tune['scalar_list']:
        for profile_index, profile_info in profile_info_df.iterrows():
            letter_index, azi_profile, _, _ = parse_profile_info(profile_info)
            plot_single_vertical_slice(
                data_df=xyz_df,
                scalar=scalar,
                scalar_range=fine_tune['scalar_range'],
                cbar_frame={
                    'cmap': fine_tune['cmap'],
                    'reverse_cmap': fine_tune['reverse_cmap'],
                    'frame': ['a10f5', f'x+l{scalar} (%)'],
                },
                output_dir=output_dir,
                output_subdir='pert',
                filename=f'profile_{scalar}_{int(azi_profile)}_{letter_index}.png',
                profile_info=profile_info,
                profile_index=profile_index + 1,
                config_vert=config_vert,
                eq_df=eq_df,
            )


def plot_vertical_slices_abs(input_dir, output_dir):
    config_vert = load_vertical_slice_config()
    fine_tune = config_vert['fine_tune_abs']
    contour_config = get_contour_config(fine_tune)
    profile_info_df, eq_df = get_profile_and_eq_data(config_vert)
    xyz_df = read_model_xyz(input_dir)

    for scalar_index, scalar in enumerate(fine_tune['scalar_list']):
        scalar_range = fine_tune['scalar_range_list'][scalar_index]
        cbar_label = fine_tune['cbar_label_list'][scalar_index]
        for profile_index, profile_info in profile_info_df.iterrows():
            letter_index, azi_profile, _, _ = parse_profile_info(profile_info)
            plot_single_vertical_slice(
                data_df=xyz_df,
                scalar=scalar,
                scalar_range=scalar_range,
                cbar_frame={
                    'cmap': fine_tune['cmap'],
                    'reverse_cmap': fine_tune['reverse_cmap'],
                    'frame': ['a1f1', f'x+l{cbar_label}'],
                },
                output_dir=output_dir,
                output_subdir='abs',
                filename=f'profile_{scalar}_{int(azi_profile)}_{letter_index}.png',
                profile_info=profile_info,
                profile_index=profile_index + 1,
                config_vert=config_vert,
                contour_config=contour_config,
                eq_df=eq_df,
            )


def plot_vertical_slices_updated(input_dir, input_dir_ref, output_dir, model_n, model_ref_n):
    config_vert = load_vertical_slice_config()
    fine_tune = config_vert['fine_tune_updated']
    contour_config = get_contour_config(fine_tune)
    profile_info_df, eq_df = get_profile_and_eq_data(config_vert)

    input_file = os.path.join(input_dir, 'model.xyz')
    input_file_ref = os.path.join(input_dir_ref, 'model.xyz')
    print(f'plot the model difference read from {input_file} and {input_file_ref}')
    xyz_df = pd.read_csv(input_file, sep=r'\s+', skiprows=5, names=MODEL_COLUMNS)
    xyz_ref = pd.read_csv(input_file_ref, sep=r'\s+', skiprows=5, names=MODEL_COLUMNS)

    log_diff = np.log(xyz_df[['vp', 'vs', 'rho']].values / xyz_ref[['vp', 'vs', 'rho']].values) * 1E+02
    result_df = xyz_df[['lon', 'lat', 'dep']].copy()
    result_df[['vp', 'vs', 'rho']] = log_diff

    for scalar_index, scalar in enumerate(fine_tune['scalar_list']):
        scalar_range = fine_tune['scalar_range_list'][scalar_index]
        cbar_label = fine_tune['cbar_label_list'][scalar_index]
        for profile_index, profile_info in profile_info_df.iterrows():
            letter_index, azi_profile, _, _ = parse_profile_info(profile_info)
            plot_single_vertical_slice(
                data_df=result_df,
                scalar=scalar,
                scalar_range=scalar_range,
                cbar_frame={
                    'cmap': fine_tune['cmap'],
                    'reverse_cmap': fine_tune['reverse_cmap'],
                    'frame': ['a10f5', f'x+l{cbar_label}'],
                },
                output_dir=output_dir,
                output_subdir='update',
                filename=(
                    f'profile_{scalar}_m{model_n:03d}_m{model_ref_n:03d}_'
                    f'{int(azi_profile)}_{letter_index}.png'
                ),
                profile_info=profile_info,
                profile_index=profile_index + 1,
                config_vert=config_vert,
                contour_config=contour_config,
                eq_df=eq_df,
            )


def plot_vertical_slices_vpvs(input_dir, output_dir):
    config_vert = load_vertical_slice_config()
    fine_tune = config_vert['fine_tune_vpvs']
    contour_config = get_contour_config(fine_tune)
    profile_info_df, eq_df = get_profile_and_eq_data(config_vert)
    xyz_df = read_model_xyz(input_dir)
    xyz_df['vpvs'] = xyz_df['vp'] / xyz_df['vs']

    for profile_index, profile_info in profile_info_df.iterrows():
        letter_index, azi_profile, _, _ = parse_profile_info(profile_info)
        plot_single_vertical_slice(
            data_df=xyz_df,
            scalar='vpvs',
            scalar_range=fine_tune['scalar_range'],
            cbar_frame={
                'cmap': fine_tune['cmap'],
                'reverse_cmap': fine_tune['reverse_cmap'],
                'frame': ['a0.1f0.1', 'x+lVp/Vs ratio'],
            },
            output_dir=output_dir,
            output_subdir='vpvs',
            filename=f'profile_VpVs_{int(azi_profile)}_{letter_index}.png',
            profile_info=profile_info,
            profile_index=profile_index + 1,
            config_vert=config_vert,
            contour_config=contour_config,
            eq_df=eq_df,
            cpt_background=True,
        )
