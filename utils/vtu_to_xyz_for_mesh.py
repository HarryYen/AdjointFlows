#%%
from pathlib import Path
from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import griddata
import numpy as np
import os
import vtk
import sys


def project_gll_to_regular(databases_dir, kernel_name, query_lon_arr, query_lat_arr, query_dep_arr, default_value):
    """
    Use vtkProbeFilter for samling
    - Fill the empty values with 'nanmean' 
    - If there is no value through the whole layer, we use *default_value*
    - Return shape: (nlon*nlat*ndep,) 1-D array
    Args:
        databases_dir (str): the directory of the databases
        kernel_name (str): the name of the kernel, e.g., 'vp', 'vs', 'rho'
        query_lon_arr (np.array): the longitude array of the query points
        query_lat_arr (np.array): the latitude array of the query points
        query_dep_arr (np.array): the depth array of the query points
        default_value (float): the default value to fill in case of NaN values
    """
    # 1) Read unstructuredGrid
    gll_file = os.path.join(databases_dir, f'{kernel_name}.vtu')
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(gll_file))
    reader.Update()
    ugrid = reader.GetOutput()

    # Check point data
    pdt = ugrid.GetPointData()
    data_array_vtk = pdt.GetArray(str(kernel_name))
    if data_array_vtk is None:
        raise ValueError(f"Point-data array '{kernel_name}' not found in {gll_file}")

    # 2) Create query points grid
    qlat, qlon = np.meshgrid(query_lat_arr, query_lon_arr, indexing='ij')
    query_x = qlon.ravel()  # lon/easting
    query_y = qlat.ravel()  # lat/northing
    nxy = query_x.size

    # 3) Sampling through each depth
    abs_list = []

    for specified_dep in query_dep_arr:
        z0 = float(specified_dep)

        # 3a) Preparing (x, y, z0) points cloud
        pts = vtk.vtkPoints()
        pts.SetNumberOfPoints(nxy)
        for i in range(nxy):
            pts.SetPoint(i, float(query_x[i]), float(query_y[i]), z0)

        pd = vtk.vtkPolyData()
        pd.SetPoints(pts)

        # 3b) Interplation
        probe = vtk.vtkProbeFilter()
        probe.SetSourceData(ugrid)
        probe.SetInputData(pd)
        probe.Update()
        sampled = probe.GetOutput()

        # 3c) get values & fill NaN
        val_vtk = sampled.GetPointData().GetArray(str(kernel_name))
        # If the whole values in this layer are invalid
        if val_vtk is None:
            vals = np.full(nxy, np.nan, dtype=float)
        # If some values are valid
        else:
            vals = vtk_to_numpy(val_vtk).astype(float)
            mask_vtk = sampled.GetPointData().GetArray("vtkValidPointMask")
            if mask_vtk is not None:
                mask = vtk_to_numpy(mask_vtk).astype(bool)
                # Use NaN to tag the invalid grid, and then use np.nanmean to fill
                vals[~mask] = np.nan

        # use mean value to fill NaN
        if np.all(np.isnan(vals)):
            # edge case: if the whole layer is invalid -> go back to default_value
            vals = np.full(nxy, float(default_value))
        else:
            layer_mean = np.nanmean(vals)
            
            if np.isnan(layer_mean):
                layer_mean = float(default_value)
            vals = np.where(np.isnan(vals), layer_mean, vals)

        abs_list.append(vals)

    # 4) Combine all depth layers
    abs_arr = np.hstack(abs_list).astype(float)
    return abs_arr

def get_points_by_projection(query_x, query_y, given_x, given_y, data_arr, default_value):

    given_points = np.vstack([given_x, given_y]).T
    try:
        grid_values = griddata(given_points, data_arr, (query_x, query_y), method='linear')
    except ValueError as e:
        print(f"Warning in griddata: {e}, we use default values to fill the value in this depth!")
        # Handle the case where griddata fails, e.g., using default values
        grid_values = np.full(query_x.shape, default_value)
    return grid_values

def output_model_txt_file(query_lon_arr, query_lat_arr, query_dep_arr, lon_interval, lat_interval, dep_interval, output_path, v1 , v2, v3):
        """
        Output the txt file for the visualization
        """

        query_dep, query_lat, query_lon = np.meshgrid(query_dep_arr, query_lat_arr, query_lon_arr, indexing='ij')
        query_dep = query_dep.flatten()
        query_lon = query_lon.flatten()
        query_lat = query_lat.flatten()
        nlon, nlat, ndep = query_lon_arr.size, query_lat_arr.size, query_dep_arr.size
        
        query_lon_min, query_lon_max = np.nanmin(query_lon), np.nanmax(query_lon)
        query_lat_min, query_lat_max = np.nanmin(query_lat), np.nanmax(query_lat)
        query_dep_min, query_dep_max = np.nanmin(query_dep), np.nanmax(query_dep)
        
        v1_abs_min, v1_abs_max = np.nanmin(v1), np.nanmax(v1)
        v2_abs_min, v2_abs_max = np.nanmin(v2), np.nanmax(v2)
        v3_abs_min, v3_abs_max = np.nanmin(v3), np.nanmax(v3)

        
        output_data = np.column_stack((query_lon, query_lat, query_dep, v1, v2, v3))
        header_info =  f'{query_lon_min:.3f} {query_lat_min:.3f} {query_dep_min:.3f} {query_lon_max:.3f} {query_lat_max:.3f} {query_dep_max:.3f}\n'
        header_info += f' {lon_interval:.3f} {lat_interval:.3f} {dep_interval:.3f}\n'
        header_info += f' {nlon:4d} {nlat:4d} {ndep:4d}\n'
        header_info += f' {v1_abs_min:.3f} {v1_abs_max:.3f} {v2_abs_min:.3f} {v2_abs_max:.3f} {v3_abs_min:.3f} {v3_abs_max:.3f}\n'

        np.savetxt(output_path, output_data, fmt='%.3f', header=header_info, comments='')


if __name__ == '__main__':

    # -------------------------
    # Parameters Setup
    # -------------------------
    model_num = 16
    lon_range = [700010.642, 1331760.679]
    lat_range = [2356665.767, 2911482.965]
    dep_range = [-200000.0, 5000.0]
    lon_interval, lat_interval, dep_interval = 5000.0, 5000.0, 5000.0
    default_values = [5000, 3000, 1736] # vp, vs, rho
    # -------------------------

    kernel_list = ['vp', 'vs', 'rho']
    model_file_name = f'm{model_num:03d}'
    current_path = Path.cwd()
    root_path = current_path.parent
    databases_dir = root_path / 'TOMO' / model_file_name / 'DATABASES_MPI'
    output_path = root_path / 'TOMO' / model_file_name / 'OUTPUT' / f'tomography_model_{model_file_name}.xyz'
    
    if not databases_dir.exists():
        print(f'{databases_dir} does not exist')
        sys.exit() 


    query_lon_arr = np.arange(lon_range[0], lon_range[1]+lon_interval, lon_interval)
    query_lat_arr = np.arange(lat_range[0], lat_range[1]+lat_interval, lat_interval)
    query_dep_arr = np.arange(dep_range[0], dep_range[1]+dep_interval, dep_interval)


    val_list = []
    for index, kernel_name in enumerate(kernel_list):
        interp_data_points = project_gll_to_regular(databases_dir=databases_dir, 
                                                    kernel_name=kernel_name, 
                                                    query_lon_arr=query_lon_arr,
                                                    query_lat_arr=query_lat_arr,
                                                    query_dep_arr=query_dep_arr,
                                                    default_value=default_values[index])
        val_list.append(interp_data_points)


    output_model_txt_file(query_lon_arr=query_lon_arr, query_lat_arr=query_lat_arr, query_dep_arr=query_dep_arr, 
                          lon_interval=lon_interval, lat_interval=lat_interval, dep_interval=dep_interval, 
                          output_path=output_path, v1=val_list[0] , v2=val_list[1], v3=val_list[2])

# %%
