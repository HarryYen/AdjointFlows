#%%
from pathlib import Path
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import numpy as np
import os
import vtk
import sys


# ---------------------------------------------------------------------------
# AK135 1D reference model (Kennett et al. 1995)
# depth_km | vp (km/s) | vs (km/s) | rho (g/cm³)
# Duplicate depth entries represent velocity discontinuities.
# ---------------------------------------------------------------------------
_AK135 = np.array([
    [  0.00, 1.4500, 0.0000, 1.0200],  # ocean water
    [  3.00, 1.4500, 0.0000, 1.0200],
    [  3.00, 1.6500, 1.0000, 2.0000],  # sediment
    [  3.30, 1.6500, 1.0000, 2.0000],
    [  3.30, 5.8000, 3.2000, 2.6000],  # upper crust
    [ 10.00, 5.8000, 3.2000, 2.6000],
    [ 10.00, 6.8000, 3.9000, 2.9200],  # lower crust
    [ 18.00, 6.8000, 3.9000, 2.9200],
    [ 18.00, 8.0355, 4.4839, 3.6410],  # upper mantle (Moho)
    [ 43.00, 8.0379, 4.4856, 3.5801],
    [ 80.00, 8.0400, 4.4800, 3.5020],
    [ 80.00, 8.0450, 4.4900, 3.5020],  # LVZ top
    [120.00, 8.0505, 4.5000, 3.4268],
    [165.00, 8.1750, 4.5090, 3.3711],
    [210.00, 8.3007, 4.5184, 3.3243],
    [210.00, 8.3007, 4.5184, 3.3243],
    [260.00, 8.4822, 4.6094, 3.3663],
    [310.00, 8.6650, 4.6964, 3.4110],
    [360.00, 8.8476, 4.7832, 3.4577],
    [410.00, 9.0302, 4.8702, 3.5068],  # 410-km discontinuity
])
_AK135_DEPTH_KM = _AK135[:, 0]
_AK135_VP  = _AK135[:, 1] * 1000.0   # km/s  → m/s
_AK135_VS  = _AK135[:, 2] * 1000.0   # km/s  → m/s
_AK135_RHO = _AK135[:, 3] * 1000.0   # g/cm³ → kg/m³

_AK135_ARRAYS = {'vp': _AK135_VP, 'vs': _AK135_VS, 'rho': _AK135_RHO}

def _ak135_value(depth_m, kernel_name):
    """
    Interpolate AK135 value at a given depth.
    depth_m: depth in metres, negative = underground (same convention as query_dep_arr)
    """
    depth_km = max(-depth_m / 1000.0, 0.0)  # above-surface clamped to 0
    return float(np.interp(depth_km, _AK135_DEPTH_KM, _AK135_ARRAYS[kernel_name]))


def project_gll_to_regular(databases_dir, kernel_name, query_lon_arr, query_lat_arr, query_dep_arr):
    """
    Use vtkProbeFilter for sampling.
    - Fill invalid points within a layer with the layer nanmean.
    - If the whole layer has no valid values, fall back to the AK135 1D reference model.
    - Return shape: (nlon*nlat*ndep,) 1-D array
    Args:
        databases_dir (str): the directory of the databases
        kernel_name (str): the name of the kernel — 'vp', 'vs', or 'rho'
        query_lon_arr (np.array): easting array of the query points (metres)
        query_lat_arr (np.array): northing array of the query points (metres)
        query_dep_arr (np.array): depth array (metres, negative = underground)
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
    
    pts = vtk.vtkPoints()
    for specified_dep in query_dep_arr:
        z0 = float(specified_dep)

        # 3a) Preparing (x, y, z0) points cloud
        coords = np.column_stack([query_x, query_y, np.full(nxy, z0)])
        pts.SetData(numpy_to_vtk(coords, deep=True, array_type=vtk.VTK_DOUBLE))

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
            # edge case: whole layer invalid -> fall back to AK135 at this depth
            vals = np.full(nxy, _ak135_value(z0, kernel_name))
        else:
            layer_mean = np.nanmean(vals)
            vals = np.where(np.isnan(vals), layer_mean, vals)

        abs_list.append(vals)

    # 4) Combine all depth layers
    abs_arr = np.hstack(abs_list).astype(float)
    return abs_arr

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
        header_info += f' {v1_abs_min:.3f} {v1_abs_max:.3f} {v2_abs_min:.3f} {v2_abs_max:.3f} {v3_abs_min:.3f} {v3_abs_max:.3f}'

        np.savetxt(output_path, output_data, fmt='%.3f', header=header_info, comments='')


if __name__ == '__main__':

    # -------------------------
    # Parameters Setup
    # -------------------------
    model_num = 26
    lon_range = [700010.642, 1331760.679]
    lat_range = [2356665.767, 2911482.965]
    dep_range = [-200000.0, 5000.0]
    lon_interval, lat_interval, dep_interval = 2500.0, 2500.0, 2500.0
    # -------------------------

    kernel_list = ['vp', 'vs', 'rho']
    model_file_name = f'm{model_num:03d}'
    root_path = Path(__file__).parent.parent
    databases_dir = root_path / 'TOMO' / model_file_name / 'DATABASES_MPI'
    output_path = root_path / 'TOMO' / model_file_name / 'OUTPUT' / f'tomography_model_{model_file_name}.xyz'

    if not databases_dir.exists():
        print(f'{databases_dir} does not exist')
        sys.exit()

    output_path.parent.mkdir(parents=True, exist_ok=True)


    query_lon_arr = np.arange(lon_range[0], lon_range[1]+lon_interval, lon_interval)
    query_lat_arr = np.arange(lat_range[0], lat_range[1]+lat_interval, lat_interval)
    query_dep_arr = np.arange(dep_range[0], dep_range[1]+dep_interval, dep_interval)


    val_list = []
    for index, kernel_name in enumerate(kernel_list):
        interp_data_points = project_gll_to_regular(databases_dir=databases_dir,
                                                    kernel_name=kernel_name,
                                                    query_lon_arr=query_lon_arr,
                                                    query_lat_arr=query_lat_arr,
                                                    query_dep_arr=query_dep_arr)
        val_list.append(interp_data_points)

    output_model_txt_file(query_lon_arr=query_lon_arr, query_lat_arr=query_lat_arr, query_dep_arr=query_dep_arr, 
                          lon_interval=lon_interval, lat_interval=lat_interval, dep_interval=dep_interval, 
                          output_path=output_path, v1=val_list[0] , v2=val_list[1], v3=val_list[2])

# %%
