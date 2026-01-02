from tools.matrix_utils import get_param_from_specfem_file
from tools.job_utils import check_dir_exists, remove_file
from vtk.util.numpy_support import vtk_to_numpy
from plotting_modules import utm_to_lonlat, get_values_by_kdtree, lonlat_to_utm, get_points_by_projection
from horizontal_slices import plot_horizontal_slices_pert, plot_horizontal_slices_abs, plot_horizontal_slices_gradient, plot_horizontal_slices_updated
from vertical_slices import plot_vertical_slices_pert, plot_vertical_slices_abs, plot_vertical_slices_updated, plot_vertical_slices_vpvs
from vertical_slices_grad_diff import plot_vertical_slices_grad_diff
import os
import json
import subprocess
import vtk
import sys
import numpy as np


class TomographyVisualizer:
    def __init__(self, config, global_params, specified_model_num):
        self.base_dir          = global_params['base_dir']
        self.adjflows_dir      = os.path.join(self.base_dir, 'adjointflows')
        self.config            = config
        self.model_num         = specified_model_num
        self.specfem_dir       = os.path.join(self.base_dir, 'specfem3d')
        self.tomo_dir          = os.path.join(self.base_dir, 'TOMO', f'm{self.model_num:03d}')
        self.output_dir        = os.path.join(self.tomo_dir, 'OUTPUT')
        
        self.params_file       = os.path.join(self.tomo_dir, 'params.json')
        self.specfem_par_file  = os.path.join(self.specfem_dir, 'DATA', 'Par_file')
        
        self.databases_dir     = os.path.join(self.tomo_dir, 'DATABASES_MPI') 
        self.kernels_dir       = os.path.join(self.tomo_dir, 'KERNEL', 'PRECOND')
        self.nproc             = get_param_from_specfem_file(file=self.specfem_par_file, param_name='NPROC', param_type=int)
        
        self.lon_range         = None
        self.lat_range         = None
        self.dep_range         = None
        self.lon_interval      = None
        self.lat_interval      = None
        self.dep_interval      = None
        self.query_lon_arr     = None
        self.query_lat_arr     = None
        self.query_dep_arr     = None
        
        check_dir_exists(self.output_dir)
        
        
        
    def load_params(self):
        """
        Load the parameters from the json file
        """
        if not os.path.exists(self.params_file):
            raise FileNotFoundError(f"Error: {self.params_file} not found!")

        with open(self.params_file, "r") as f:
            self.params = json.load(f)
    
    def setup_spatial_range(self, lon_range, lat_range, dep_range, lon_interval, lat_interval, dep_interval):
        """
        Set the longitude and latitude range
        Args:
            lon_range (list) : the longitude range [min, max]
            lat_range (list) : the latitude range [min, max]
            dep_range (list) : the depth range [min, max]
            lon_interval (float) : the interval of the longitude
            lat_interval (float) : the interval of the latitude
            dep_interval (float) : the interval of the depth
        """
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.dep_range = dep_range
        self.lon_interval = lon_interval
        self.lat_interval = lat_interval
        self.dep_interval = dep_interval
        
        self.query_lon_arr = np.arange(self.lon_range[0], self.lon_range[1]+self.lon_interval, self.lon_interval)
        self.query_lat_arr = np.arange(self.lat_range[0], self.lat_range[1]+self.lon_interval, self.lat_interval)
        self.query_dep_arr = np.arange(self.dep_range[0], self.dep_range[1]+self.dep_interval, self.dep_interval)
        
    def from_bin_to_vtu_model(self, model_name):
        """
        Convert the binary kernel file to vtu file using 
        the script from SPECFEM3D (bin/xcombine_vol_data_vtu)
        """
        os.chdir(self.specfem_dir)
        target_file_name = f'{model_name}.vtu'
        remove_file(target_file_name)
        
        subprocess.run(
            ['./bin/xcombine_vol_data_vtu', '0', f'{self.nproc-1}', f'{model_name}', f'{self.databases_dir}', f'{self.databases_dir}', '0'],
            check=True
        )
        os.chdir(self.adjflows_dir)

    def from_bin_to_vtu_gradient(self, kernel_name):
        """
        Convert the binary kernel file to vtu file using 
        the script from SPECFEM3D (bin/xcombine_vol_data_vtu)
        """
        os.chdir(self.specfem_dir)
        target_file_name = f'{kernel_name}.vtu'
        remove_file(target_file_name)
        
        subprocess.run(
            ['./bin/xcombine_vol_data_vtu', '0', f'{self.nproc-1}', f'{kernel_name}', f'{self.kernels_dir}', f'{self.databases_dir}', '0'],
            check=True
        )
        os.chdir(self.adjflows_dir)
        
    def project_gll_to_regular(self, kernel_name, utm_zone, is_north_hemisphere):
        """
        Input the GLL table (.vtu) and output values on a regular (lon, lat, dep) grid
        using element-aware probing instead of 'thick-slab' point selection.

        Args:
            kernel_name (str): name of the kernel; read {kernel_name}.vtu under DATABASES_MPI
            utm_zone (int): UTM zone number (e.g., 50)
            is_north_hemisphere (bool): True if northern hemisphere

        Returns:
            abs_arr  (np.array, 1-D): concatenated absolute values at each requested depth
            pert_arr (np.array, 1-D): concatenated percent perturbations relative to
                                    the nan-mean at each depth
        """
        # --- 1) Read UnstructuredGrid ---
        gll_file = os.path.join(self.databases_dir, f'{kernel_name}.vtu')
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(gll_file))
        reader.Update()
        ugrid = reader.GetOutput()  # vtkUnstructuredGrid

        # Check
        pdt = ugrid.GetPointData()
        data_array_vtk = pdt.GetArray(str(kernel_name))
        if data_array_vtk is None:
            raise ValueError(f"Point-data array '{kernel_name}' not found in {gll_file}")

        # --- 2) create query lon/lat grids ---
        query_lon_arr = self.query_lon_arr
        query_lat_arr = self.query_lat_arr 
        qlon, qlat = np.meshgrid(query_lon_arr, query_lat_arr, indexing='ij')
        qlon = qlon.ravel()
        qlat = qlat.ravel()

        query_x, query_y = lonlat_to_utm(
            lon=qlon, lat=qlat, utm_zone=utm_zone, is_north_hemisphere=is_north_hemisphere
        )

        # --- 3) post-processing with depth ---
        abs_list, pert_list = [], []

        for specified_dep_km in self.query_dep_arr:
            
            z0 = -float(specified_dep_km) * 1e3

            # 3a) (x, y, z0) -> vtkPolyData
            pts = vtk.vtkPoints()
            pts.SetNumberOfPoints(len(query_x))
            for i in range(len(query_x)):
                pts.SetPoint(i, float(query_x[i]), float(query_y[i]), z0)

            pd = vtk.vtkPolyData()
            pd.SetPoints(pts)

            # 3b) Use ProbeFilter to sample
            probe = vtk.vtkProbeFilter()
            probe.SetSourceData(ugrid) 
            probe.SetInputData(pd)      
            probe.Update()
            sampled = probe.GetOutput()

            # 3c) get array
            val_vtk = sampled.GetPointData().GetArray(str(kernel_name))
            if val_vtk is None:
                vals = np.full(len(query_x), np.nan, dtype=float)
            else:
                vals = vtk_to_numpy(val_vtk).astype(float)

                mask_vtk = sampled.GetPointData().GetArray("vtkValidPointMask")
                if mask_vtk is not None:
                    mask = vtk_to_numpy(mask_vtk).astype(bool)
                    vals[~mask] = np.nan

            # 3d) calculate perturbation
            mean_in_this_dep = np.nanmean(vals)
            if not np.isfinite(mean_in_this_dep) or mean_in_this_dep == 0.0:
                pert = np.full_like(vals, np.nan)
            else:
                pert = (vals - mean_in_this_dep) / mean_in_this_dep * 1e2

            abs_list.append(vals)
            pert_list.append(pert)

        # --- 4) concatenate all depths ---
        abs_arr = np.hstack(abs_list)
        pert_arr = np.hstack(pert_list)
        return abs_arr, pert_arr
    
    
    def output_model_txt_file(self, output_file_name, v1_abs, v1_pert, v2_abs, v2_pert, v3_abs, v3_pert):
        """
        Output the txt file for the visualization
        """
        query_lon_arr = self.query_lon_arr
        query_lat_arr = self.query_lat_arr
        query_dep_arr = self.query_dep_arr
        query_dep, query_lon, query_lat = np.meshgrid(query_dep_arr, query_lon_arr, query_lat_arr, indexing='ij')
        query_dep = query_dep.flatten()
        query_lon = query_lon.flatten()
        query_lat = query_lat.flatten()
        nlon, nlat, ndep = query_lon_arr.size, query_lat_arr.size, query_dep_arr.size
        
        query_lon_min, query_lon_max = np.nanmin(query_lon), np.nanmax(query_lon)
        query_lat_min, query_lat_max = np.nanmin(query_lat), np.nanmax(query_lat)
        query_dep_min, query_dep_max = np.nanmin(query_dep), np.nanmax(query_dep)
        
        v1_abs_min, v1_abs_max = np.nanmin(v1_abs), np.nanmax(v1_abs)
        v2_abs_min, v2_abs_max = np.nanmin(v2_abs), np.nanmax(v2_abs)
        v3_abs_min, v3_abs_max = np.nanmin(v3_abs), np.nanmax(v3_abs)
        v1_pert_min, v1_pert_max = np.nanmin(v1_pert), np.nanmax(v1_pert)
        v2_pert_min, v2_pert_max = np.nanmin(v2_pert), np.nanmax(v2_pert)
        v3_pert_min, v3_pert_max = np.nanmin(v3_pert), np.nanmax(v3_pert)
        
        output_data = np.column_stack((query_lon, query_lat, query_dep, v1_abs, v2_abs, v3_abs, v1_pert, v2_pert, v3_pert))
        header_info =  f'{query_lon_min:.3f} {query_lat_min:.3f} {query_dep_min:.3f} {query_lon_max:.3f} {query_lat_max:.3f} {query_dep_max:.3f}\n'
        header_info += f' {self.lon_interval:.3f} {self.lat_interval:.3f} {self.dep_interval:.3f}\n'
        header_info += f' {nlon:4d} {nlat:4d} {ndep:4d}\n'
        header_info += f' {v1_abs_min:.3f} {v1_abs_max:.3f} {v2_abs_min:.3f} {v2_abs_max:.3f} {v3_abs_min:.3f} {v3_abs_max:.3f}\n'
        header_info += f' {v1_pert_min:.1f} {v1_pert_max:.1f} {v2_pert_min:.1f} {v2_pert_max:.1f} {v3_pert_min:.1f} {v3_pert_max:.1f}'

        output_file = os.path.join(self.output_dir, output_file_name)
        np.savetxt(output_file, output_data, fmt='%.3f', header=header_info, comments='')
        
        
    def output_kernel_txt_file(self, output_file_name, v1_abs, v1_pert, v2_abs, v2_pert, v3_abs, v3_pert):
        """
        Output the txt file for the visualization
        """
        query_lon_arr = self.query_lon_arr
        query_lat_arr = self.query_lat_arr
        query_dep_arr = self.query_dep_arr
        query_dep, query_lon, query_lat = np.meshgrid(query_dep_arr, query_lon_arr, query_lat_arr, indexing='ij')
        query_dep = query_dep.flatten()
        query_lon = query_lon.flatten()
        query_lat = query_lat.flatten()
        nlon, nlat, ndep = query_lon_arr.size, query_lat_arr.size, query_dep_arr.size
        
        query_lon_min, query_lon_max = np.nanmin(query_lon), np.nanmax(query_lon)
        query_lat_min, query_lat_max = np.nanmin(query_lat), np.nanmax(query_lat)
        query_dep_min, query_dep_max = np.nanmin(query_dep), np.nanmax(query_dep)
        
        v1_abs_min, v1_abs_max = np.nanmin(v1_abs), np.nanmax(v1_abs)
        v2_abs_min, v2_abs_max = np.nanmin(v2_abs), np.nanmax(v2_abs)
        v3_abs_min, v3_abs_max = np.nanmin(v3_abs), np.nanmax(v3_abs)
        v1_pert_min, v1_pert_max = np.nanmin(v1_pert), np.nanmax(v1_pert)
        v2_pert_min, v2_pert_max = np.nanmin(v2_pert), np.nanmax(v2_pert)
        v3_pert_min, v3_pert_max = np.nanmin(v3_pert), np.nanmax(v3_pert)
        
        output_data = np.column_stack((query_lon, query_lat, query_dep, v1_abs, v2_abs, v3_abs, v1_pert, v2_pert, v3_pert))
        header_info =  f'{query_lon_min:.3f} {query_lat_min:.3f} {query_dep_min:.3f} {query_lon_max:.3f} {query_lat_max:.3f} {query_dep_max:.3f}\n'
        header_info += f' {self.lon_interval:.3f} {self.lat_interval:.3f} {self.dep_interval:.3f}\n'
        header_info += f' {nlon:4d} {nlat:4d} {ndep:4d}\n'
        header_info += f' {v1_abs_min:.4e} {v1_abs_max:.4e} {v2_abs_min:.4e} {v2_abs_max:.4e} {v3_abs_min:.4e} {v3_abs_max:.4e}\n'
        header_info += f' {v1_pert_min:.1f} {v1_pert_max:.1f} {v2_pert_min:.1f} {v2_pert_max:.1f} {v3_pert_min:.1f} {v3_pert_max:.1f}'

        fmt = ['%.3f', '%.3f', '%.3f',  # lon, lat, dep
               '%.4e', '%.4e', '%.4e',  # v1_abs, v2_abs, v3_abs 
               '%.1f', '%.1f', '%.1f']  # v1_pert, v2_pert, v3_pert

        output_file = os.path.join(self.output_dir, output_file_name)
        np.savetxt(output_file, output_data, fmt=fmt, header=header_info, comments='')

        

    
    def plot_horizontal_slices_3x3_pert(self):
        """
        Plot the horizontal slice of the 3x3 kernels
        """
        fig_output_dir = os.path.join(self.output_dir, 'fig')
        plot_horizontal_slices_pert(map_region=[self.lon_range[0], self.lon_range[1], self.lat_range[0], self.lat_range[1]], 
                                    base_dir=self.base_dir, input_dir=self.output_dir, output_dir=fig_output_dir)

    def plot_horizontal_slices_3x3_abs(self):
        """
        Plot the horizontal slice of the 3x3 kernels
        """
        fig_output_dir = os.path.join(self.output_dir, 'fig')
        plot_horizontal_slices_abs(map_region=[self.lon_range[0], self.lon_range[1], self.lat_range[0], self.lat_range[1]], 
                                    base_dir=self.base_dir, input_dir=self.output_dir, output_dir=fig_output_dir)
        
    def plot_horizontal_slices_3x3_gradient(self):
        """
        Plot the horizontal slice of the 3x3 gradient
        """
        fig_output_dir = os.path.join(self.output_dir, 'fig')
        plot_horizontal_slices_gradient(map_region=[self.lon_range[0], self.lon_range[1], self.lat_range[0], self.lat_range[1]], 
                                    base_dir=self.base_dir, input_dir=self.output_dir, output_dir=fig_output_dir)
    
    
    def plot_horizontal_slices_3x3_updated(self, model_ref_num):
        """
        Plot the horizontal slice of the 3x3 updated amount
        """
        fig_output_dir = os.path.join(self.output_dir, 'fig')
        plot_horizontal_slices_updated(map_region=[self.lon_range[0], self.lon_range[1], self.lat_range[0], self.lat_range[1]], 
                                    base_dir=self.base_dir, model_ref_num=model_ref_num, input_dir=self.output_dir, output_dir=fig_output_dir)
    
    def plot_vertical_profile_pert(self):
        """
        Plot the vertical profile of the kernel (perturbation)
        """
        plot_vertical_slices_pert(input_dir=self.output_dir, output_dir=self.output_dir)
    
    def plot_vertical_profile_abs(self):
        """
        Plot the vertical profile of the kernel (absolute values)
        """
        plot_vertical_slices_abs(input_dir=self.output_dir, output_dir=self.output_dir)

    def plot_vertical_profile_vpvs(self):
        """
        Plot the vertical profile of the kernel (vpvs ratio values)
        """
        plot_vertical_slices_vpvs(input_dir=self.output_dir, output_dir=self.output_dir)
    
    def plot_vertical_profile_updated(self, model_ref_num):
        """
        Plot the vertical profile of the kernel (updated values)
        Args:
            model_ref_n (int) : the reference model number
        """
        input_dir_ref = os.path.join(self.base_dir, 'TOMO', f'm{model_ref_num:03d}', 'OUTPUT')
        plot_vertical_slices_updated(input_dir=self.output_dir, input_dir_ref=input_dir_ref, output_dir=self.output_dir,
                                     model_n=self.model_num, model_ref_n=model_ref_num)