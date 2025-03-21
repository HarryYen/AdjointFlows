from tools.matrix_utils import get_param_from_specfem_file
from tools.job_utils import check_dir_exists
from vtk.util.numpy_support import vtk_to_numpy
from plotting_modules import utm_to_lonlat, get_values_by_kdtree, lonlat_to_utm
from horizontal_slices import plot_horizontal_slices_pert, plot_horizontal_slices_abs
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
        subprocess.run(
            ['./bin/xcombine_vol_data_vtu', '0', f'{self.nproc-1}', f'{model_name}', f'{self.databases_dir}', f'{self.databases_dir}', '1'],
            check=True
        )
        os.chdir(self.adjflows_dir)

    def from_bin_to_vtu_gradient(self, kernel_name):
        """
        Convert the binary kernel file to vtu file using 
        the script from SPECFEM3D (bin/xcombine_vol_data_vtu)
        """
        os.chdir(self.specfem_dir)
        subprocess.run(
            ['./bin/xcombine_vol_data_vtu', '0', f'{self.nproc-1}', f'{kernel_name}', f'{self.kernels_dir}', f'{self.databases_dir}', '1'],
            check=True
        )
        os.chdir(self.adjflows_dir)
        
    
    
    def project_gll_to_regular(self, kernel_name, utm_zone, is_north_hemisphere, max_distance):
        """
        Input the GLL table and output the values array based on regular coordinates
        Include absolute value and perturbation
        Args:
            kernel_name (str) : the name of the kernel 
                                (find the file in the DATABASES_MPI called {kernel_name}.vtu)
            utm_zone (int)    : the UTM zone (e.g. 50)
            is_north_hemisphere (bool) : whether the region is in the northern
        Return:
            abs_arr (np.array)  : the absolute value of the interpolated values (1-D)
            pert_arr (np.array) : the perturbation of the interpolated values (1-D)
        """
        gll_file = os.path.join(self.databases_dir, f'{kernel_name}.vtu')
        
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(gll_file))
        reader.Update()
          
        ugrid = reader.GetOutput()
        points = ugrid.GetPoints().GetData()
        points_array = vtk_to_numpy(points)
  
        point_data = ugrid.GetPointData()
        data = point_data.GetArray(str(kernel_name))
        data_array = vtk_to_numpy(data)
        
        position_arr = points_array.T
        # change depth of the positive direction
        given_dep = position_arr[2,:] / -1.
        given_x, given_y = position_arr[0,:], position_arr[1,:]
        
        query_lon_arr = self.query_lon_arr
        query_lat_arr = self.query_lat_arr
        query_lon, query_lat = np.meshgrid(query_lon_arr, query_lat_arr, indexing='ij')
        query_lon = query_lon.flatten()
        query_lat = query_lat.flatten()
        query_x, query_y = lonlat_to_utm(lon=query_lon, lat=query_lat, utm_zone=utm_zone, is_north_hemisphere=is_north_hemisphere)
        
        abs_list = []
        pert_list = []
        for specified_dep in self.query_dep_arr:
            print(specified_dep)
            specified_dep_meter = specified_dep * 1E+03
            double_dep_interval = self.dep_interval * 2 * 1E+03
            dep_filter = (given_dep >= specified_dep_meter - double_dep_interval)&(given_dep <= specified_dep_meter + double_dep_interval)
            query_dep = np.ones_like(query_x) * specified_dep_meter
            given_dep_filter = given_dep[dep_filter]
            given_x_filter = given_x[dep_filter]
            given_y_filter = given_y[dep_filter]
            data_array_filter = data_array[dep_filter]
            
            interp_arr = get_values_by_kdtree(query_lon=query_x, query_lat=query_y, query_dep=query_dep,
                                            given_lon=given_x_filter, given_lat=given_y_filter, given_dep=given_dep_filter,
                                            data_arr=data_array_filter, max_distance=max_distance)
            mean_in_this_dep = np.nanmean(interp_arr)
            data_values_pert_arr = (interp_arr - mean_in_this_dep) / mean_in_this_dep * 1E+02
            abs_list.append(interp_arr)
            pert_list.append(data_values_pert_arr)
        
        abs_arr = np.hstack(abs_list)
        pert_arr = np.hstack(pert_list)
        
        return abs_arr, pert_arr
    
    def output_txt_file(self, output_file_name, v1_abs, v1_pert, v2_abs, v2_pert, v3_abs, v3_pert):
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
    
    def plot_vertical_profile(self):
        """
        Plot the vertical profile of the kernel
        """
        pass