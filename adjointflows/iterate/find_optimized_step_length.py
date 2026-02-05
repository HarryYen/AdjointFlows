from tools import GLOBAL_PARAMS, FileManager
from tools.job_utils import remove_path, check_if_directory_not_empty, remove_file, remove_files_with_pattern, move_files, wait_for_launching, clean_and_initialize_directories, check_path_is_correct
from tools.matrix_utils import get_param_from_specfem_file
from tools.dataset_loader import load_dataset_config, get_by_path, deep_merge, resolve_dataset_list_path
from pathlib import Path
from kernel import ModelGenerator

import pandas as pd
import numpy as np
import subprocess
import logging
import os
import sys
import shutil
import time
import csv
import yaml

class StepLengthOptimizer:
    
    def __init__(self, current_model_num, config, dataset_config=None):
        """
        Args:
            config (dict): The configuration dictionary
            current_model_num (int): The current model number
        """
        self.config            = config
        self.step_index        = 0
        self.optimized_step_length = 0.
        
        self.debug_logger  = logging.getLogger("debug_logger")
        self.result_logger = logging.getLogger("result_logger")
        
        self.base_dir          = GLOBAL_PARAMS['base_dir']
        self.mpirun_path       = GLOBAL_PARAMS['mpirun_path']
        self.py_mpirun_path    = GLOBAL_PARAMS['mpirun_python_path']
        self.current_model_num = current_model_num
        self.tomo_dir          = os.path.join(self.base_dir, 'TOMO')
        self.line_search_dir   = os.path.join(self.tomo_dir, f"mtest{self.current_model_num:03d}")
        self.current_tomo_dir  = os.path.join(self.tomo_dir, f"m{self.current_model_num:03d}")
        self.specfem_dir       = os.path.join(self.base_dir, 'specfem3d')

        self.databases_mpi_dir = os.path.join(self.specfem_dir, 'DATABASES_MPI')
        self.measure_adj_dir   = os.path.join(self.base_dir, 'measure_adj')
        self.flexwin_dir       = os.path.join(self.base_dir, 'flexwin')
        
        self.specfem_par_file    = os.path.join(self.specfem_dir, 'DATA', 'Par_file')
        self.adjflows_dir        = os.path.join(self.base_dir, 'adjointflows')
        self.pbs_nodefile        = os.path.join(self.adjflows_dir, 'nodefile')
        self.dummy_cmt_date      = '2000/01/01'
        self.dummy_cmt_time      = '00:00:00'
        self.dummy_cmt_mag       = 1.0
        self.dummy_cmt_moment    = (1.0e20, 1.0e20, 1.0e20, 0.0, 0.0, 0.0)
        self.force_factor        = '1.d10'
        self.force_direction     = (0.0, 0.0, -1.0)
        self.force_stf_type      = 0
        self.force_hdurorf0      = 0.0
        if dataset_config is None:
            dataset_config = load_dataset_config(self.adjflows_dir, logger=self.debug_logger)
        if not isinstance(dataset_config, dict):
            dataset_config = {}
        self.dataset_config       = dataset_config
        self.dataset_entries      = self._build_dataset_entries(dataset_config)
        self.dataset_config_paths = {}
        self.dataset_config_path  = None
        self.dataset_name         = None
        self.default_evlst_name   = get_by_path(dataset_config, "defaults.list.evchk")
        self.default_stlst_name   = get_by_path(dataset_config, "defaults.list.stlst")
        self.evlst                = None
        self.stlst                = None
        self.default_source_type  = get_by_path(dataset_config, "defaults.source.type", default="cmt")
        if self.default_source_type not in ('cmt', 'force'):
            raise ValueError(f"Unknown source.type: {self.default_source_type}")
        self.default_force_depth_km = get_by_path(dataset_config, "defaults.source.force.depth_km", default=0.0)
        if self.default_force_depth_km is None:
            self.default_force_depth_km = 0.0
        self.source_type          = self.default_source_type
        self.force_depth_km       = self.default_force_depth_km
        self.default_egf_n_wavelength = get_by_path(
            dataset_config, "defaults.seismogram.fine_tune.EGF.criteria.n_wavelength"
        )
        self.default_egf_ref_velocity_km_s = get_by_path(
            dataset_config, "defaults.seismogram.fine_tune.EGF.criteria.ref_velocity_km_s"
        )
        self.default_egf_max_period = get_by_path(
            dataset_config, "defaults.seismogram.filter.P2"
        )
        self.egf_n_wavelength     = self.default_egf_n_wavelength
        self.egf_ref_velocity_km_s = self.default_egf_ref_velocity_km_s
        self.egf_max_period       = self.default_egf_max_period
        self.file_manager         = FileManager()
        self.file_manager.set_model_number(current_model_num=self.current_model_num)
        self.flexwin_mode         = config.get('setup.flexwin.flexwin_mode')
        self.flexwin_user_dir     = config.get('setup.flexwin.flexwin_user_dir')
        self.stage_initial_model  = int(config.get('setup.stage.stage_initial_model'))

        
        self.step_interval       = config.get('line_search.step_interval')
        self.step_beg            = config.get('line_search.step_beg')
        self.step_end            = config.get('line_search.step_end')
        self.max_fail            = config.get('inversion.max_fail')
        self.shrink_factor       = config.get('inversion.backtracking_shrink')
        
        self.nproc               = get_param_from_specfem_file(file=self.specfem_par_file, param_name='NPROC', param_type=int)

    def _build_dataset_entries(self, dataset_config):
        defaults = dataset_config.get("defaults", {})
        datasets = dataset_config.get("datasets", [])
        merged = []
        for entry in datasets:
            if not isinstance(entry, dict):
                continue
            merged.append(deep_merge(defaults, entry))
        return merged

    def _iter_datasets(self):
        for entry in self.dataset_entries:
            yield entry

    def _line_search_syn_dir(self, dataset_name):
        if dataset_name:
            return f"{self.line_search_dir}/SYN{self.step_index}_{dataset_name}"
        return f"{self.line_search_dir}/SYN{self.step_index}"

    def _line_search_measure_dir(self, dataset_name):
        if dataset_name:
            return f"{self.line_search_dir}/MEASURE{self.step_index}_{dataset_name}"
        return f"{self.line_search_dir}/MEASURE{self.step_index}"

    def _measure_dir_name(self, dataset_name):
        if dataset_name:
            return f"MEASURE_{dataset_name}"
        return "MEASURE"

    def get_script_env(self):
        """Return environment with dataset-specific config path for scripts."""
        env = os.environ.copy()
        if self.dataset_config_path:
            env["AF_CONFIG"] = self.dataset_config_path
        return env

    def write_dataset_config_file(self, dataset_config):
        """Write a dataset-specific config file for FLEXWIN/MEASURE scripts."""
        dataset_name = get_by_path(dataset_config, "name", default="dataset")
        out_dir = os.path.join(self.adjflows_dir, ".dataset_configs")
        os.makedirs(out_dir, exist_ok=True)
        config_path = os.path.join(out_dir, f"line_search_{dataset_name}.yaml")

        evchk = get_by_path(dataset_config, "list.evchk")
        stlst = get_by_path(dataset_config, "list.stlst")
        if not evchk or not stlst:
            raise ValueError("Dataset config missing list.evchk or list.stlst.")

        config_data = {
            "source": {
                "type": get_by_path(dataset_config, "source.type", default="cmt"),
                "force": {
                    "depth_km": get_by_path(dataset_config, "source.force.depth_km", default=0.0),
                },
            },
            "data": {
                "list": {
                    "evlst": evchk,
                    "stlst": stlst,
                    "evchk": evchk,
                },
                "seismogram": {
                    "tbeg": get_by_path(dataset_config, "seismogram.tbeg"),
                    "tend": get_by_path(dataset_config, "seismogram.tend"),
                    "tcor": get_by_path(dataset_config, "seismogram.tcor"),
                    "dt": get_by_path(dataset_config, "seismogram.dt"),
                    "filter": {
                        "P1": get_by_path(dataset_config, "seismogram.filter.P1"),
                        "P2": get_by_path(dataset_config, "seismogram.filter.P2"),
                    },
                    "component": {
                        "COMP": get_by_path(dataset_config, "seismogram.component.COMP"),
                        "EN2RT": get_by_path(dataset_config, "seismogram.component.EN2RT"),
                    },
                },
            },
        }

        with open(config_path, "w") as f:
            yaml.safe_dump(config_data, f, sort_keys=False)
        return config_path

    def _configure_dataset(self, dataset_config):
        if not dataset_config:
            self.dataset_name = None
            self.dataset_config_path = None
            self.source_type = self.default_source_type
            self.force_depth_km = self.default_force_depth_km
            self.evlst = None
            self.stlst = None
            self.egf_n_wavelength = self.default_egf_n_wavelength
            self.egf_ref_velocity_km_s = self.default_egf_ref_velocity_km_s
            self.egf_max_period = self.default_egf_max_period
            return

        dataset_name = get_by_path(dataset_config, "name", default="dataset")
        self.dataset_name = dataset_name
        self.dataset_config_path = self.dataset_config_paths.get(dataset_name)
        if not self.dataset_config_path:
            self.dataset_config_path = self.write_dataset_config_file(dataset_config)
            self.dataset_config_paths[dataset_name] = self.dataset_config_path

        self.source_type = (get_by_path(dataset_config, "source.type", default="cmt")).lower()
        if self.source_type not in ('cmt', 'force'):
            raise ValueError(f"Unknown source.type: {self.source_type}")
        self.force_depth_km = get_by_path(dataset_config, "source.force.depth_km", default=0.0)
        if self.force_depth_km is None:
            self.force_depth_km = 0.0

        self.evlst = resolve_dataset_list_path(
            self.base_dir,
            dataset_config,
            "list.evchk",
            "evlst",
            default=self.default_evlst_name,
            required=True,
        )
        self.stlst = resolve_dataset_list_path(
            self.base_dir,
            dataset_config,
            "list.stlst",
            "stlst",
            default=self.default_stlst_name,
            required=True,
        )
        self.egf_n_wavelength = get_by_path(
            dataset_config,
            "seismogram.fine_tune.EGF.criteria.n_wavelength",
            default=self.default_egf_n_wavelength,
        )
        self.egf_ref_velocity_km_s = get_by_path(
            dataset_config,
            "seismogram.fine_tune.EGF.criteria.ref_velocity_km_s",
            default=self.default_egf_ref_velocity_km_s,
        )
        self.egf_max_period = get_by_path(
            dataset_config,
            "seismogram.filter.P2",
            default=self.default_egf_max_period,
        )

    def _link_dataset_resources(self, dataset_config):
        if not dataset_config:
            return
        dataset_name = self.dataset_name or get_by_path(dataset_config, "name", default="dataset")
        data_waveform_dir = get_by_path(dataset_config, "data.waveform_dir")
        syn_waveform_dir = get_by_path(dataset_config, "synthetics.waveform_dir")
        self.file_manager.ensure_dataset_dirs(dataset_name, syn_waveform_dir=syn_waveform_dir)
        self.file_manager.link_dataset_dirs(
            dataset_name,
            data_waveform_dir,
            syn_waveform_dir=syn_waveform_dir,
        )
        self.file_manager.link_measurement_tools(
            flexwin_bin=get_by_path(dataset_config, "flexwin.bin_file"),
            flexwin_par=get_by_path(dataset_config, "flexwin.par_file"),
            measure_adj_bin=get_by_path(dataset_config, "measure_adj.bin_file"),
            measure_adj_par=get_by_path(dataset_config, "measure_adj.par_file"),
        )

    # --------------------------------------------------
    # ---------------- For Controlling -----------------
    # --------------------------------------------------
    def run_line_search(self):
        """
        Run the line search algorithm to find the optimized step length
        This is for SD method
        """
        
        model_generator_line_search = ModelGenerator()
        datasets = list(self._iter_datasets())
        if not datasets:
            raise ValueError("No datasets defined in dataset.yaml; line search requires dataset entries.")
        
        self.result_logger.info(f"Starting line search for model {self.current_model_num:03d}...")
        self.give_current_best_step_length(step_length_tmp=0.)
        for step_length in np.arange(self.step_beg, self.step_end, self.step_interval):
            
            self.result_logger.info(f"LINE SEARCH: Start step length {step_length}")
            self.increase_step_index()
            self.setup_directory()
            self.make_symbolic_links()
            self.update_model(step_fac=step_length, lbfgs_flag=False)
            os.chdir(self.specfem_dir)
            model_generator_line_search.model_setup(mesh_flag=False)

            for dataset_entry in datasets:
                self._configure_dataset(dataset_entry)
                self._link_dataset_resources(dataset_entry)
                self.setup_directory(dataset_name=self.dataset_name)
                self.make_symbolic_links(dataset_name=self.dataset_name)
                os.chdir(self.specfem_dir)
                self.preprocessing()
                self.process_each_event(index_evt_last=0)
                os.chdir(self.adjflows_dir)
            
            if self.is_misfit_reduced():
                self.give_current_best_step_length(step_length_tmp=step_length)
                continue
            else:
                break
              
    
    # --------------------------------------------------
    def increase_step_index(self):
        self.step_index += 1
        
    def give_current_best_step_length(self, step_length_tmp):
        """
        Give the temporary step length for the inversion
        Args:
            step_length_tmp (float): The temporary step length
        """
        self.optimized_step_length = step_length_tmp
    
    def get_current_best_step_length(self):
        """
        Get the optimized step length
        """
        return self.optimized_step_length
        
    def setup_directory(self, dataset_name=None):
        """
        Create a series of directories for inversion.
        it will call clean_and_initialize_directories to remove all files in the directories
        if the clear_directories is True.
        """
        syn_dir = self._line_search_syn_dir(dataset_name)
        measure_dir = self._line_search_measure_dir(dataset_name)
        dirs = [
            f"{self.line_search_dir}/DATABASES_MPI",
            syn_dir,
            measure_dir,
    ]
    
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)
        
        clean_and_initialize_directories(dirs[1:])
    
    def make_symbolic_links(self, dataset_name=None):
        """
        We need to create the symbolic links for the SEM simulation.
        Here, it will link 
        """
        syn_dir = self._line_search_syn_dir(dataset_name)
        measure_dir = self._line_search_measure_dir(dataset_name)
        link_directories = [
            f'{self.specfem_dir}/DATABASES_MPI', 
            f'{self.base_dir}/SYN',  
            f'{self.base_dir}/measure_adj/PACK'
        ]
        target_directories = [
            f"{self.line_search_dir}/DATABASES_MPI",
            syn_dir,
            measure_dir,
        ]
        
        remove_path(link_directories)
        
        if len(link_directories) != len(target_directories):
            message = "The length of link_directories and target_directories should be the same"
            self.debug_logger.error(message)
            raise ValueError(message)
        
        for index in range(len(target_directories)):
            target = target_directories[index]
            link = link_directories[index]
            os.symlink(target, link)
    
    
    

    def preprocessing(self):
        """
        some preprocessing before forward simulation
        check if the DATABASES_MPI is not empty, then we can do the forward simulation
        """
        
        if not check_if_directory_not_empty(self.databases_mpi_dir):
            self.debug_logger.error(f"STOP: {self.databases_mpi_dir} is empty!")
            sys.exit()
        
        if not check_path_is_correct(self.specfem_dir):
            error_message = f"STOP: the current directory is not {self.specfem_dir}!"
            self.debug_logger.error(error_message)
            raise ValueError(error_message)
        self.ensure_force_point_source()
        
        kernel_ready_file = os.path.join(self.specfem_dir, 'kernel_databases_ready')
        remove_file(kernel_ready_file)

    def load_event_list(self):
        if self.source_type == 'force':
            return pd.read_csv(self.evlst, sep=r'\s+', names=['name', 'lon', 'lat', 'elev'])
        return pd.read_csv(self.evlst, sep=r'\s+',
                           names=['name', 'date', 'time', 'lon', 'lat', 'dep',
                                  'strike1', 'dip1', 'rake1', 'strike2', 'dip2', 'rake2',
                                  'Mw', 'MR', 'mrr', 'mtt', 'mpp', 'mrt', 'mrp', 'mtp'])

    def process_each_event(self, index_evt_last):
        """
        It controls the loop for doing forward and adjoint simulation of each event.
        Args:
            index_evt_last (int): The index of the event we will start here
        """
        evt_df = self.load_event_list()
        sta_df = pd.read_csv(self.stlst, sep='\s+',
                             names=['sta', 'lon', 'lat', 'elev'])
        self.debug_logger.info(f'We start from event {index_evt_last}')
        for evt_i in np.arange(index_evt_last, evt_df.shape[0]):
            event_info = evt_df.iloc[evt_i]
            event_name = str(event_info.iloc[0])
            self.debug_logger.info(f"Processing event {event_name}")
            self.write_source_files(event_info)
            filtered_sta_df = sta_df
            if self.source_type == 'force':
                filtered_sta_df = self.filter_stations_for_egf(event_info, sta_df, event_name)
                if filtered_sta_df.empty:
                    self.result_logger.warning(
                        f"EGF distance filter: no stations left for {event_name}; skipping event."
                    )
                    continue
            self.write_station_file(filtered_sta_df)
            
            time.sleep(2)
            
            # -----------------------
            # forward modeling
            # -----------------------
            subprocess.run('./utils/change_simulation_type.pl -F', shell=True)
            remove_files_with_pattern('OUTPUT_FILES/*.sem?')
            self.run_simulator()
            self.debug_logger.info(f'Done {event_name} forward simulation')
            self.prepare_adjoint_simulation(event_name)            
            self.select_windows_and_measure_misfit(event_name=event_name)
            
            os.chdir(self.specfem_dir)
            time.sleep(2)
            
            # -----------------------
            # adjoint modeling
            # -----------------------
            # subprocess.run('./utils/change_simulation_type.pl -b', shell=True)
            # self.run_simulator()
            self.debug_logger.info(f'Done {event_name} adjoint simulation')
            
            move_files(src_dir = f'{self.specfem_dir}/DATABASES_MPI', 
                       dst_dir = f'{self.specfem_dir}/KERNEL/DATABASE/{event_name}', 
                       pattern = 'proc*kernel.bin')
            
            self.result_logger.info("kernel constructed and collected!")
            

    def write_cmt_file(self, event_info):
        """
        Write the cmt file (CMTSOLUTION) of the given event for SPECFEM3D 
        Args:
            event_info (pd.Series): The information of the current event
        """
        name = str(event_info['name'])
        date = str(event_info['date'])
        time = str(event_info['time'])
        lat = event_info['lat']
        lon = event_info['lon']
        dep = event_info['dep']
        mag = event_info['Mw']
        mrr = event_info['mrr'] * 1e+24
        mtt = event_info['mtt'] * 1e+24
        mpp = event_info['mpp'] * 1e+24
        mrt = event_info['mrt'] * 1e+24
        mrp = event_info['mrp'] * 1e+24
        mtp = event_info['mtp'] * 1e+24
        
        date_split = date.split('/')
        time_split = time.split(':')
        year, month, day = int(date_split[0]), int(date_split[1]), int(date_split[2])
        hour, min, sec = int(time_split[0]), int(time_split[1]), float(time_split[2])
        
        cmt_file = os.path.join(self.specfem_dir, 'DATA', 'CMTSOLUTION')
        remove_file(cmt_file)
        
        with open(cmt_file, 'w') as f:
            f.write(f"PDE {year:4d} {month:2d} {day:2d} {hour:2d} {min:2d} {sec:5.2f} {lat:6.3f} {lon:6.3f} {dep:4.1f} {mag:3.1f} {mag:3.1f} {name:>12}\n")
            f.write(f"event name: {name:>12}\n")
            f.write(f"time shift: 0.0\n")
            f.write(f"half duration: 0.0\n")
            f.write(f"latitude: {lat:6.3f}\n")
            f.write(f"longitude: {lon:6.3f}\n")
            f.write(f"depth: {dep:6.2f}\n")
            f.write(f"Mrr: {mrr:13.6e}\n")
            f.write(f"Mtt: {mtt:13.6e}\n")
            f.write(f"Mpp: {mpp:13.6e}\n")
            f.write(f"Mrt: {mrt:13.6e}\n")
            f.write(f"Mrp: {mrp:13.6e}\n")
            f.write(f"Mtp: {mtp:13.6e}\n")
        
        shutil.copy(cmt_file, self.measure_adj_dir)

    def write_dummy_cmt_file(self, source_info):
        name = str(source_info['name'])
        date = self.dummy_cmt_date.replace('-', '/')
        time_str = self.dummy_cmt_time
        lat = float(source_info['lat'])
        lon = float(source_info['lon'])
        dep = float(self.force_depth_km)
        mag = float(self.dummy_cmt_mag)
        mrr, mtt, mpp, mrt, mrp, mtp = self.dummy_cmt_moment

        date_split = date.split('/')
        time_split = time_str.split(':')
        year, month, day = int(date_split[0]), int(date_split[1]), int(date_split[2])
        hour, min, sec = int(time_split[0]), int(time_split[1]), float(time_split[2])

        cmt_file = os.path.join(self.specfem_dir, 'DATA', 'CMTSOLUTION')
        remove_file(cmt_file)

        with open(cmt_file, 'w') as f:
            f.write(f"PDE {year:4d} {month:2d} {day:2d} {hour:2d} {min:2d} {sec:5.2f} {lat:6.3f} {lon:6.3f} {dep:4.1f} {mag:3.1f} {mag:3.1f} {name:>12}\n")
            f.write(f"event name: {name:>12}\n")
            f.write("time shift: 0.0\n")
            f.write("half duration: 0.0\n")
            f.write(f"latitude: {lat:6.3f}\n")
            f.write(f"longitude: {lon:6.3f}\n")
            f.write(f"depth: {dep:6.2f}\n")
            f.write(f"Mrr: {mrr:13.6e}\n")
            f.write(f"Mtt: {mtt:13.6e}\n")
            f.write(f"Mpp: {mpp:13.6e}\n")
            f.write(f"Mrt: {mrt:13.6e}\n")
            f.write(f"Mrp: {mrp:13.6e}\n")
            f.write(f"Mtp: {mtp:13.6e}\n")

        shutil.copy(cmt_file, self.measure_adj_dir)

    def write_force_file(self, source_info):
        lat = float(source_info['lat'])
        lon = float(source_info['lon'])
        dep = float(self.force_depth_km)
        dir_e, dir_n, dir_z = self.force_direction
        output_lines = [
            "FORCE  001\n",
            "time shift:     0.0000\n",
            f"hdurorf0:        {self.force_hdurorf0}\n",
            f"latorUTM:       {lat:10.4f}\n",
            f"longorUTM:      {lon:10.4f}\n",
            f"depth:          {dep:10.4f}\n",
            f"source time function:            {self.force_stf_type}\n",
            f"factor force source:             {self.force_factor}\n",
            f"component dir vect source E:     {dir_e}\n",
            f"component dir vect source N:     {dir_n}\n",
            f"component dir vect source Z_UP:  {dir_z}\n",
        ]

        force_file = os.path.join(self.specfem_dir, 'DATA', 'FORCESOLUTION')
        remove_file(force_file)
        with open(force_file, 'w') as f:
            f.writelines(output_lines)

    def write_source_files(self, event_info):
        if self.source_type == 'force':
            self.write_force_file(event_info)
            self.write_dummy_cmt_file(event_info)
            return
        self.write_cmt_file(event_info)

    def ensure_force_point_source(self):
        target_value = '.true.' if self.source_type == 'force' else '.false.'
        updated = False
        output_lines = []
        with open(self.specfem_par_file, 'r') as f:
            for line in f:
                stripped = line.strip()
                if not stripped.startswith('USE_FORCE_POINT_SOURCE'):
                    output_lines.append(line)
                    continue
                key, rest = line.split('=', 1)
                value_part = rest
                comment = ''
                if '#' in rest:
                    value_part, comment = rest.split('#', 1)
                    comment = '#' + comment.rstrip('\n')
                if target_value in value_part.lower():
                    output_lines.append(line)
                    continue
                new_line = f"{key}= {target_value}"
                if comment:
                    new_line += f" {comment}"
                output_lines.append(new_line.rstrip() + "\n")
                updated = True

        if updated:
            with open(self.specfem_par_file, 'w') as f:
                f.writelines(output_lines)

        
    def haversine_km(self, lon1, lat1, lon2, lat2):
        """Calculate great-circle distance in kilometers.

        Args:
            lon1 (float or np.ndarray): Longitude(s) of the first point(s).
            lat1 (float or np.ndarray): Latitude(s) of the first point(s).
            lon2 (float or np.ndarray): Longitude(s) of the second point(s).
            lat2 (float or np.ndarray): Latitude(s) of the second point(s).

        Returns:
            np.ndarray: Great-circle distance(s) in kilometers.
        """
        lon1_rad = np.radians(lon1)
        lat1_rad = np.radians(lat1)
        lon2_rad = np.radians(lon2)
        lat2_rad = np.radians(lat2)

        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.sqrt(a))
        return 6371.0 * c

    def filter_stations_for_egf(self, event_info, sta_df, event_name):
        """Filter stations based on EGF distance rule in force mode.

        Args:
            event_info (pd.Series): Event info for the current EGF source.
            sta_df (pd.DataFrame): Station list.
            event_name (str): Event name for logging.

        Returns:
            pd.DataFrame: Filtered stations.
        """
        if self.egf_n_wavelength is None or self.egf_ref_velocity_km_s is None or self.egf_max_period is None:
            self.debug_logger.warning(
                "EGF distance filter skipped: missing data.egf.n_wavelength, "
                "data.egf.ref_velocity_km_s, or data.filter.P2."
            )
            return sta_df

        min_distance_km = (
            float(self.egf_max_period) * float(self.egf_n_wavelength) * float(self.egf_ref_velocity_km_s)
        )
        if min_distance_km <= 0.0:
            self.debug_logger.warning("EGF distance filter skipped: min distance is <= 0 km.")
            return sta_df

        src_lat = float(event_info['lat'])
        src_lon = float(event_info['lon'])
        distances_km = self.haversine_km(
            src_lon,
            src_lat,
            sta_df['lon'].to_numpy(dtype=float),
            sta_df['lat'].to_numpy(dtype=float),
        )
        keep_mask = distances_km >= min_distance_km
        filtered_df = sta_df.loc[keep_mask].copy()
        removed = len(sta_df) - len(filtered_df)
        self.result_logger.info(
            f"EGF distance filter: {event_name} min={min_distance_km:.2f} km "
            f"kept={len(filtered_df)} removed={removed}"
        )
        return filtered_df


    def write_station_file(self, sta_df):
        """
        Write STATIONS file for SPECFEM3D in specfem3d/DATA.
        Args:
            sta_df (pd.DataFrame): The dataframe of the stations
        """
        sta_file = os.path.join(self.specfem_dir, 'DATA', 'STATIONS')
        remove_file(sta_file)
        new_sta_data = {
            'sta': sta_df['sta'].values,
            'net': ['TW'] * sta_df.shape[0],
            'lat': sta_df['lat'].values,
            'lon': sta_df['lon'].values,
            'zero1': np.zeros(sta_df.shape[0]),
            'zero2': np.zeros(sta_df.shape[0])
        }
        new_sta_df = pd.DataFrame(new_sta_data)
        
        # Formatted
        new_sta_df['lon'] = new_sta_df['lon'].apply(lambda x: f"{x:6.3f}")
        new_sta_df['lat'] = new_sta_df['lat'].apply(lambda x: f"{x:6.3f}")
        new_sta_df['zero1'] = new_sta_df['zero1'].apply(lambda x: f"{x:2.1f}")
        new_sta_df['zero2'] = new_sta_df['zero2'].apply(lambda x: f"{x:2.1f}")
        
        new_sta_df.to_csv(sta_file, index=False, header=False, sep=" ",
                          quoting=csv.QUOTE_NONE)
    
    def run_simulator(self):
        """ 
        run xspecfem3D        
        """
        env = os.environ.copy()
        nproc = self.nproc
        self.result_logger.info(f"Starting xspecfem3D on {nproc} processors...")

        if nproc == 1:
            subprocess.run(["./bin/xspecfem3D", ], env=env, check=True)
        else:
            subprocess.run([str(self.mpirun_path), '-np' , str(nproc), './bin/xspecfem3D'], env=env, check=True)
    
    def prepare_adjoint_simulation(self, event_name):
        """
        Prepare the adjoint simulation
        """
        syn_path = f"../SYN/{event_name}"
        syn_dir = Path(syn_path)  
        syn_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy("DATA/STATIONS_FILTERED", "../measure_adj/PLOTS")
        remove_files_with_pattern(f'{syn_path}/*')
        
        for file in Path("OUTPUT_FILES").glob("*.sem?"):
            shutil.move(str(file), str(syn_dir))
            
        remove_files_with_pattern(f'../DATA/wav/{event_name}/*.sac.tomo')
        remove_file('DATA/STATIONS_ADJOINT')
        
    def select_windows_and_measure_misfit(self, event_name):
        """
        Run flexwin and measure_adj
        """
        os.chdir(self.flexwin_dir)
        env = self.get_script_env()
        measure_dir_name = self._measure_dir_name(self.dataset_name)

        if (self.flexwin_mode == 'every_stage' and (self.stage_initial_model == self.current_model_num)) or (self.flexwin_mode == 'every_iter'):
            subprocess.run(['bash', 'run_win.bash', f'{event_name}'], env=env)
        else:
            subprocess.run(['bash', 'ini_proc.bash', f'{event_name}'], env=env)
            initial_model_dir = f'm{self.stage_initial_model:03d}'
            if self.flexwin_mode == 'user':
                windows_dir = (
                    f"../TOMO/{self.flexwin_user_dir}/{measure_dir_name}"
                    f"/windows/{event_name}/MEASUREMENT.WINDOWS"
                )
            else:
                windows_dir = (
                    f"../TOMO/{initial_model_dir}/{measure_dir_name}"
                    f"/adjoints/{event_name}/MEASUREMENT.WINDOWS"
                )
            if not os.path.isfile(windows_dir):
                self.result_logger.warning(
                    f"MEASUREMENT.WINDOWS missing for {event_name}; skip measure_adj. "
                    f"path={windows_dir}"
                )
                return
            shutil.copy(windows_dir, "../measure_adj")
            os.chdir(self.measure_adj_dir)
            subprocess.run(['bash', 'run_adj.bash', f'{event_name}'], env=env)
        
        
    def _misfit_for_dataset(self, step_index, dataset_name, evlst):
        if step_index == 0:
            measure_dir = f'{self.current_tomo_dir}/{self._measure_dir_name(dataset_name)}/adjoints'
        else:
            measure_dir = self._line_search_measure_dir(dataset_name)

        evt_df = pd.read_csv(evlst, header=None, sep=r'\s+')
        chi_df = pd.DataFrame()
        missing_chi = []
        for evt in evt_df[0]:
            adjoints_dir = f'{measure_dir}/{evt}'
            chi_file = f'{adjoints_dir}/window_chi'
            if not os.path.isfile(chi_file):
                missing_chi.append(evt)
                continue
            try:
                tmp_df = pd.read_csv(chi_file, header=None, sep=r'\s+')
            except Exception as exc:
                self.result_logger.warning(f"Skip {evt}: failed to read {chi_file} ({exc})")
                missing_chi.append(evt)
                continue
            chi_df = pd.concat([chi_df, tmp_df])

        if missing_chi:
            missing_str = ", ".join(missing_chi)
            self.result_logger.warning(f"Missing window_chi for events: {missing_str}. Please check window selection.")
        if chi_df.empty:
            self.result_logger.warning("No window_chi files found; misfit set to 0.")
            return 0.0

        chi_filtered_df = chi_df[(chi_df[28] != 0.) | (chi_df[29] != 0.)]
        if chi_filtered_df.empty:
            self.result_logger.warning("No valid windows; misfit set to 0.")
            return 0.0
        total_misfit = chi_filtered_df[28].sum()
        win_num = len(chi_filtered_df)
        if win_num == 0:
            self.result_logger.warning("No valid windows; misfit set to 0.")
            return 0.0
        average_misfit = round(total_misfit / win_num, 5)
        return average_misfit

    def misfit_calculation(self, step_index):
        """
        Calculate the misfit for the given model number
        Return:
            misfit (float): The misfit value
        """
        if not self.dataset_entries:
            raise ValueError("No datasets defined in dataset.yaml; misfit calculation requires dataset entries.")

        total_misfit = 0.0
        total_weight = 0.0
        for dataset_entry in self.dataset_entries:
            dataset_name = get_by_path(dataset_entry, "name", default="dataset")
            weight = float(get_by_path(dataset_entry, "inversion.weight", 1.0))
            evlst = resolve_dataset_list_path(
                self.base_dir,
                dataset_entry,
                "list.evchk",
                "evlst",
                default=self.default_evlst_name,
                required=True,
            )
            misfit = self._misfit_for_dataset(step_index, dataset_name, evlst)
            total_misfit += misfit * weight
            total_weight += weight

        if total_weight == 0.0:
            self.result_logger.warning("Total dataset weight is 0; misfit set to 0.")
            return 0.0
        return total_misfit / total_weight
    
    def is_misfit_reduced(self):
        """
        Determine if the misfit is reduced.
        Note:
            check if the misfit is reduced in the current step. If the misfit doesn't decrease in the first round,
            the program will stop. 
        Return:
            is_reduced (bool): True for PASS, False otherwise
        """
        
        current_misfit  = self.misfit_calculation(self.step_index)
        previous_misfit = self.misfit_calculation(self.step_index - 1)
        if current_misfit < previous_misfit:
            self.result_logger.info(f"LINE SEARCH: the misfit of mtest{self.step_index} is smaller than mtest{self.step_index - 1}. Continue!")
            return True
        else:
            if self.step_index == 1:
                error_message = f"LINE SEARCH: the misfit increased in the first step. Stop!"
                self.result_logger.error(error_message)
                raise ValueError(error_message)
            else:
                self.result_logger.info(f"LINE SEARCH: the misfit of mtest{self.step_index} is larger than mtest{self.step_index - 1}. We choose index:{self.step_index - 1} / step fac: {self.get_current_best_step_length} to be step length. Stop!")
                return False
    
    def update_model(self, step_fac, lbfgs_flag=False):
        """
        Update the model for line search!
        step_fac (float): The step factor for the update (e.g. 0.03 for +-3% update)
        lbfgs_flag (bool): True for L-BFGS method, False for SD
        """
        env = os.environ.copy()
        nproc = self.nproc
        self.result_logger.info(f"Line Search: Starting updating model on {nproc} processors...")
        os.chdir(self.adjflows_dir)
        
        script_dir = "iterate/model_update.py"
        line_search_flag = True
        
        if nproc == 1:
            command = f"python {script_dir} {step_fac} {int(lbfgs_flag)} {int(line_search_flag)}"
            subprocess.run(command, shell=True, check=True)
        else:
            command = f'{self.py_mpirun_path} -np {nproc} python {script_dir} {step_fac} {int(lbfgs_flag)} {int(line_search_flag)}'
            self.debug_logger.info(f"Command: {command}")
            subprocess.run(command, shell=True, check=True, env=env)
        
    # -----------------------------------------------
    # ----------------- For L-BFGS ------------------
    # -----------------------------------------------
    def run_backtracking_line_search(self, step_length_init, min_alpha=1e-02):
        """
        Run the backtracking line search algorithm to find the optimized step length
        This is for LBFGS method
        Args:
            step_length_init (float): The initial step length for the line search
            min_alpha (float): The minimum step length, default is 0.01
        """
        
        model_generator_line_search = ModelGenerator()
        datasets = list(self._iter_datasets())
        if not datasets:
            raise ValueError("No datasets defined in dataset.yaml; line search requires dataset entries.")
        
        self.result_logger.info(f"Starting backtracking line search for model {self.current_model_num:03d}...")
        self.give_current_best_step_length(step_length_tmp=0.)
        
        alpha = step_length_init
        
        misfit_zero_step = self.misfit_calculation(step_index=0)
        # setup a initial step length
        self.give_current_best_step_length(step_length_tmp=0.)

        pass_zero_step = False
        while True:
            
            if alpha < min_alpha:
                self.result_logger.warning(f"[L-BFGS line search] the step length is smaller than minimum, STOP!")
                raise ValueError(f"[L-BFGS line search] the step length is smaller than minimum, STOP!")
        
            self.increase_step_index() 
            self.result_logger.info(f"BACKTRACKING LINE SEARCH: step index: {self.step_index}")
            self.result_logger.info(f"BACKTRACKING LINE SEARCH: Start step length from {alpha}")
                   
            self.setup_directory()
            self.make_symbolic_links()
            self.update_model(step_fac=alpha, lbfgs_flag=True)
            os.chdir(self.specfem_dir)
            model_generator_line_search.model_setup(mesh_flag=False)
            for dataset_entry in datasets:
                self._configure_dataset(dataset_entry)
                self._link_dataset_resources(dataset_entry)
                self.setup_directory(dataset_name=self.dataset_name)
                self.make_symbolic_links(dataset_name=self.dataset_name)
                os.chdir(self.specfem_dir)
                self.preprocessing()
                self.process_each_event(index_evt_last=0)
                os.chdir(self.adjflows_dir)
            
            misfit_new = self.misfit_calculation(self.step_index)
            misfit_old = self.misfit_calculation(self.step_index - 1)                
            self.result_logger.info(f"BACKTRACKING LINE SEARCH: old misfit: {misfit_old}")
            self.result_logger.info(f"BACKTRACKING LINE SEARCH: new misfit: {misfit_new}")

            # Once the misfit is smaller than the misfit in zero step, we can set pass_zero_step flag as True
            if misfit_new < misfit_zero_step:
                pass_zero_step = True
            
            if not pass_zero_step:
                self.result_logger.info(f"BACKTRACKING LINE SEARCH: in step index {self.step_index} and misfit_new is still larger than the misfit in zero step")
                self.result_logger.info(f"BACKTRACKING LINE SEARCH: so we reduce the step length and keep finding a better step which can reduce misfit!")
                alpha *= self.shrink_factor 
                continue

            elif misfit_new < misfit_old:
                self.result_logger.info(f"BACKTRACKING LINE SEARCH: misfit reduced! NEXT")
                self.give_current_best_step_length(step_length_tmp=alpha)
                alpha *= self.shrink_factor 
                continue
            else:
                break


    def quadratic_interpolation(self, phi_o, phi0, alpha, phi_grad):
        """
        Calculating a new step length by quadratic interpolation
        Args:
            phi_o (float): The misfit of the previous model
            phi0 (float): The misfit of the current model
            alpha (float): The step length
            phi_grad (float): The gradient of the misfit
        """
        numerator = phi_grad * alpha ** 2
        denomintor = 2 * (phi0 - phi_o - alpha * phi_grad)
        return - numerator / denomintor

    def cubic_interpolation(self, phi_o, phi0, phi1, alpha0, alpha1, phi_grad):
        """
        Calculating a new step length by cubic interpolation
        Args:
            phi_o (float): The misfit of the previous model
            phi0 (float): The misfit of the model tried 2 steps ago
            phi1 (float): The misfit of the model tried 1 step ago
            alpha0 (float): The step length of the model 2 steps ago
            alpha1 (float): The step length of the model 1 step ago
            phi_grad (float): The gradient of the misfit
        """
        
        denomimator = (alpha0 ** 2) * (alpha1 ** 2) * (alpha1 - alpha0)
        a = ( (alpha0 ** 2) * (phi1 - phi_o - phi_grad * alpha1 ) - ( alpha1 ** 2 ) * (phi0 - phi_o - phi_grad * alpha0) ) / denomimator
        b = (-(alpha0 ** 3) * (phi1 - phi_o - phi_grad * alpha1 ) + ( alpha1 ** 3 ) * (phi0 - phi_o - phi_grad * alpha0) ) / denomimator
        
        discriminant = b ** 2 - 3 * a * phi_grad
        if discriminant < 0 or a == 0:
            return alpha1 / 2
        else:
            return (-b + np.sqrt(discriminant)) / (3 * a)
