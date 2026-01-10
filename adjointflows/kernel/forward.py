from tools import GLOBAL_PARAMS
from tools.job_utils import check_if_directory_not_empty, remove_file, remove_files_with_pattern, move_files, wait_for_launching, check_path_is_correct
from tools.matrix_utils import get_param_from_specfem_file
from pathlib import Path

import pandas as pd
import numpy as np
import subprocess
import logging
import os
import sys
import shutil
import time
import csv

class ForwardGenerator:
    
    def __init__(self, current_model_num, config):
        """
        Args:
            config (dict): The configuration dictionary
            current_model_num (int): The current model number
        """
        self.base_dir          = GLOBAL_PARAMS['base_dir']
        self.mpirun_path       = GLOBAL_PARAMS['mpirun_path']
        self.current_model_num = current_model_num
        self.specfem_dir       = os.path.join(self.base_dir, 'specfem3d')
        self.databases_mpi_dir = os.path.join(self.specfem_dir, 'DATABASES_MPI')
        self.measure_adj_dir   = os.path.join(self.base_dir, 'measure_adj')
        self.flexwin_dir       = os.path.join(self.base_dir, 'flexwin')
        
        self.stage_initial_model = config.get('setup.stage.stage_initial_model')
        self.ichk                = config.get('preprocessing.ICHK')
        # self.flexwin_flag        = config.get('setup.flexwin.FLEXWIN_FLAG')
        self.flexwin_mode        = config.get('setup.flexwin.flexwin_mode')
        self.flexwin_user_dir    = config.get('setup.flexwin.flexwin_user_dir')
        self.source_type         = (config.get('source.type') or 'cmt').lower()
        if self.source_type not in ('cmt', 'force'):
            raise ValueError(f"Unknown source.type: {self.source_type}")
        self.force_depth_km      = config.get('source.force.depth_km', 0.0)
        if self.force_depth_km is None:
            self.force_depth_km = 0.0
        self.force_auto_set_par  = bool(config.get('source.force.auto_set_par_file', True))
        self.dummy_cmt_date      = '2000/01/01'
        self.dummy_cmt_time      = '00:00:00'
        self.dummy_cmt_mag       = 1.0
        self.dummy_cmt_moment    = (1.0e20, 1.0e20, 1.0e20, 0.0, 0.0, 0.0)
        self.force_factor        = '1.d10'
        self.force_direction     = (0.0, 0.0, -1.0)
        self.force_stf_type      = 0
        self.force_hdurorf0      = 0.0
        
        self.evlst               = os.path.join(self.base_dir, 'DATA', 'evlst', config.get('data.list.evlst'))
        self.stlst               = os.path.join(self.base_dir, 'DATA', 'stlst', config.get('data.list.stlst'))
        self.specfem_par_file    = os.path.join(self.specfem_dir, 'DATA', 'Par_file')   
        
        self.nproc               = get_param_from_specfem_file(file=self.specfem_par_file, param_name='NPROC', param_type=int)
        self.pbs_nodefile      = os.path.join(self.base_dir, 'adjointflows', 'nodefile')

        self.debug_logger      = logging.getLogger("debug_logger")
        self.result_logger     = logging.getLogger("result_logger")

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
        if self.source_type == 'force':
            self.ensure_force_point_source()

    def output_vars_file(self):
        """
        Save the current model number in the txt file
        """
        env_vars = {
            "mrun": str(self.current_model_num),
            "stage_initial_model": str(self.stage_initial_model)
        }
        env_file = os.path.join(self.specfem_dir, "env_vars.txt")
        
        remove_file(env_file)
        
        with open(env_file, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key} = {value}\n")
    
    """
    the functions below are the features for controlling specfem3d forward
    REPLACE the original kernel_serial.bash
    """
    def check_last_event(self):
        """
        If ICHK is set to 1, check which event is the last event, and start from that event.
        """

        if self.ichk == 1:
            index_evt_last = 0
            ev_list_path = self.evlst
            if os.path.exists(ev_list_path):
                with open(ev_list_path, "r") as f:
                    for line in f:
                        event_name = line.split()[0]
                        if os.path.isdir(f"KERNEL/DATABASE/{event_name}"):
                            index_evt_last += 1
            self.debug_logger.info(f"Last time stopped at event {index_evt_last}")
            return index_evt_last
        return 0

    def load_event_list(self):
        if self.source_type == 'force':
            return pd.read_csv(self.evlst, sep=r'\s+', names=['name', 'lon', 'lat', 'elev'])
        return pd.read_csv(self.evlst, sep=r'\s+',
                           names=['name', 'date', 'time', 'lon', 'lat', 'dep',
                                  'strike1', 'dip1', 'rake1', 'strike2', 'dip2', 'rake2',
                                  'Mw', 'MR', 'mrr', 'mtt', 'mpp', 'mrt', 'mrp', 'mtp'])
    
    def process_each_event(self, index_evt_last, do_forward, do_adjoint):
        """
        It controls the loop for doing forward and adjoint simulation of each event.
        Args:
            index_evt_last (int): The index of the event we will start here
            do_forward (bool): Whether we do the forward modeling using SPECFEM3D
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
            self.write_station_file(sta_df)
            
            time.sleep(2)
            
            # -----------------------
            # forward modeling
            # -----------------------
            if do_forward:
                subprocess.run('./utils/change_simulation_type.pl -F', shell=True)
                remove_files_with_pattern('OUTPUT_FILES/*.sem?')
                self.run_simulator()
                self.result_logger.info(f'Done {event_name} forward simulation')
            else:
                self.result_logger.info(f'Skip {event_name} forward simulation')
                shutil.copy("DATA/STATIONS", "DATA/STATIONS_FILTERED")
            
            keep_syn_wav = not do_forward  
            self.prepare_adjoint_simulation(event_name=event_name, keep_syn_wav=keep_syn_wav)            
            self.select_windows_and_measure_misfit(event_name=event_name)
    
            os.chdir(self.specfem_dir)
            time.sleep(2)
            
            # -----------------------
            # adjoint modeling
            # -----------------------
            if do_adjoint:
                subprocess.run('./utils/change_simulation_type.pl -b', shell=True)
                self.run_simulator()
                self.debug_logger.info(f'Done {event_name} adjoint simulation')
                
                move_files(src_dir = f'{self.specfem_dir}/DATABASES_MPI', 
                        dst_dir = f'{self.specfem_dir}/KERNEL/DATABASE/{event_name}', 
                        pattern = 'proc*kernel.bin')
                
                self.result_logger.info("kernel constructed and collected!")
            
            
    def process_each_event_for_tuning_flexwin(self, index_evt_last, do_forward):
        """
        This is a side function for tuning the flexwin parameters.
        You can choose if you want to do forward simulation (do_forward).
        If you turn do_forward to False, it will skip the forward simulation.
        Then you can tune the flexwin parameters with a short time period.
        
        Args:
            index_evt_last (int): The index of the event we will start here
            do_forward (bool): If True, do forward simulation.
        """
        evt_df = self.load_event_list()
        sta_df = pd.read_csv(self.stlst, sep=r'\s+',
                             names=['sta', 'lon', 'lat', 'elev'])
        self.debug_logger.info(f'We start from event {index_evt_last}')
        for evt_i in np.arange(index_evt_last, evt_df.shape[0]):
            event_info = evt_df.iloc[evt_i]
            event_name = str(event_info.iloc[0])
            
            self.debug_logger.info(f"Processing event {event_name}")
            self.write_source_files(event_info)
            self.write_station_file(sta_df)
            
            time.sleep(2)
            
            # -----------------------
            # forward modeling
            # -----------------------
            if do_forward:
                subprocess.run('./utils/change_simulation_type.pl -F', shell=True)
                remove_files_with_pattern('OUTPUT_FILES/*.sem?')
                self.run_simulator()
                self.debug_logger.info(f'Done {event_name} forward simulation')
                self.prepare_adjoint_simulation(event_name)
            else:
                self.debug_logger.info(f'TUNING FLEXWIN: Skip forward modeling!')          
            self.select_windows_for_tuning_flexwin(event_name=event_name)
            
            os.chdir(self.specfem_dir)
            time.sleep(2)
            
            self.result_logger.info("kernel constructed and collected!")
            self.result_logger.info("Finish testing FLEXWIN!")
            
            
            
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
        if not self.force_auto_set_par:
            return
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
                if '.true.' in value_part:
                    output_lines.append(line)
                    continue
                new_line = f"{key}= .true."
                if comment:
                    new_line += f" {comment}"
                output_lines.append(new_line.rstrip() + "\n")
                updated = True

        if updated:
            with open(self.specfem_par_file, 'w') as f:
                f.writelines(output_lines)

        
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
        nproc = self.nproc
        self.result_logger.info(f"Starting xspecfem3D on {nproc} processors...")

        if nproc == 1:
            subprocess.run(["./bin/xspecfem3D"], check=True, env=os.environ)
        else:
            subprocess.run([str(self.mpirun_path),'-np' , str(nproc), './bin/xspecfem3D'], 
                           check=True, env=os.environ)
    
    def prepare_adjoint_simulation(self, event_name, keep_syn_wav):
        """
        Prepare the adjoint simulation
        """
        syn_path = f"../SYN/{event_name}"
        syn_dir = Path(syn_path)  
        syn_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy("DATA/STATIONS_FILTERED", "../measure_adj/PLOTS")
        if not keep_syn_wav:
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
        
        if (self.flexwin_mode == 'every_stage' and (self.stage_initial_model == self.current_model_num)) or (self.flexwin_mode == 'every_iter'):
            subprocess.run(['bash', 'run_win.bash', f'{event_name}'])
        else:
            subprocess.run(['bash', 'ini_proc.bash', f'{event_name}'])
            initial_model_dir = f'm{self.stage_initial_model:03d}'
            if self.flexwin_mode == 'user':
                windows_dir = f"../TOMO/{self.flexwin_user_dir}/MEASURE/windows/{event_name}/MEASUREMENT.WINDOWS"
            else:
                windows_dir = f"../TOMO/{initial_model_dir}/MEASURE/adjoints/{event_name}/MEASUREMENT.WINDOWS"
            shutil.copy(windows_dir, "../measure_adj")
            os.chdir(self.measure_adj_dir)
            subprocess.run(['bash', 'run_adj.bash', f'{event_name}'])
    
    def select_windows_for_tuning_flexwin(self, event_name):
        """
        Run flexwin for tuning the flexwin parameters
        """
        os.chdir(self.flexwin_dir)
        subprocess.run(['bash', 'run_win_for_tune_par.bash', f'{event_name}'])
        
        put_windows_file_dir = os.path.join(f'{self.flexwin_dir}', 'PACK', f'{event_name}')
        move_files(src_dir = f'{self.flexwin_dir}', 
                       dst_dir = f'{put_windows_file_dir}', 
                       pattern = 'MEASUREMENT.WINDOWS')

            
