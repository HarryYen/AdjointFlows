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
        self.flexwin_flag        = config.get('setup.flexwin.FLEXWIN_FLAG')
        
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
    
    def process_each_event(self, index_evt_last):
        """
        It controls the loop for doing forward and adjoint simulation of each event.
        Args:
            index_evt_last (int): The index of the event we will start here
        """
        evt_df = pd.read_csv(self.evlst, sep='\s+', 
                             names=['name', 'date', 'time', 'lon', 'lat', 'dep',
                                    'strike1', 'dip1', 'rake1', 'strike2', 'dip2', 'rake2',
                                    'Mw', 'MR', 'mrr', 'mtt', 'mpp', 'mrt', 'mrp', 'mtp'])
        sta_df = pd.read_csv(self.stlst, sep='\s+',
                             names=['sta', 'lon', 'lat', 'elev'])
        self.debug_logger.info(f'We start from event {index_evt_last}')
        for evt_i in np.arange(index_evt_last, evt_df.shape[0]):
            event_info = evt_df.iloc[evt_i]
            event_name = event_info[0]
            
            self.debug_logger.info(f"Processing event {event_name}")
            self.write_cmt_file(event_info)
            self.write_station_file(sta_df)
            
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
            subprocess.run('./utils/change_simulation_type.pl -b', shell=True)
            self.run_simulator()
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
        name = event_info['name']
        date = event_info['date']
        time = event_info['time']
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
            f.write(f"PDE {year:4d} {month:2d} {day:2d} {hour:2d} {min:2d} {sec:5.2f} {lat:6.3f} {lon:6.3f} {dep:4.1f} {mag:3.1f} {mag:3.1f} {name:12d}\n")
            f.write(f"event name: {name:12d}\n")
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
            subprocess.run([str(self.mpirun_path), '--hostfile', str(self.pbs_nodefile), '-np' , str(nproc), './bin/xspecfem3D'], 
                           check=True, env=os.environ)
    
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
        
        if (self.stage_initial_model == self.current_model_num) or (self.flexwin_flag == 1):
            subprocess.run(['bash', 'run_win.bash', f'{event_name}'])
        else:
            subprocess.run(['bash', 'ini_proc.bash', f'{event_name}'])
            initial_model_dir = f'{self.stage_initial_model:03d}'
            shutil.copy(f"../TOMO/{initial_model_dir}/MEASURE/adjoints/{event_name}/MEASUREMENT.WINDOWS", "../measure_adj")
            os.chdir(self.measure_adj_dir)
            subprocess.run(['bash', 'run_adj.bash', f'{event_name}'])