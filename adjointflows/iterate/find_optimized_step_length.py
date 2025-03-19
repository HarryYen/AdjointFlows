from tools import GLOBAL_PARAMS
from tools.job_utils import remove_path, check_if_directory_not_empty, remove_file, remove_files_with_pattern, move_files, wait_for_launching, clean_and_initialize_directories, check_path_is_correct
from tools.matrix_utils import get_param_from_specfem_file
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

class StepLengthOptimizer:
    
    def __init__(self, current_model_num, config):
        """
        Args:
            config (dict): The configuration dictionary
            current_model_num (int): The current model number
        """
        self.step_index        = 0
        self.optimized_step_length = 0.
        
        
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
        
        self.evlst               = os.path.join(self.base_dir, 'DATA', 'evlst', config.get('data.list.evchk'))
        self.stlst               = os.path.join(self.base_dir, 'DATA', 'stlst', config.get('data.list.stlst'))
        self.specfem_par_file    = os.path.join(self.specfem_dir, 'DATA', 'Par_file')
        self.adjflows_dir        = os.path.join(self.base_dir, 'adjointflows')
        self.pbs_nodefile        = os.path.join(self.adjflows_dir, 'nodefile')

        
        self.step_interval       = config.get('line_search.step_interval')
        self.step_max            = config.get('line_search.step_max')
        self.max_fail            = config.get('inversion.max_fail')

        
        self.nproc               = get_param_from_specfem_file(file=self.specfem_par_file, param_name='NPROC', param_type=int)

        self.debug_logger  = logging.getLogger("debug_logger")
        self.result_logger = logging.getLogger("result_logger")
            
    # --------------------------------------------------
    # ---------------- For Controlling -----------------
    # --------------------------------------------------
    def run_line_search(self):
        """
        Run the line search algorithm to find the optimized step length
        This is for SD method
        """
        
        model_generator_line_search = ModelGenerator()
        
        self.result_logger.info(f"Starting line search for model {self.current_model_num:03d}...")
        self.give_current_best_step_length(step_length_tmp=0.)
        for step_length in np.arange(self.step_interval, self.step_max, self.step_interval):
            
            self.result_logger.info(f"LINE SEARCH: Start step length {step_length}")
            self.increase_step_index()        
            self.setup_directory()
            self.make_symbolic_links()
            self.update_model(step_fac=step_length, lbfgs_flag=False)
            os.chdir(self.specfem_dir)
            model_generator_line_search.model_setup(mesh_flag=False)
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
        
    def setup_directory(self):
        """
        Create a series of directories for inversion.
        it will call clean_and_initialize_directories to remove all files in the directories
        if the clear_directories is True.
        """
        
        dirs = [
            f"{self.line_search_dir}/DATABASES_MPI",
            f"{self.line_search_dir}/SYN{self.step_index}",
            f"{self.line_search_dir}/MEASURE{self.step_index}",
    ]
    
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)
        
        clean_and_initialize_directories(dirs[1:])
    
    def make_symbolic_links(self):
        """
        We need to create the symbolic links for the SEM simulation.
        Here, it will link 
        """

        link_directories = [
            f'{self.specfem_dir}/DATABASES_MPI', 
            f'{self.base_dir}/SYN',  
            f'{self.base_dir}/measure_adj/PACK'
        ]
        target_directories = [
            f"{self.line_search_dir}/DATABASES_MPI",
            f"{self.line_search_dir}/SYN{self.step_index}",
            f"{self.line_search_dir}/MEASURE{self.step_index}",
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
        
        kernel_ready_file = os.path.join(self.specfem_dir, 'kernel_databases_ready')
        remove_file(kernel_ready_file)
        

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
            event_name = event_info.iloc[0]
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
        env = os.environ.copy()
        nproc = self.nproc
        self.result_logger.info(f"Starting xspecfem3D on {nproc} processors...")

        if nproc == 1:
            subprocess.run(["./bin/xspecfem3D", ], env=env, check=True)
        else:
            subprocess.run([str(self.mpirun_path), '--hostfile', str(self.pbs_nodefile), '-np' , str(nproc), './bin/xspecfem3D'], env=env, check=True)
    
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
        Run measure_adj by directly using the MEASUREMENT.WINDOWS calculated by this model
        """
        os.chdir(self.flexwin_dir)
        
        subprocess.run(['bash', 'ini_proc.bash', f'{event_name}'])
        initial_model_dir = f'm{self.current_model_num:03d}'
        shutil.copy(f"../TOMO/{initial_model_dir}/MEASURE/adjoints/{event_name}/MEASUREMENT.WINDOWS", "../measure_adj")
        os.chdir(self.measure_adj_dir)
        subprocess.run(['bash', 'run_adj.bash', f'{event_name}'])
        
        
    def misfit_calculation(self, step_index):
        """
        Calculate the misfit for the given model number
        Return:
            misfit (float): The misfit value
        """
        if step_index == 0:
            measure_dir = f'{self.current_tomo_dir}/MEASURE/adjoints'
        else:
            measure_dir = f'{self.line_search_dir}/MEASURE{step_index}'
    
        evt_df = pd.read_csv(self.evlst, header=None, sep=r'\s+')
        chi_df = pd.DataFrame()
        for evt in evt_df[0]:
            adjoints_dir = f'{measure_dir}/{evt}'
            chi_file = f'{adjoints_dir}/window_chi'
            tmp_df = pd.read_csv(chi_file, header=None, sep=r'\s+')
            chi_df = pd.concat([chi_df, tmp_df])
        
        misfit = round(chi_df[28].sum() / len(chi_df), 5)
        self.result_logger.info(f'Misfit: {misfit}')
        return misfit
    
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
            command = f'{self.py_mpirun_path} --hostfile {self.pbs_nodefile} -np {nproc} python {script_dir} {step_fac} {int(lbfgs_flag)} {int(line_search_flag)}'
            self.debug_logger.info(f"Command: {command}")
            subprocess.run(command, shell=True, check=True, env=env)
        
    # -----------------------------------------------
    # ----------------- For L-BFGS ------------------
    # -----------------------------------------------
    
    def quadratic_interpolation(self, phi_o, phi0, alpha, phi_grad):
        return - ( phi_grad * alpha ** 2 ) / ( 2 * ( phi0 - phi_o - alpha * phi_grad ) )

    def cubic_interpolation(self, phi_o, phi0, phi1, alpha0, alpha1, phi_grad):
        denomimator = (alpha0 ** 2) * (alpha1 ** 2) * (alpha1 - alpha0)
        a = ( (alpha0 ** 2) * (phi1 - phi_o - phi_grad * alpha1 ) - ( alpha1 ** 2 ) * (phi0 - phi_o - phi_grad * alpha0) ) / denomimator
        b = (-(alpha0 ** 3) * (phi1 - phi_o - phi_grad * alpha1 ) + ( alpha1 ** 3 ) * (phi0 - phi_o - phi_grad * alpha0) ) / denomimator
        return -b + np.sqrt(b ** 2 - 3 * a * phi_grad) / (3 * a)