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

        
        self.step_interval       = config.get('line_search.step_interval')
        self.step_beg            = config.get('line_search.step_beg')
        self.step_end            = config.get('line_search.step_end')
        self.max_fail            = config.get('inversion.max_fail')
        self.shrink_factor       = config.get('inversion.backtracking_shrink')
        
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
        for step_length in np.arange(self.step_beg, self.step_end, self.step_interval):
            
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
        if self.source_type == 'force':
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
        average_misfit = round(total_misfit / win_num, 5)
        return total_misfit
    
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
