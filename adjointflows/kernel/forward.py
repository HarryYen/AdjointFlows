from tools import GLOBAL_PARAMS
from tools.job_utils import check_if_directory_not_empty, remove_file, wait_for_launching, check_path_is_correct
import subprocess
import logging
import os
import sys

class ForwardGenerator:
    
    def __init__(self, current_model_num, config):
        """
        Args:
            base_dir (str): The work directorty for model generation (it shouold be specfem3d/)
            current_model_num (int): The current model number
        """
        self.base_dir          = GLOBAL_PARAMS['base_dir']
        self.current_model_num = current_model_num
        self.specfem_dir       = os.path.join(self.base_dir, 'specfem3d')
        self.databases_mpi_dir = os.path.join(self.specfem_dir, 'DATABASES_MPI')
        
        self.stage_initial_model = config.get('setup.stage.stage_initial_model')
        
    def do_forward(self):
        """
        check if the DATABASES_MPI is not empty, then do the forward simulation
        """
        
        if not check_if_directory_not_empty(self.databases_mpi_dir):
            logging.error(f"STOP: {self.databases_mpi_dir} is empty!")
            sys.exit()
        
        if not check_path_is_correct(self.specfem_dir):
            error_message = f"STOP: the current directory is not {self.specfem_dir}!"
            logging.error(error_message)
            raise ValueError(error_message)
        
        kernel_ready_file = os.path.join(self.specfem_dir, 'kernel_databases_ready')
        remove_file(kernel_ready_file)
        
        self.output_vars_file()
        subprocess.run(['qsub', 'kernel_serial.bash'])
        
        wait_for_launching(check_file = 'kernel_databases_ready', 
                           message = 'databases are ready!\n')
        
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