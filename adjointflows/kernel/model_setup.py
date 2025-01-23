from tools.job_utils import remove_file, clean_symlink_target, wait_for_launching, check_path_is_correct
from tools import GLOBAL_PARAMS
import subprocess
import logging
import os

class ModelGenerator:
    
    def __init__(self, current_model_num):
        """
        Args:
            base_dir (str): The work directorty for model generation (it shouold be specfem3d/)
            current_model_num (int): The current model number
        """
        self.base_dir          = GLOBAL_PARAMS['base_dir']
        self.current_model_num = current_model_num
        self.specfem_dir       = os.path.join(self.base_dir, 'specfem3d')
        self.databases_mpi_dir = os.path.join(self.specfem_dir, 'DATABASES_MPI')
        self.model_ready_file  = os.path.join(self.specfem_dir, 'OUTPUT_FILES', 'model_database_ready') 
    
    def model_setup(self, mesh_flag=True):
        """
        Setup the model mesh and model generation
        """
        if not check_path_is_correct(self.specfem_dir):
            error_message = f"STOP: the current directory is not {self.specfem_dir}!"
            logging.error(error_message)
            raise ValueError(error_message)
        
        remove_file(self.model_ready_file)
        if mesh_flag:
            self.model_mesh_generation()
        else:
            self.model_only_generation()
            
            
    def model_mesh_generation(self):
        """
        Do the WHOLE processes of model setup
        1. mesh
        2. generating
        """
        clean_symlink_target(self.databases_mpi_dir)
        subprocess.run(['qsub', 'model_setup.bash'])
        wait_for_launching(check_file = 'OUTPUT_FILES/model_database_ready', 
                           message = "model mesh and databases established\n")
    
    def model_only_generation(self):
        """
        ONLY do model generation
        1. generation
        """
        subprocess.run(['qsub', 'model_only_generating.bash'])

        wait_for_launching(check_file = 'OUTPUT_FILES/model_database_ready', 
                           message = "model databases have been generated\n")

