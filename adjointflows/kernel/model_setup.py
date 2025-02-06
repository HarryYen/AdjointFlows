from tools.job_utils import remove_file, clean_symlink_target, wait_for_launching, check_path_is_correct
from tools.parameter_tool import get_par_from_specfem_parfile
from tools import GLOBAL_PARAMS

from mpi4py import MPI
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
        self.mpirun_path       = GLOBAL_PARAMS['mpirun_path']
        self.current_model_num = current_model_num
        self.specfem_dir       = os.path.join(self.base_dir, 'specfem3d')
        self.databases_mpi_dir = os.path.join(self.specfem_dir, 'DATABASES_MPI')
        self.model_ready_file  = os.path.join(self.specfem_dir, 'OUTPUT_FILES', 'model_database_ready') 
        self.pbs_nodefile      = os.path.join(self.base_dir, 'adjointflows', 'nodefile')
            
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
        nproc = int(get_par_from_specfem_parfile(self.specfem_dir, 'NPROC'))
        logging.info(f"Starting model meshing and database generation for model {self.current_model_num:03d}...")
        self.run_mesher(nproc)
        subprocess.run(['./utils/change_model_type.pl', '-t'], check=True)
        self.run_generate_databases(nproc)
    
    def model_only_generation(self):
        """
        ONLY do model generation
        1. generation
        """
        nproc = int(get_par_from_specfem_parfile(self.specfem_dir, 'NPROC'))
        logging.info(f"Starting model database generation for model {self.current_model_num:03d}...")
        subprocess.run(['./utils/change_model_type.pl', '-t'], check=True)
        self.run_generate_databases(nproc)

    
    def run_mesher(self, nproc):
        """ 
        run xmeshfem3D        
        """
        print(f"Starting MPI mesher on {nproc} processors...")

        if nproc == 1:
            subprocess.run(["./bin/xmeshfem3D"], check=True)
        else:
            subprocess.run([str(self.mpirun_path), '--hostfile', str(self.pbs_nodefile), '-np' , str(nproc), './bin/xmeshfem3D'], check=True)
        
        logging.info("Done meshing")

    def run_generate_databases(self, nproc):
        """ 
        run xgenerate_databases 
        """
        print(f"Starting MPI database generation on {nproc} processors...")

        if nproc == 1:
            subprocess.run(['./bin/xgenerate_databases'], check=True)
        else:
            subprocess.run([str(self.mpirun_path), '--hostfile', str(self.pbs_nodefile), '-np' , str(nproc), './bin/xgenerate_databases'], check=True)
        
        logging.info("Done database generating")