from tools.job_utils import remove_file, clean_symlink_target, wait_for_launching, check_path_is_correct
from tools.matrix_utils import get_param_from_specfem_file
from tools import GLOBAL_PARAMS

# from mpi4py import MPI
import subprocess
import logging
import os
import sys

class ModelGenerator:
    
    def __init__(self):

        self.base_dir          = GLOBAL_PARAMS['base_dir']
        self.mpirun_path       = GLOBAL_PARAMS['mpirun_path']
        self.specfem_dir       = os.path.join(self.base_dir, 'specfem3d')
        self.databases_mpi_dir = os.path.join(self.specfem_dir, 'DATABASES_MPI')
        self.pbs_nodefile      = os.path.join(self.base_dir, 'adjointflows', 'nodefile')
        self.specfem_par_file  = os.path.join(self.specfem_dir, 'DATA', 'Par_file')
        
        self.debug_logger      = logging.getLogger("debug_logger")
        
    def model_setup(self, mesh_flag):
        """
        Setup the model mesh and model generation
        """
        if not check_path_is_correct(self.specfem_dir):
            error_message = f"STOP: the current directory is not {self.specfem_dir}!"
            self.debug_logger.error(error_message)
            raise ValueError(error_message)
        
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
        nproc = get_param_from_specfem_file(file=self.specfem_par_file, param_name='NPROC', param_type=int)
        self.debug_logger.info(f"Starting model meshing and database generation for model...")
        self.run_mesher(nproc)
        subprocess.run(['./utils/change_model_type.pl', '-t'], check=True)
        self.run_generate_databases(nproc)
    
    def model_only_generation(self):
        """
        ONLY do model generation
        1. generation
        """
        nproc = get_param_from_specfem_file(file=self.specfem_par_file, param_name='NPROC', param_type=int)
        self.debug_logger.info(f"Starting model database generation for model...")
        subprocess.run(['./utils/change_model_type.pl', '-g'], check=True)
        self.run_generate_databases(nproc)

    
    def run_mesher(self, nproc):
        """ 
        run xmeshfem3D        
        """
        self.debug_logger.info(f"Starting MPI mesher on {nproc} processors...")
        if nproc == 1:
            subprocess.run(["./bin/xmeshfem3D"], check=True)
        else:
            subprocess.run([str(self.mpirun_path), '--hostfile', str(self.pbs_nodefile), '-np' , str(nproc), './bin/xmeshfem3D'], 
                           check=True, env=os.environ)
        self.debug_logger.info("Done meshing")
        
    def run_generate_databases(self, nproc):
        """ 
        run xgenerate_databases 
        """
        print(f"Starting MPI database generation on {nproc} processors...")

        if nproc == 1:
            subprocess.run(['./bin/xgenerate_databases'], check=True)
        else:
            subprocess.run([str(self.mpirun_path), '--hostfile', str(self.pbs_nodefile), '-np' , str(nproc), './bin/xgenerate_databases'], 
                           check=True, env=os.environ)
        
        self.debug_logger.info("Done database generating")