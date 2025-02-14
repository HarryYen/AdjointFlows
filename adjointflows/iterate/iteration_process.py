from tools.job_utils import remove_file, remove_files_with_pattern, make_symlink, move_files
from tools.matrix_utils import get_param_from_specfem_file, read_bin, kernel_pad_and_output
from tools.global_params import GLOBAL_PARAMS
from pathlib import Path
import os
import sys
import logging
import subprocess
import json

class IterationProcess:
    def __init__(self, current_model_num, config):
        self.config              = config
        self.base_dir            = GLOBAL_PARAMS['base_dir']
        self.mpirun_path         = GLOBAL_PARAMS['mpirun_python_path']
        self.specfem_dir         = os.path.join(self.base_dir, 'specfem3d')
        self.current_model_num   = current_model_num
        self.specfem_par_file    = os.path.join(self.specfem_dir, 'DATA', 'Par_file')
        self.model_generate_file = os.path.join(self.specfem_dir, 'OUTPUT_FILES', 'output_generate_databases.txt')
        self.pbs_nodefile        = os.path.join(self.base_dir, 'adjointflows', 'nodefile')
        
        self.kernel_list         = config.get('kernel.type.list')
        self.dtype               = config.get('kernel.type.dtype')
        
        self.nproc = get_param_from_specfem_file(file=self.specfem_par_file, param_name='NPROC', param_type=int)
        self.nspec = get_param_from_specfem_file(file=self.model_generate_file, param_name='nspec', param_type=int)
        self.NGLLX = get_param_from_specfem_file(file=self.model_generate_file, param_name='NGLLX', param_type=int)
        self.NGLLY = get_param_from_specfem_file(file=self.model_generate_file, param_name='NGLLY', param_type=int)
        self.NGLLZ = get_param_from_specfem_file(file=self.model_generate_file, param_name='NGLLZ', param_type=int)

    def save_params_json(self):
        """
        We save the parameters in the json file to let the other scripts read them.
        """
        params = {
            'base_dir': self.base_dir,
            'specfem_dir': self.specfem_dir,
            'kernel_list': self.kernel_list,
            'current_model_num': self.current_model_num,
            'dtype': self.dtype,
            'nproc': self.nproc,
            'nspec': self.nspec,
            'NGLLX': self.NGLLX,
            'NGLLY': self.NGLLY,
            'NGLLZ': self.NGLLZ    
        }
        with open(f'{self.base_dir}/adjointflows/params.json', 'w') as f:
            json.dump(params, f)
    
    
    def hess_times_kernel(self):
        """
        Precodition the kernel through multiplying it with the Hessian
        """
        env = os.environ.copy()
        nproc = self.nproc
        logging.info(f"Starting preconditioning on {nproc} processors...")
        
        # script_dir = f"{self.base_dir}/adjointflows/iterate/hess_times_kernel.py"
        script_dir = "iterate/hess_times_kernel.py"
        
        
        if nproc == 1:
            subprocess.run(["python", f"{script_dir}"], check=True)
        else:
            subprocess.run(f'{self.mpirun_path} --hostfile {self.pbs_nodefile} -np {nproc} python {script_dir}', shell=True, check=True, env=env)
            # subprocess.run(f'python {script_dir}', shell=True, check=True, env=env)

        logging.info("Done preconditioning!")