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
        self.adjflows_dir        = os.path.join(self.base_dir, 'adjointflows')
        self.current_model_num   = current_model_num
        self.specfem_par_file    = os.path.join(self.specfem_dir, 'DATA', 'Par_file')
        self.model_generate_file = os.path.join(self.specfem_dir, 'OUTPUT_FILES', 'output_generate_databases.txt')
        self.gradient_file       = os.path.join(self.adjflows_dir, 'output_inner_product.txt')
        self.pbs_nodefile        = os.path.join(self.adjflows_dir, 'nodefile')
        
        self.kernel_list         = config.get('kernel.type.list')
        self.dtype               = config.get('kernel.type.dtype')
        self.n_store_lbfgs       = config.get('inversion.n_store_lbfgs')
        self.stage_initial_model = config.get('setup.stage.stage_initial_model')
        
        self.nproc = get_param_from_specfem_file(file=self.specfem_par_file, param_name='NPROC', param_type=int)
        self.nspec = get_param_from_specfem_file(file=self.model_generate_file, param_name='nspec', param_type=int)
        self.nglob = get_param_from_specfem_file(file=self.model_generate_file, param_name='NGLOB_global_min', param_type=int)
        self.NGLLX = get_param_from_specfem_file(file=self.model_generate_file, param_name='NGLLX', param_type=int)
        self.NGLLY = get_param_from_specfem_file(file=self.model_generate_file, param_name='NGLLY', param_type=int)
        self.NGLLZ = get_param_from_specfem_file(file=self.model_generate_file, param_name='NGLLZ', param_type=int)

        self.debug_logger  = logging.getLogger("debug_logger")
        self.result_logger = logging.getLogger("result_logger")
    
    def save_params_json(self):
        """
        We save the parameters in the json file to let the other scripts read them.
        """
        params = {
            'base_dir': self.base_dir,
            'specfem_dir': self.specfem_dir,
            'kernel_list': self.kernel_list,
            'current_model_num': self.current_model_num,
            'n_store_lbfgs': self.n_store_lbfgs,
            'stage_initial_model': self.stage_initial_model,
            'dtype': self.dtype,
            'nproc': self.nproc,
            'nspec': self.nspec,
            'nglob': self.nglob,
            'NGLLX': self.NGLLX,
            'NGLLY': self.NGLLY,
            'NGLLZ': self.NGLLZ    
        }
        with open(f'{self.adjflows_dir}/params.json', 'w') as f:
            json.dump(params, f)
    
    
    def hess_times_kernel(self):
        """
        Precodition the kernel through multiplying it with the Hessian
        """
        env = os.environ.copy()
        nproc = self.nproc
        self.result_logger.info(f"Starting preconditioning on {nproc} processors...")
        
        script_dir = "iterate/hess_times_kernel.py"
        
        
        if nproc == 1:
            command = f"python {script_dir}"
            subprocess.run(command, shell=True, check=True)
        else:
            command = f'{self.mpirun_path} --hostfile {self.pbs_nodefile} -np {nproc} python {script_dir}'
            subprocess.run(command, shell=True, check=True, env=env)

        self.result_logger.info("Done preconditioning!")
        
    def calculate_direction_sd(self):
        """
        Calculate the direction using the steepest descent method
        """
        env = os.environ.copy()
        nproc = self.nproc
        self.result_logger.info(f"Starting calculating direction through SD on {nproc} processors...")
        
        script_dir = "iterate/calculate_direction_sd.py"
        
        
        if nproc == 1:
            command = f"python {script_dir}"
            subprocess.run(command, shell=True, check=True)
        else:
            command = f'{self.mpirun_path} --hostfile {self.pbs_nodefile} -np {nproc} python {script_dir}'
            subprocess.run(command, shell=True, check=True, env=env)

        self.result_logger.info("Done Steepest direction method!")
    

    def calculate_direction_lbfgs(self):
        """
        Calculate the direction using L-BFGS method
        """
        env = os.environ.copy()
        nproc = self.nproc
        self.result_logger.info(f"Starting calculating direction through LBFGS on {nproc} processors...")
        
        script_dir = "iterate/calculate_direction_lbfgs.py"
        
        
        if nproc == 1:
            command = f"python {script_dir}"
            subprocess.run(command, shell=True, check=True)
        else:
            command = f'{self.mpirun_path} --hostfile {self.pbs_nodefile} -np {nproc} python {script_dir}'
            subprocess.run(command, shell=True, check=True, env=env)

        self.result_logger.info("Done L-BFGS method!")
    
    def update_model(self, step_fac, lbfgs_flag=True):
        """
        Update the model from old_model to new_model from TOMO/KERNEL/UPDATE
        step_fac (float): The step factor for the update (e.g. 0.03 for +-3% update)
        lbfgs_flag (bool): True for L-BFGS method, False for SD
        """
        env = os.environ.copy()
        nproc = self.nproc
        self.result_logger.info(f"Starting updating model on {nproc} processors...")
        os.chdir(self.adjflows_dir)
        
        script_dir = "iterate/model_update.py"
        line_search_flag = False
        
        if nproc == 1:
            command = f"python {script_dir} {step_fac} {int(lbfgs_flag)} {int(line_search_flag)}"
            subprocess.run(command, shell=True, check=True)
        else:
            command = f'{self.mpirun_path} --hostfile {self.pbs_nodefile} -np {nproc} python {script_dir} {step_fac} {int(lbfgs_flag)} {int(line_search_flag)}'
            subprocess.run(command, shell=True, check=True, env=env)

        self.result_logger.info("Done Updating Model!")
    
    def get_gradient_info(self, param_name):
        """
        get the gradient information from iterate/output_inner_product.txt
        Args:
            param_name (str): the parameter name (e.g. 'g_dot_p')
        """
        with open(self.gradient_file) as f:
            for line in f:
                if param_name in line:
                    return float(line.split()[1])
        return None
    
    def save_polynomial_to_json(self, misfit_list, step_list, g_dot_p):
        """
        Save the polynomial to the json file
        Args:
            misfit_list (list): The list of misfit values
            step_list (list): The list of step values
            g_dot_p (float): The gradient dot product
        """
        params = {
            'misfit_list': misfit_list,
            'step_list': step_list,
            'g_dot_p': [g_dot_p]
        }
        with open(f'{self.adjflows_dir}/polynomial.json', 'w') as f:
            json.dump(params, f, indent=4)