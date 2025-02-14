from tools.job_utils import remove_file, remove_files_with_pattern, make_symlink, move_files
from tools.matrix_utils import get_param_from_specfem_file, read_bin, kernel_pad_and_output
from tools.global_params import GLOBAL_PARAMS
from mpi4py import MPI
from pathlib import Path
import os
import sys
import logging
import subprocess



class PostProcessing:
    def __init__(self, current_model_num, config):
        self.config              = config
        self.base_dir            = GLOBAL_PARAMS['base_dir']
        self.mpirun_path         = GLOBAL_PARAMS['mpirun_path']
        self.specfem_dir         = os.path.join(self.base_dir, 'specfem3d')
        self.current_model_num   = current_model_num
        self.specfem_par_file    = os.path.join(self.specfem_dir, 'DATA', 'Par_file')
        self.model_generate_file = os.path.join(self.specfem_dir, 'OUTPUT_FILES', 'output_generate_databases.txt')
        self.pbs_nodefile        = os.path.join(self.base_dir, 'adjointflows', 'nodefile')
        
        self.kernel_list         = config.get('kernel.type.list')
        self.dtype               = config.get('kernel.type.dtype')
        self.ismooth             = config.get('kernel.smoothing.ISMOOTH')
        self.sigma_h             = config.get('kernel.smoothing.SIGMA_H')
        self.sigma_v             = config.get('kernel.smoothing.SIGMA_V')
        self.ivtkout             = config.get('kernel.visualization.IVTKOUT')
        
        self.nproc = get_param_from_specfem_file(file=self.specfem_par_file, param_name='NPROC', param_type=int)
        self.nspec = get_param_from_specfem_file(file=self.model_generate_file, param_name='nspec', param_type=int)
        self.NGLLX = get_param_from_specfem_file(file=self.model_generate_file, param_name='NGLLX', param_type=int)
        self.NGLLY = get_param_from_specfem_file(file=self.model_generate_file, param_name='NGLLY', param_type=int)
        self.NGLLZ = get_param_from_specfem_file(file=self.model_generate_file, param_name='NGLLZ', param_type=int)
        self.gpu_flag = get_param_from_specfem_file(file=self.specfem_par_file, param_name='GPU_MODE', param_type=str)
        
        
    def sum_and_smooth_kernels(self):
        """
        Sum and smooth the kernels
        Note:
            Currently, we just call the sum_smooth_kernel.bash script to do the job.
        """
        logging.info(f"Start smoothing the kernels for the {self.current_model_num:03d} model...") 
        remove_file('INPUT_KERNELS')
        self.make_kernels_list()
        make_symlink(src=os.path.join(self.specfem_dir, 'KERNEL', 'DATABASE'), 
                     dst=os.path.join(self.specfem_dir, 'INPUT_KERNELS'))
        
        Path("OUTPUT_SUM").mkdir(parents=True, exist_ok=True)
        remove_files_with_pattern('OUTPUT_SUM/*')
        
        self.run_precond()
        self.run_sum_kernels()
        
        move_files(src_dir = 'OUTPUT_SUM', 
                   dst_dir = 'KERNEL/SUM', 
                   pattern = '*')
        Path(f'{self.specfem_dir}/KERNEL/SMOOTH').mkdir(parents=True, exist_ok=True)
        
        if self.ismooth:
            self.run_smoothing()
        if self.ivtkout:
            self.combine_kernels()
        
    
    def make_kernels_list(self):
        """
        Make a list of kernels
        replace the linux command `ls KERNEL/DATABASE > kernels_list.txt`
        """
        kernel_dir = f'{self.specfem_dir}/KERNEL/DATABASE'
        kernel_files = [f.name for f in Path(kernel_dir).iterdir()]
        with open('kernels_list.txt', 'w') as f:
            f.write("\n".join(kernel_files))
    
    def run_precond(self):
        """ 
        run xsum_preconditioned_kernels
        """
        nproc = self.nproc
        logging.info(f"Starting precond on {nproc} processors...")

        if nproc == 1:
            subprocess.run(["./bin/xsum_preconditioned_kernels"], check=True)
        else:
            subprocess.run([str(self.mpirun_path), '--hostfile', str(self.pbs_nodefile), '-np' , str(nproc), './bin/xsum_preconditioned_kernels'], check=True)
        
    def run_sum_kernels(self):
        """ 
        run xsum_kernels
        """
        nproc = self.nproc
        logging.info(f"Starting precond on {nproc} processors...")

        if nproc == 1:
            subprocess.run(["./bin/xsum_kernels"], check=True)
        else:
            subprocess.run([str(self.mpirun_path), '--hostfile', str(self.pbs_nodefile), '-np' , str(nproc), './bin/xsum_kernels'], check=True)
    
    def run_smoothing(self):
        """
        run xsmooth_sem
        """
        nproc = self.nproc
        logging.info(f"Starting smoothing on {nproc} processors...")

        if nproc == 1:
            subprocess.run(["./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "alpha_kernel", 
                            "KERNEL/SUM/", "KERNEL/SMOOTH/", f"{self.gpu_flag}"], check=True)
            subprocess.run(["./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "beta_kernel", 
                            "KERNEL/SUM/", "KERNEL/SMOOTH/", f"{self.gpu_flag}"], check=True)
            subprocess.run(["./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "rho_kernel", 
                            "KERNEL/SUM/", "KERNEL/SMOOTH/", f"{self.gpu_flag}"], check=True)
            subprocess.run(["./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "hess_inv_kernel", 
                            "KERNEL/SUM/", "KERNEL/SMOOTH/", f"{self.gpu_flag}"], check=True)
        else:
            subprocess.run([str(self.mpirun_path), '--hostfile', str(self.pbs_nodefile), '-np' , str(nproc),
                            "./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "alpha_kernel", 
                            "KERNEL/SUM/", "KERNEL/SMOOTH/", f"{self.gpu_flag}"], check=True)
            subprocess.run([str(self.mpirun_path), '--hostfile', str(self.pbs_nodefile), '-np' , str(nproc),
                            "./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "beta_kernel", 
                            "KERNEL/SUM/", "KERNEL/SMOOTH/", f"{self.gpu_flag}"], check=True)
            subprocess.run([str(self.mpirun_path), '--hostfile', str(self.pbs_nodefile), '-np' , str(nproc),
                            "./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "rho_kernel", 
                            "KERNEL/SUM/", "KERNEL/SMOOTH/", f"{self.gpu_flag}"], check=True)
            subprocess.run([str(self.mpirun_path), '--hostfile', str(self.pbs_nodefile), '-np' , str(nproc),
                            "./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "hess_inv_kernel", 
                            "KERNEL/SUM/", "KERNEL/SMOOTH/", f"{self.gpu_flag}"], check=True)
    
    def combine_kernels(self):
        """
        Combine the kernels using xcombine_vol_data_vtk
        """
        logging.info(f"Starting combining the smoothed kernels into vtk")
        nslice = int(self.nproc) - 1
        
        if self.ismooth:
            subprocess.run(["./bin/xcombine_vol_data_vtk", "0", f"{nslice}", "alpha_kernel_smooth",
                            "KERNEL/SMOOTH/", "KERNEL/VTK/"], check=True)
            subprocess.run(["./bin/xcombine_vol_data_vtk", "0", f"{nslice}", "beta_kernel_smooth",
                            "KERNEL/SMOOTH/", "KERNEL/VTK/"], check=True)
            subprocess.run(["./bin/xcombine_vol_data_vtk", "0", f"{nslice}", "rho_kernel_smooth",
                            "KERNEL/SMOOTH/", "KERNEL/VTK/"], check=True)
        else:
            subprocess.run(["./bin/xcombine_vol_data_vtk", "0", f"{nslice}", "alpha_kernel",
                            "KERNEL/SMOOTH/", "KERNEL/VTK/"], check=True)
            subprocess.run(["./bin/xcombine_vol_data_vtk", "0", f"{nslice}", "beta_kernel",
                            "KERNEL/SMOOTH/", "KERNEL/VTK/"], check=True)
            subprocess.run(["./bin/xcombine_vol_data_vtk", "0", f"{nslice}", "rho_kernel",
                            "KERNEL/SMOOTH/", "KERNEL/VTK/"], check=True)