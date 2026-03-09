from tools.job_utils import remove_file, remove_files_with_pattern, make_symlink, move_files
from tools.matrix_utils import get_param_from_specfem_file, read_bin, kernel_pad_and_output, get_data_type
from tools.global_params import GLOBAL_PARAMS
from tools.dataset_loader import get_by_path
from mpi4py import MPI
from pathlib import Path
import os
import sys
import logging
import subprocess
import numpy as np
import shutil



class PostProcessing:
    def __init__(self, current_model_num, config, ismooth=True, sigma_h=15000, sigma_v=10000):
        self.config              = config
        self.base_dir            = GLOBAL_PARAMS['base_dir']
        self.adjflows_dir        = os.path.join(self.base_dir, 'adjointflows')
        self.mpirun_path         = GLOBAL_PARAMS['mpirun_path']
        self.specfem_dir         = os.path.join(self.base_dir, 'specfem3d')
        self.current_model_num   = current_model_num
        self.tomo_dir            = os.path.join(self.base_dir, 'TOMO', f'm{self.current_model_num:03d}')
        self.specfem_par_file    = os.path.join(self.specfem_dir, 'DATA', 'Par_file')
        self.model_generate_file = os.path.join(self.specfem_dir, 'OUTPUT_FILES', 'output_generate_databases.txt')
        self.pbs_nodefile        = os.path.join(self.base_dir, 'adjointflows', 'nodefile')
        
        self.kernel_list         = config.get('kernel.type.list')
        self.dtype               = config.get('kernel.type.dtype')
        self.ismooth             = ismooth
        self.sigma_h             = sigma_h
        self.sigma_v             = sigma_v
        self.ivtkout             = config.get('kernel.visualization.IVTKOUT')
        # self.evlst               = os.path.join(self.base_dir, 'DATA', 'evlst', config.get('data.list.evlst'))
        
        self.nproc = get_param_from_specfem_file(file=self.specfem_par_file, param_name='NPROC', param_type=int)
        self.nspec = get_param_from_specfem_file(file=self.model_generate_file, param_name='nspec', param_type=int)
        self.NGLLX = get_param_from_specfem_file(file=self.model_generate_file, param_name='NGLLX', param_type=int)
        self.NGLLY = get_param_from_specfem_file(file=self.model_generate_file, param_name='NGLLY', param_type=int)
        self.NGLLZ = get_param_from_specfem_file(file=self.model_generate_file, param_name='NGLLZ', param_type=int)
        self.gpu_flag = get_param_from_specfem_file(file=self.specfem_par_file, param_name='GPU_MODE', param_type=str)
        
        self.debug_logger  = logging.getLogger("debug_logger")
        self.result_logger = logging.getLogger("result_logger")
    
    
    def sum_and_smooth_kernels(self, dataset_name, evlst, precond_flag=False):
        """
        Sum and smooth the kernels
        Note:
            Currently, we just call the sum_smooth_kernel.bash script to do the job.
        """
        self.result_logger.info(f"Start smoothing the kernels for the {self.current_model_num:03d} model...") 
        remove_file('INPUT_KERNELS')
        self.make_kernels_list(dataset_name, evlst)
        make_symlink(src=os.path.join(self.tomo_dir, f'KERNEL_{dataset_name}', 'DATABASE'), 
                     dst=os.path.join(self.specfem_dir, 'INPUT_KERNELS'))
        
        Path("OUTPUT_SUM").mkdir(parents=True, exist_ok=True)
        remove_files_with_pattern('OUTPUT_SUM/*')
        
        if precond_flag:
            self.run_precond()
        
        self.run_sum_kernels()
        
        move_files(src_dir = 'OUTPUT_SUM', 
                   dst_dir = f'{self.tomo_dir}/KERNEL_{dataset_name}/SUM', 
                   pattern = '*')
        Path(f'{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH').mkdir(parents=True, exist_ok=True)
        
        if self.ismooth:
            self.run_smoothing(dataset_name=dataset_name, precond_flag=precond_flag)
        if self.ivtkout:
            self.combine_kernels()

    def compute_gradient_max(self, dataset_name, use_smooth=None, source_subdir=None):
        """
        Compute the maximum absolute gradient value for a dataset.
        """
        smooth_dir = Path(self.tomo_dir) / f"KERNEL_{dataset_name}" / "SMOOTH"
        sum_dir = Path(self.tomo_dir) / f"KERNEL_{dataset_name}" / "SUM"
        precond_dir = Path(self.tomo_dir) / f"KERNEL_{dataset_name}" / "PRECOND"
        kernel_dir = None
        suffix = None

        if source_subdir == "PRECOND":
            kernel_dir = precond_dir
            suffix = "_kernel_smooth.bin"
        elif source_subdir == "SUM":
            kernel_dir = sum_dir
            suffix = "_kernel.bin"
        elif source_subdir == "SMOOTH":
            kernel_dir = smooth_dir
            suffix = "_kernel_smooth.bin"
        elif use_smooth is True:
            kernel_dir = smooth_dir
            suffix = "_kernel_smooth.bin"
        elif use_smooth is False:
            kernel_dir = sum_dir
            suffix = "_kernel.bin"
        else:
            if smooth_dir.is_dir() and list(smooth_dir.glob("proc*_*_kernel_smooth.bin")):
                kernel_dir = smooth_dir
                suffix = "_kernel_smooth.bin"
            elif sum_dir.is_dir() and list(sum_dir.glob("proc*_*_kernel.bin")):
                kernel_dir = sum_dir
                suffix = "_kernel.bin"

        if kernel_dir is None or not kernel_dir.is_dir():
            self.result_logger.warning(f"No kernel directory found for dataset {dataset_name}.")
            return 0.0

        dtype = get_data_type(self.dtype)
        max_abs = 0.0
        for kernel_name in self.kernel_list:
            pattern = f"proc*_{kernel_name}{suffix}"
            kernel_files = list(kernel_dir.glob(pattern))
            if not kernel_files:
                self.debug_logger.warning(f"No kernel files for {kernel_name} in {kernel_dir}.")
                continue
            for kernel_file in kernel_files:
                try:
                    data, _padding = read_bin(
                        file_name=str(kernel_file),
                        NGLLX=self.NGLLX,
                        NGLLY=self.NGLLY,
                        NGLLZ=self.NGLLZ,
                        NSPEC=self.nspec,
                        dtype=dtype,
                    )
                except Exception as exc:
                    self.debug_logger.warning(f"Failed to read {kernel_file}: {exc}")
                    continue
                max_abs = max(max_abs, float(np.max(np.abs(data))))
        return max_abs

    def prepare_precond(self, dataset_name, use_smooth, precond_flag):
        """
        Prepare PRECOND kernels for a dataset.
        """
        kernel_subdir = "SMOOTH" if use_smooth else "SUM"
        kernel_dir = Path(self.tomo_dir) / f"KERNEL_{dataset_name}" / kernel_subdir
        precond_dir = Path(self.tomo_dir) / f"KERNEL_{dataset_name}" / "PRECOND"
        precond_dir.mkdir(parents=True, exist_ok=True)

        suffix_in = "_kernel_smooth.bin" if use_smooth else "_kernel.bin"
        suffix_out = "_kernel_smooth.bin"

        if not precond_flag:
            for kernel_name in self.kernel_list:
                for kernel_file in kernel_dir.glob(f"proc*_{kernel_name}{suffix_in}"):
                    output_name = kernel_file.name.replace(suffix_in, suffix_out)
                    shutil.copy2(kernel_file, precond_dir / output_name)
            return

        env = os.environ.copy()
        script_path = os.path.join(self.adjflows_dir, "iterate", "hess_times_kernel.py")
        args = [
            "python",
            script_path,
            "--kernel-dir",
            str(kernel_dir),
            "--output-dir",
            str(precond_dir),
            "--use-smooth",
            "1" if use_smooth else "0",
        ]
        if self.nproc == 1:
            subprocess.run(args, check=True, env=env, cwd=self.adjflows_dir)
        else:
            command = [str(self.mpirun_path), "-np", str(self.nproc)] + args
            subprocess.run(command, check=True, env=env, cwd=self.adjflows_dir)

    def accumulate_normalized_gradients(self, dataset_name, norm, use_smooth, output_dir, source_subdir=None, weight=1.0):
        """
        Accumulate normalized gradients into a combined directory.
        """
        kernel_subdir = source_subdir or ("SMOOTH" if use_smooth else "SUM")
        kernel_dir = Path(self.tomo_dir) / f"KERNEL_{dataset_name}" / kernel_subdir
        if not kernel_dir.is_dir():
            self.result_logger.warning(f"Kernel directory not found for dataset {dataset_name}: {kernel_dir}")
            return

        norm_value = float(norm) if norm else 0.0
        if norm_value <= 0.0:
            self.result_logger.warning(f"Invalid norm for dataset {dataset_name}; use 1.0.")
            norm_value = 1.0
        weight_value = float(weight) if weight is not None else 1.0

        if kernel_subdir == "PRECOND":
            suffix_in = "_kernel_smooth.bin"
        else:
            suffix_in = "_kernel_smooth.bin" if use_smooth else "_kernel.bin"
        suffix_out = "_kernel_smooth.bin"
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        dtype = get_data_type(self.dtype)
        for kernel_name in self.kernel_list:
            pattern = f"proc*_{kernel_name}{suffix_in}"
            kernel_files = list(kernel_dir.glob(pattern))
            if not kernel_files:
                self.debug_logger.warning(f"No kernel files for {kernel_name} in {kernel_dir}.")
                continue
            for kernel_file in kernel_files:
                try:
                    data, padding = read_bin(
                        file_name=str(kernel_file),
                        NGLLX=self.NGLLX,
                        NGLLY=self.NGLLY,
                        NGLLZ=self.NGLLZ,
                        NSPEC=self.nspec,
                        dtype=dtype,
                    )
                except Exception as exc:
                    self.debug_logger.warning(f"Failed to read {kernel_file}: {exc}")
                    continue

                data = (data / norm_value) * weight_value
                output_name = kernel_file.name.replace(suffix_in, suffix_out)
                output_file = out_dir / output_name
                if output_file.exists():
                    try:
                        existing, _padding = read_bin(
                            file_name=str(output_file),
                            NGLLX=self.NGLLX,
                            NGLLY=self.NGLLY,
                            NGLLZ=self.NGLLZ,
                            NSPEC=self.nspec,
                            dtype=dtype,
                        )
                        data = data + existing
                    except Exception as exc:
                        self.debug_logger.warning(f"Failed to read {output_file}: {exc}")
                kernel_pad_and_output(kernel=data, output_file=str(output_file), padding_num=padding)
        
    
    def make_kernels_list(self, dataset_name, evlst):
        """
        Make a list of kernels
        replace the linux command `ls KERNEL/DATABASE > kernels_list.txt`
        """
        kernel_dir = f'{self.tomo_dir}/KERNEL_{dataset_name}/DATABASE'
        event_names = None
        if evlst and os.path.isfile(evlst):
            with open(evlst, 'r') as f:
                event_names = [line.split()[0] for line in f if line.split()]

        if event_names:
            # Filter by evlst and skip events without kernel files.
            kernel_files = []
            for name in event_names:
                event_dir = Path(kernel_dir) / name
                if not event_dir.is_dir():
                    continue
                if not list(event_dir.glob("proc*kernel.bin")):
                    self.result_logger.warning(
                        f"Empty kernel directory for dataset {dataset_name}, event {name}: {event_dir}"
                    )
                    continue
                kernel_files.append(name)
        else:
            kernel_files = [f.name for f in Path(kernel_dir).iterdir() if f.is_dir()]
        with open('kernels_list.txt', 'w') as f:
            f.write("\n".join(kernel_files))
    
    def run_precond(self):
        """ 
        run xsum_preconditioned_kernels
        """
        nproc = self.nproc
        self.result_logger.info(f"Starting precond on {nproc} processors...")

        if nproc == 1:
            subprocess.run(["./bin/xsum_preconditioned_kernels"], check=True, env=os.environ)
        else:
            subprocess.run([str(self.mpirun_path), '-np' , str(nproc), './bin/xsum_preconditioned_kernels'], 
                           check=True, env=os.environ)
        
    def run_sum_kernels(self):
        """ 
        run xsum_kernels
        """
        nproc = self.nproc
        self.result_logger.info(f"Starting summation on {nproc} processors...")

        if nproc == 1:
            subprocess.run(["./bin/xsum_kernels"], check=True, env=os.environ)
        else:
            subprocess.run([str(self.mpirun_path), '-np' , str(nproc), './bin/xsum_kernels'], 
                           check=True, env=os.environ)
    
    def run_smoothing(self, dataset_name, precond_flag=False):
        """
        run xsmooth_sem
        """
        nproc = self.nproc
        self.result_logger.info(f"Starting smoothing on {nproc} processors...")

        if nproc == 1:
            subprocess.run(["./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "alpha_kernel", 
                            f"{self.tomo_dir}/KERNEL_{dataset_name}/SUM/", f"{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH/", f"{self.gpu_flag}"], check=True, env=os.environ)
            subprocess.run(["./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "beta_kernel", 
                            f"{self.tomo_dir}/KERNEL_{dataset_name}/SUM/", f"{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH/", f"{self.gpu_flag}"], check=True, env=os.environ)
            subprocess.run(["./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "rho_kernel", 
                            f"{self.tomo_dir}/KERNEL_{dataset_name}/SUM/", f"{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH/", f"{self.gpu_flag}"], check=True, env=os.environ)
            if precond_flag:
                subprocess.run(["./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "hess_inv_kernel", 
                                f"{self.tomo_dir}/KERNEL_{dataset_name}/SUM/", f"{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH/", f"{self.gpu_flag}"], check=True, env=os.environ)
        else:
            subprocess.run([str(self.mpirun_path), '-np' , str(nproc),
                            "./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "alpha_kernel", 
                            f"{self.tomo_dir}/KERNEL_{dataset_name}/SUM/", f"{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH/", f"{self.gpu_flag}"], check=True, env=os.environ)
            subprocess.run([str(self.mpirun_path), '-np' , str(nproc),
                            "./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "beta_kernel", 
                            f"{self.tomo_dir}/KERNEL_{dataset_name}/SUM/", f"{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH/", f"{self.gpu_flag}"], check=True, env=os.environ)
            subprocess.run([str(self.mpirun_path), '-np' , str(nproc),
                            "./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "rho_kernel", 
                            f"{self.tomo_dir}/KERNEL_{dataset_name}/SUM/", f"{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH/", f"{self.gpu_flag}"], check=True, env=os.environ)
            if precond_flag:
                subprocess.run([str(self.mpirun_path), '-np' , str(nproc),
                                "./bin/xsmooth_sem", f"{self.sigma_h}", f"{self.sigma_v}", "hess_inv_kernel", 
                                f"{self.tomo_dir}/KERNEL_{dataset_name}/SUM/", f"{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH/", f"{self.gpu_flag}"], check=True, env=os.environ)
    
    def combine_kernels(self, dataset_name):
        """
        Combine the kernels using xcombine_vol_data_vtk
        """
        self.debug_logger.info(f"Starting combining the smoothed kernels into vtk")
        nslice = int(self.nproc) - 1
        
        if self.ismooth:
            subprocess.run(["./bin/xcombine_vol_data_vtk", "0", f"{nslice}", "alpha_kernel_smooth",
                            f"{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH/", f"{self.tomo_dir}/KERNEL_{dataset_name}/VTK/", "0"], check=True, env=os.environ)
            subprocess.run(["./bin/xcombine_vol_data_vtk", "0", f"{nslice}", "beta_kernel_smooth",
                            f"{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH/", f"{self.tomo_dir}/KERNEL_{dataset_name}/VTK/", "0"], check=True, env=os.environ)
            subprocess.run(["./bin/xcombine_vol_data_vtk", "0", f"{nslice}", "rho_kernel_smooth",
                            f"{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH/", f"{self.tomo_dir}/KERNEL_{dataset_name}/VTK/", "0"], check=True, env=os.environ)
        else:
            subprocess.run(["./bin/xcombine_vol_data_vtk", "0", f"{nslice}", "alpha_kernel",
                            f"{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH/", f"{self.tomo_dir}/KERNEL_{dataset_name}/VTK/", "0"], check=True, env=os.environ)
            subprocess.run(["./bin/xcombine_vol_data_vtk", "0", f"{nslice}", "beta_kernel",
                            f"{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH/", f"{self.tomo_dir}/KERNEL_{dataset_name}/VTK/", "0"], check=True, env=os.environ)
            subprocess.run(["./bin/xcombine_vol_data_vtk", "0", f"{nslice}", "rho_kernel",
                            f"{self.tomo_dir}/KERNEL_{dataset_name}/SMOOTH/", f"{self.tomo_dir}/KERNEL_{dataset_name}/VTK/", "0"], check=True, env=os.environ)
