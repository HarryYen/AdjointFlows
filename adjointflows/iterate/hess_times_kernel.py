import sys
import os
import importlib
from pathlib import Path
import logging
import subprocess
import json

from mpi4py import MPI

    
def main():
    
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    os.environ["MPI_RANK"] = str(rank)

    # --------------------------------------------------------------------------------------------
    # IMPORT TOOLS
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from tools.job_utils import remove_file, remove_files_with_pattern, make_symlink, move_files
    from tools.matrix_utils import get_param_from_specfem_file, read_bin, kernel_pad_and_output, get_data_type
    from tools.global_params import GLOBAL_PARAMS
    import numpy as np
    # --------------------------------------------------------------------------------------------

    if rank == 0:
        with open("params.json", "r") as f:
            params = json.load(f)
    else:
        params = None
    
    params = comm.bcast(params, root=0)

    mrun = params['current_model_num']
    model_num = f"m{mrun:03d}"
    
    kernel_dir = os.path.join(params['base_dir'], "TOMO", model_num, "KERNEL", "SMOOTH")
    output_dir = os.path.join(params['base_dir'], "TOMO", model_num, "KERNEL", "PRECOND")
    for kernel_name in params['kernel_list']:
                kernel_file = os.path.join(kernel_dir, f"proc{rank:06d}_{kernel_name}_kernel_smooth.bin")
                hess_inv_file = os.path.join(kernel_dir, f"proc{rank:06d}_hess_inv_kernel_smooth.bin")
                hess_inv, padding_num = read_bin(file_name=hess_inv_file, 
                                                 NGLLX=params['NGLLX'], 
                                                 NGLLY=params['NGLLY'], 
                                                 NGLLZ=params['NGLLZ'], 
                                                 NSPEC=params['nspec'], 
                                                 dtype=get_data_type(params['dtype']))
                kernel, padding_num = read_bin(file_name=kernel_file, 
                                               NGLLX=params['NGLLX'], 
                                               NGLLY=params['NGLLY'], 
                                               NGLLZ=params['NGLLZ'], 
                                               NSPEC=params['nspec'], 
                                               dtype=get_data_type(params['dtype']))
                
                precond_kernel = hess_inv * kernel
                
                output_file = os.path.join(output_dir, f"proc{rank:06d}_{kernel_name}_kernel_smooth.bin")
                kernel_pad_and_output(kernel=precond_kernel, output_file=output_file, padding_num=padding_num)
    
    
    
if __name__ == "__main__":

    main()
