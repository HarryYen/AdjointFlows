import sys
import os
import json
import logging
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
    from tools.matrix_utils import read_bin, kernel_pad_and_output, get_data_type
    from tools.job_utils import check_dir_exists
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
    
    kernel_dir = os.path.join(params['base_dir'], "TOMO", model_num, "KERNEL", "PRECOND")
    output_dir = os.path.join(params['base_dir'], "TOMO", model_num, "KERNEL", "UPDATE")
    check_dir_exists(kernel_dir)
    check_dir_exists(output_dir)
    
    for kernel_name in params['kernel_list']:
        kernel_file = os.path.join(kernel_dir, f"proc{rank:06d}_{kernel_name}_kernel_smooth.bin")
        gradient_kernel, padding_num = read_bin(file_name=kernel_file, 
                                        NGLLX=params['NGLLX'], 
                                        NGLLY=params['NGLLY'], 
                                        NGLLZ=params['NGLLZ'], 
                                        NSPEC=params['nspec'], 
                                        dtype=get_data_type(params['dtype']))
        
        direction_kernel = gradient_kernel * -1
        
        output_file = os.path.join(output_dir, f"proc{rank:06d}_{kernel_name}_kernel_smooth.bin")
        kernel_pad_and_output(kernel=direction_kernel, output_file=output_file, padding_num=padding_num)


if __name__ == "__main__":

    main()