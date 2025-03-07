import sys
import os
import json
import logging
from mpi4py import MPI

debug_logger = logging.getLogger("debug_logger")
result_logger = logging.getLogger("result_logger")

    
def check_args():
    if len(sys.argv) != 4:
        raise ValueError('Error: model number is required.')
    if not sys.argv[2].isdigit():
        raise ValueError('Error: lbfgs_flag must be 1 or 0.')
    if not sys.argv[3].isdigit():
        raise ValueError('Error: line_search_flag must be 1 or 0.')

def main():
    
    # --------------------------------------------------------------------------------------------
    # MULTI-PROCESSING SETTING
    # --------------------------------------------------------------------------------------------
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    os.environ["MPI_RANK"] = str(rank)

    # --------------------------------------------------------------------------------------------
    # IMPORT TOOLS
    # --------------------------------------------------------------------------------------------
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from tools.matrix_utils import check_model_threshold, find_minmax, read_bin, kernel_pad_and_output, get_data_type, get_gradient, get_model, get_model_list_from_kernel_type, write_inner_product, compute_inner_product, create_final_gradient
    from tools.job_utils import check_dir_exists, move_files
    import numpy as np
    
    # ----------------------------------------
    # Set Threshold for the model (m/s or g/cm^3)
    # ----------------------------------------
    use_threshold_flag = True
    vp_min, vp_max = 2600, 9500
    vs_min, vs_max = 1500, 5500
    # ----------------------------------------  
    # Check Args
    # ----------------------------------------
    check_args()
    step_fac = float(sys.argv[1])
    lbfgs_flag = int(sys.argv[2])
    line_search_flag = int(sys.argv[3])
    
    
    
    # --------------------------------------------------------------------------------------------
    # Read Parameters
    # --------------------------------------------------------------------------------------------
    
    if rank == 0:
        with open("params.json", "r") as f:
            params = json.load(f)
    else:
        params = None
    params = comm.bcast(params, root=0)
    # --------------------------------------------------------------------------------------------
    # Set up some variables
    # --------------------------------------------------------------------------------

    mrun = params['current_model_num']
    n_store_lbfgs = params['n_store_lbfgs']
    NGLLX = params['NGLLX']
    NGLLY = params['NGLLY']
    NGLLZ = params['NGLLZ']
    NSPEC = params['nspec']
    NGLOB = params['nglob']
    
    kernel_list = params['kernel_list']
    model_list = get_model_list_from_kernel_type(kernel_list)
    stage_initial_model = params['stage_initial_model']
    model_num = f"m{mrun:03d}"
    dtype_int = np.int32
    
    old_model_dir = os.path.join(params['base_dir'], "TOMO", model_num,     "DATABASES_MPI")
    if line_search_flag:
        new_model_dir = os.path.join(params['base_dir'], "TOMO", f"mtest{mrun:03d}", "DATABASES_MPI")
    else:   
        new_model_dir = os.path.join(params['base_dir'], "TOMO", model_num + 1, "DATABASES_MPI")
    direction_dir = os.path.join(params['base_dir'], "TOMO", model_num, "KERNEL", "UPDATE")

    
    # --------------------------------------------------------------------------------------------
    # Calculations & Processing
    # --------------------------------------------------------------------------------------------
    if rank == 0:
        check_dir_exists(old_model_dir)
        check_dir_exists(direction_dir)
        check_dir_exists(new_model_dir)
        
        result_logger.info('--------------------------------------------')
        result_logger.info('--------- Ready for updating model ---------')
        result_logger.info('--------------------------------------------')
        result_logger.info('Reading the direction and models...')
    
    kernels, models = [], []
    for iker, kernel_name in enumerate(kernel_list):
        
        direction_file = f'{direction_dir}/proc{rank:06d}_{kernel_name}_kernel_smooth.bin'
        old_model_file = f'{old_model_dir}/proc{rank:06d}_{model_list[iker]}.bin'
        
        
        direction, _padding_num = read_bin(file_name=direction_file, 
                                NGLLX=params['NGLLX'], 
                                NGLLY=params['NGLLY'], 
                                NGLLZ=params['NGLLZ'], 
                                NSPEC=params['nspec'], 
                                dtype=get_data_type(params['dtype']))
        old_model, _padding_num = read_bin(file_name=old_model_file, 
                                NGLLX=params['NGLLX'], 
                                NGLLY=params['NGLLY'], 
                                NGLLZ=params['NGLLZ'], 
                                NSPEC=params['nspec'], 
                                dtype=get_data_type(params['dtype']))
        
        kernels.append(direction)
        models.append(old_model)
    
    direction_arr = np.vstack(kernels)
    old_model_arr = np.vstack(models)
    _all_kernels_min, all_kernels_max = find_minmax(direction_arr)
    
    
    if lbfgs_flag:
        step_length = step_fac
        new_model_arr = old_model_arr + step_length * direction_arr
    else:
        step_length = step_fac / all_kernels_max
        new_model_arr = old_model_arr * np.exp(step_length * direction_arr)
        
    
    if use_threshold_flag:
        new_model_arr = check_model_threshold(new_model_arr, vp_min, vp_max, vs_min, vs_max)
    
    for iker, kernel_name in enumerate(kernel_list):
        new_model = new_model_arr[iker, :]
        new_model_file = os.path.join(new_model_dir, f"proc{rank:06d}_{model_list[iker]}.bin")
        kernel_pad_and_output(kernel = new_model, output_file = new_model_file, padding_num=_padding_num)
        debug_logger.info(f"Writing the updated model to {new_model_file}")


    # move the mesh files to the new_dir so we don't need to mesh again
    move_files(src_dir=old_model_dir, dst_dir=new_model_dir, pattern='*_Database.bin')
    
if __name__ == "__main__":

    main()
