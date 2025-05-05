import sys
import os
import json
import logging
from mpi4py import MPI



def setup_logging():
    # -----------------------------------------------------
    # Debug Logger (Recorded in both debug.log and terminal)
    # -----------------------------------------------------
    debug_logger = logging.getLogger("update_debug_logger")
    debug_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    debug_file_handler = logging.FileHandler("logger/update.log", mode="w")
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    debug_file_handler.setFormatter(debug_file_formatter)

    debug_logger.addHandler(console_handler)
    debug_logger.addHandler(debug_file_handler)

    debug_logger.propagate = False

    
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
    from tools.matrix_utils import find_minmax, read_bin, kernel_pad_and_output, get_data_type, get_model_list_from_kernel_type, check_model_threshold, check_final_threshold, check_model_vpvs_ratio, check_model_poisson_ratio
    from tools.job_utils import check_dir_exists, copy_files
    import numpy as np
    
    # ----------------------------------------
    # Set Threshold for the model (m/s or g/cm^3)
    # ----------------------------------------
    use_threshold_flag = True
    vp_min, vp_max = 2590, 9500
    vs_min, vs_max = 1490, 5500
    
    # poisson ratio
    # nu_min, nu_max = -0.95, 0.45
    # vp vs ratio limit
    vpvs_min, vpvs_max = 1.16, 2.8
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

    if rank == 0:
        setup_logging()
        debug_logger = logging.getLogger("update_debug_logger")
    
    
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
    model_next_num = f'm{mrun+1:03d}'
    dtype_int = np.int32
    
    old_model_dir = os.path.join(params['base_dir'], "TOMO", model_num,     "DATABASES_MPI")
    if line_search_flag:
        new_model_dir = os.path.join(params['base_dir'], "TOMO", f"mtest{mrun:03d}", "DATABASES_MPI")
    else:   
        new_model_dir = os.path.join(params['base_dir'], "TOMO", model_next_num, "DATABASES_MPI")
    direction_dir = os.path.join(params['base_dir'], "TOMO", model_num, "KERNEL", "UPDATE")

    
    # --------------------------------------------------------------------------------------------
    # Calculations & Processing
    # --------------------------------------------------------------------------------------------

    
    if rank == 0:
        debug_logger.info(f"model_update: updating the model with step_fac: {step_fac}")
        check_dir_exists(old_model_dir)
        check_dir_exists(direction_dir)
        check_dir_exists(new_model_dir)
        
        debug_logger.info('--------------------------------------------')
        debug_logger.info('--------- Ready for updating model ---------')
        debug_logger.info('--------------------------------------------')
        debug_logger.info('Reading the direction and models...')
    
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
    
    
    _local_min, local_max = find_minmax(np.abs(direction_arr))
    all_kernels_max = comm.allreduce(local_max, op=MPI.MAX)

    if lbfgs_flag:
        step_length = step_fac
        new_model_arr = old_model_arr + step_length * direction_arr
    else:
        step_length = step_fac / all_kernels_max
        new_model_arr = old_model_arr * np.exp(step_length * direction_arr)
        
    
    if use_threshold_flag:
        new_model_arr = check_model_threshold(new_model_arr, vp_min, vp_max, vs_min, vs_max)
        new_model_arr = check_model_vpvs_ratio(new_model_arr=new_model_arr, min=vpvs_min, max=vpvs_max)
        check_final_threshold(new_model_arr, vp_min, vp_max, vs_min, vs_max)
    
    # new_model_arr = check_model_poisson_ratio(new_model_arr, nu_min=nu_min, nu_max=nu_max)
    
    for iker, kernel_name in enumerate(kernel_list):
        old_model = old_model_arr[iker, :]
        new_model = new_model_arr[iker, :]
        model_diff = new_model - old_model
        new_model_file = os.path.join(new_model_dir, f"proc{rank:06d}_{model_list[iker]}.bin")
        kernel_pad_and_output(kernel = new_model, output_file = new_model_file, padding_num=_padding_num)
        
        if rank == 0 :
            debug_logger.info(f"Writing the updated model to {new_model_file}")

        # ---------------------
        # print min max before and after
        # ---------------------
        old_model_local_min, old_model_local_max = find_minmax(old_model)
        old_model_global_min = comm.allreduce(old_model_local_min, op=MPI.MIN)
        old_model_global_max = comm.allreduce(old_model_local_max, op=MPI.MAX)
        old_local_sum = np.sum(old_model)
        old_local_count = old_model.size
        old_global_sum = comm.allreduce(old_local_sum, op=MPI.SUM)
        old_global_count = comm.allreduce(old_local_count, op=MPI.SUM)
        old_global_mean = old_global_sum / old_global_count
        
        new_model_local_min, new_model_local_max = find_minmax(new_model)
        new_model_global_min = comm.allreduce(new_model_local_min, op=MPI.MIN)
        new_model_global_max = comm.allreduce(new_model_local_max, op=MPI.MAX)
        new_local_sum = np.sum(new_model)
        new_local_count = new_model.size
        new_global_sum = comm.allreduce(new_local_sum, op=MPI.SUM)
        new_global_count = comm.allreduce(new_local_count, op=MPI.SUM)
        new_global_mean = new_global_sum / new_global_count
        
        # print global model diff
        model_diff_local_min, model_diff_local_max = find_minmax(model_diff)
        model_diff_global_min = comm.allreduce(model_diff_local_min, op=MPI.MIN)
        model_diff_global_max = comm.allreduce(model_diff_local_max, op=MPI.MAX)
        
        
        
        
        if rank == 0:
            debug_logger.info(f"OLD {kernel_name} model min: {old_model_global_min}, max: {old_model_global_max}")
            debug_logger.info(f"NEW {kernel_name} model min: {new_model_global_min}, max: {new_model_global_max}")
            debug_logger.info(f"OLD {kernel_name} model mean: {old_global_mean}")
            debug_logger.info(f"NEW {kernel_name} model mean: {new_global_mean}")
            debug_logger.info(f"Model diff min: {model_diff_global_min}, max: {model_diff_global_max}")
            

    # move the mesh files to the new_dir so we don't need to mesh again
    if rank == 0:
        debug_logger.debug(f"Moving {old_model_dir} to {new_model_dir}...")
        copy_files(src_dir=old_model_dir, dst_dir=new_model_dir, pattern='*_Database')
    
if __name__ == "__main__":

    main()
