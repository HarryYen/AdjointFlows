import sys
import os
import json
import logging
from mpi4py import MPI

debug_logger = logging.getLogger("debug_logger")


def get_range_lbfgs_memory(iter_currnet, n_store, mstart_min):
    """
    get the range of the iteration number to store in the memory
    Args:
        iter_currnet (int): the current iteration number
        n_store (int)     : the number of iterations to store in the memory
        mstart_min (int)  : the minimum iteration number IN THIS STAGE!!
    """
    if iter_currnet < n_store:
        iter_range = range(iter_currnet)
    else:
        iter_range = range(iter_currnet - n_store, iter_currnet)
    iter_range = [i for i in iter_range if i >= mstart_min]
    
    debug_logger.info(f'LBFGS: the range of memory models is {iter_range}')
    return iter_range


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
    from tools.matrix_utils import read_bin, kernel_pad_and_output, get_data_type, get_gradient, get_model, get_model_list_from_kernel_type, write_inner_product, compute_inner_product, create_final_gradient
    from tools.job_utils import check_dir_exists
    import numpy as np
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
    output_inner_product_file = 'output_inner_product.txt'
    
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
    epsilon = 1e-12
    
    databases_dir = os.path.join(params['base_dir'], "TOMO", model_num, "DATABASES_MPI")
    output_dir = os.path.join(params['base_dir'], "TOMO", model_num, "KERNEL", "UPDATE")
    lbfgs_memory_range = get_range_lbfgs_memory(iter_currnet=mrun, 
                                                n_store=n_store_lbfgs, 
                                                mstart_min=stage_initial_model)
    
    # --------------------------------------------------------------------------------------------
    # Calculations & Processing
    # --------------------------------------------------------------------------------------------
    
    if rank == 0:
        check_dir_exists(output_dir)
    
    ibool_file = os.path.join(databases_dir, f"proc{rank:06d}_ibool.bin")
    ibool_kernel, _padding_num = read_bin(file_name=ibool_file, 
                                    NGLLX=params['NGLLX'], 
                                    NGLLY=params['NGLLY'], 
                                    NGLLZ=params['NGLLZ'], 
                                    NSPEC=params['nspec'], 
                                    dtype=dtype_int)
    
    common_params = dict(
        base_dir=params['base_dir'], 
        NGLLX=NGLLX, NGLLY=NGLLY, NGLLZ=NGLLZ, 
        NSPEC=NSPEC, NGLOB=NGLOB, ibool_arr=ibool_kernel, 
        kernel_list=kernel_list, dtype=get_data_type(params['dtype'])
    )
    
    current_gradient = get_gradient(**common_params, rank=rank, model_num=mrun)

        
    q_vector = current_gradient
    
    if rank == 0:
        p_dict = {}
        a_dict = {}
    else:
        p_dict = None
        a_dict = None
        
    for iter in reversed(lbfgs_memory_range):
        model_1 = iter + 1
        model_0 = iter

        gradient1_arr = get_gradient(**common_params, rank=rank, model_num=model_1)
        model1_arr    = get_model(**common_params, rank=rank, model_num=model_1)
        gradient0_arr = get_gradient(**common_params, rank=rank, model_num=model_0)
        model0_arr    = get_model(**common_params, rank=rank, model_num=model_0)
        
        grad_diff = gradient1_arr - gradient0_arr
        model_diff = model1_arr - model0_arr
        
        p_tmp = grad_diff * model_diff
        a_tmp = model_diff * q_vector
        
        # -----------------------------------------------------------
        # MPI COUMMUNICATION 
        # -----------------------------------------------------------
        p_tmp_sum_local = np.sum(p_tmp)
        p_sum = np.array(0.0)
        comm.Allreduce(p_tmp_sum_local, p_sum, op=MPI.SUM)
        
        a_tmp_sum_local = np.sum(a_tmp)
        a_sum = np.array(0.0)
        comm.Allreduce(a_tmp_sum_local, a_sum, op=MPI.SUM)
        # -----------------------------------------------------------
        if rank == 0:
            p_dict[iter] = 1. / p_sum
            a_dict[iter] = p_dict[iter] * a_sum
            
            if np.abs(p_sum) < epsilon:
                logging.error(f"Warning: p_sum ({p_sum}) is too small at iteration {iter}, skipping update.")
            
        # broadcast the p_dict and a_dict
        p_dict = comm.bcast(p_dict, root=0)
        a_dict = comm.bcast(a_dict, root=0)
        # -----------------------------------------------------------
        q_vector = q_vector - a_dict[iter] * grad_diff
    
    iter = mrun - 1
    
    gradient1_arr = get_gradient(**common_params, rank=rank, model_num=iter+1)
    model1_arr    = get_model(**common_params, rank=rank, model_num=iter+1)
    gradient0_arr = get_gradient(**common_params, rank=rank, model_num=iter)
    model0_arr    = get_model(**common_params, rank=rank, model_num=iter)
    
    grad_diff = gradient1_arr - gradient0_arr
    model_diff = model1_arr - model0_arr
    
    p_k_up   = grad_diff * model_diff
    p_k_down = grad_diff * grad_diff
    # -----------------------------------------------------------
    # MPI COUMMUNICATION 
    # -----------------------------------------------------------
    p_k_up_tmp_sum_local = np.sum(p_k_up)
    p_k_up_sum = np.array(0.0)
    comm.Allreduce(p_k_up_tmp_sum_local, p_k_up_sum, op=MPI.SUM)
    
    p_k_down_tmp_sum_local = np.sum(p_k_down)
    p_k_down_sum = np.array(0.0)
    comm.Allreduce(p_k_down_tmp_sum_local, p_k_down_sum, op=MPI.SUM)
     # -----------------------------------------------------------
    if rank == 0:
        p_k = p_k_up_sum / p_k_down_sum
    # broadcast the p_dict and a_dict
    p_k = comm.bcast(p_k, root=0)
    # -----------------------------------------------------------
    r_vector = q_vector * p_k
    
    
    for iter in lbfgs_memory_range:
        model_1 = iter + 1
        model_0 = iter
    
        gradient1_arr = get_gradient(**common_params, rank=rank, model_num=model_1)
        model1_arr    = get_model(**common_params, rank=rank, model_num=model_1)
        gradient0_arr = get_gradient(**common_params, rank=rank, model_num=model_0)
        model0_arr    = get_model(**common_params, rank=rank, model_num=model_0)

        grad_diff = gradient1_arr - gradient0_arr
        model_diff = model1_arr - model0_arr
        
        b_tmp = grad_diff * r_vector
        # -----------------------------------------------------------
        # MPI COUMMUNICATION 
        # -----------------------------------------------------------
        b_tmp_sum_local = np.sum(b_tmp)
        b_sum = np.array(0.0)
        comm.Allreduce(b_tmp_sum_local, b_sum, op=MPI.SUM)
        # -----------------------------------------------------------
        if rank == 0:
            b_value = b_sum * p_dict[iter]
        # broadcast the p_dict and a_dict
        b_value = comm.bcast(b_value, root=0)
        # -----------------------------------------------------------
        r_vector = r_vector + (a_dict[iter] - b_value) * model_diff
    
    r_vector = -1. * r_vector

    gp_local, gg_local, pp_local = compute_inner_product(vector1=current_gradient, vector2=r_vector)
    # -----------------------------------------------------------
    # MPI COUMMUNICATION 
    # -----------------------------------------------------------
    g_dot_p, g_dot_g, p_dot_p = np.array(0.0), np.array(0.0), np.array(0.0)
    comm.Allreduce(gp_local, g_dot_p, op=MPI.SUM)
    comm.Allreduce(gg_local, g_dot_g, op=MPI.SUM)
    comm.Allreduce(pp_local, p_dot_p, op=MPI.SUM)
    # -----------------------------------------------------------
    write_inner_product(g_dot_p=g_dot_p, g_dot_g=g_dot_g, p_dot_p=p_dot_p, file_name=output_inner_product_file)

    # ----------------------------------------------------------------------------------------------
    # output
    # ----------------------------------------------------------------------------------------------
    vector = create_final_gradient(gradient=r_vector, 
                                   NGLOB=NGLOB, NGLLX=NGLLX, NGLLY=NGLLY, NGLLZ=NGLLZ, 
                                   NSPEC=NSPEC, kernel_list=kernel_list, ibool_arr=ibool_kernel)
    for iker, kernel_name in enumerate(kernel_list):
        output_file = f'{output_dir}/proc{rank:06d}_{kernel_name}_kernel_smooth.bin'
        output_arr = vector[iker, :, :, :, :].flatten()
        kernel_pad_and_output(kernel=output_arr, output_file=output_file, padding_num=_padding_num)
    
if __name__ == "__main__":

    main()