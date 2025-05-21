import numpy as np
import logging
import sys
import os

debug_logger = logging.getLogger("debug_logger")

def get_model_list_from_kernel_type(kernel_type):
    """
    kernel_type (list): the list of the kernel types, 
                        e.g. ['alpha', 'beta', 'rho']
    Returns:
        model_list (list): the list of the model names, 
                           e.g. ['vp', 'vs', 'rho']
    """
    type_dict = {
        ('alpha', 'beta', 'rho'): ['vp', 'vs', 'rho'],
    }
    return type_dict[tuple(kernel_type)]

def get_data_type(data_type):
    """
    check the data type, and transform it into numpy data type
    Args:
        data_type (str): data type (float32 or float64)
    Returns:
        numpy data type (np.float32 or np.float64)
    """
    if data_type == "float32":
        return np.float32
    elif data_type == "float64":
        return np.float64
    else:
        error_message = f"Unknown data type: {data_type}"
        debug_logger.error(error_message)
        raise ValueError(error_message)

def check_bin_size(NGLLX, NGLLY, NGLLZ, NSPEC, array):
    """
    Check if the size of the given array is the same as the expected size.
    If the size is not correct, then raise a ValueError.
    Args:
        NGLLX (int)  : the number of GLL points in the x direction
        NGLLY (int)  : the number of GLL points in the y direction
        NGLLZ (int)  : the number of GLL points in the z direction
        NSPEC (int)  : the number of spectral elements
        array (np.array): the array to check
    """
    expected_size = NGLLX * NGLLY * NGLLZ * NSPEC
    if array.size != expected_size:
        error_message = f'Error: size of hess_inv is {array.size}, expected {expected_size}'
        debug_logger.error(error_message)
        raise ValueError(error_message)

def read_bin(file_name, NGLLX, NGLLY, NGLLZ, NSPEC, dtype):
    """
    Read in the given binary file and return the array and its padding number.
    Args:
        file_name (str) : the path of the binary file
        NGLLX (int)     : the number of GLL points in the x direction
        NGLLY (int)     : the number of GLL points in the y direction
        NGLLZ (int)     : the number of GLL points in the z direction
        NSPEC (int)     : the number of spectral elements
        dtype (type)    : the type of the array
    Return:
        array (np.array)       : the array read from the binary file
        padding_num (int/float): the padding number
    """
    array = np.fromfile(file_name, dtype=dtype)
    padding_num = array[0]
    array = array[1:-1]
    check_bin_size(NGLLX, NGLLY, NGLLZ, NSPEC, array)

    debug_logger.info(f'Sucessfully read the file from {file_name}')
    return array, padding_num

def get_param_from_specfem_file(file, param_name, param_type=int, digit=3):
    """
    Get the parameters from specfem3d PAR_FILE / OUTPUT_FILES/output_generate_databases.txt
    
    Args:
        param_name (str)  : parameter name
        param_type (type) : the type of the parameter
        digit      (float): if the parameter is float, the number of digits after the decimal point
    return: 
        the value of the parameter
    """
    with open(file) as f:
        for line in f:
            if param_name in line:
                value = line.split()[2]
                if param_type == int:
                    return int(value)
                elif param_type == float:
                    return round(float(value), digit)
                elif param_type == str:
                    return value
    return None

def kernel_pad_and_output(kernel, output_file, padding_num=1e-27):
    """
    Padding a number in the begining and endding of the kernel 
    and output it to the binary file.
    Args:
        kernel (np.array)  : the kernel for outputting
        output_file (str)  : the path of the output binary file
        padding_num (float): the padding number
    """
    kernel_pad = np.pad(kernel, (1, 1), mode='constant', constant_values=padding_num)
    kernel_pad.tofile(output_file)
    
def get_gradient(base_dir, rank, model_num, NGLLX, NGLLY, NGLLZ, NSPEC, NGLOB, ibool_arr, kernel_list, dtype):
    """
    get the gradient from all the different kernels type and combine together.
    Args:
        base_dir (str): the base directory ('AdjointFlows')
        rank (int): the rank index of the processor for multi-processing
        model_num (int): the model number, e.g. 0, 1, 2...
        NGLLX (int): the number of GLL points in the x direction
        NGLLY (int): the number of GLL points in the y direction
        NGLLZ (int): the number of GLL points in the z direction
        NSPEC (int): the number of spectral elements
        NGLOB (int): the number of global points
        ibool_arr (np.array): the array of the boolean index
        kernel_list (list): the list of the kernel names
        dtype (type): the type of the array (np.float32 or np.float64)
    Returns:
        gradient_arr (np.array): the gradient array combined from all the kernels
    """
    model_num = f"m{model_num:03d}"
    print(NGLOB)
    kernel_dir = os.path.join(base_dir, "TOMO", model_num, "KERNEL", "SMOOTH")
    vector_gll = np.zeros((len(kernel_list), NGLOB), dtype=dtype)
    for iker, kernel_name in enumerate(kernel_list):
        gradient_file = os.path.join(kernel_dir, f"proc{rank:06d}_{kernel_name}_kernel_smooth.bin")
        vector, _dummy = read_bin(gradient_file, NGLLX, NGLLY, NGLLZ, NSPEC, dtype)
        vector = vector.reshape((NGLLX, NGLLY, NGLLZ, NSPEC), order='F')
        ibool_arr = ibool_arr.reshape((NGLLX, NGLLY, NGLLZ, NSPEC), order='F')
        for ispec in range(NSPEC):
            indices = ibool_arr[:, :, :, ispec] - 1
            vector_gll[iker, indices] = vector[:, :, :, ispec]
    gradient_arr = np.hstack(vector_gll)
    return gradient_arr

def get_precond(base_dir, rank, model_num, NGLLX, NGLLY, NGLLZ, NSPEC, NGLOB, ibool_arr, kernel_list, dtype):
    """
    get the PRECOND gradient from all the different kernels type and combine together.
    Args:
        base_dir (str): the base directory ('AdjointFlows')
        rank (int): the rank index of the processor for multi-processing
        model_num (int): the model number, e.g. 0, 1, 2...
        NGLLX (int): the number of GLL points in the x direction
        NGLLY (int): the number of GLL points in the y direction
        NGLLZ (int): the number of GLL points in the z direction
        NSPEC (int): the number of spectral elements
        NGLOB (int): the number of global points
        ibool_arr (np.array): the array of the boolean index
        kernel_list (list): the list of the kernel names
        dtype (type): the type of the array (np.float32 or np.float64)
    Returns:
        gradient_arr (np.array): the gradient array combined from all the kernels
    """
    model_num = f"m{model_num:03d}"
    print(NGLOB)
    kernel_dir = os.path.join(base_dir, "TOMO", model_num, "KERNEL", "PRECOND")
    vector_gll = np.zeros((len(kernel_list), NGLOB), dtype=dtype)
    for iker, kernel_name in enumerate(kernel_list):
        gradient_file = os.path.join(kernel_dir, f"proc{rank:06d}_{kernel_name}_kernel_smooth.bin")
        vector, _dummy = read_bin(gradient_file, NGLLX, NGLLY, NGLLZ, NSPEC, dtype)
        vector = vector.reshape((NGLLX, NGLLY, NGLLZ, NSPEC), order='F')
        ibool_arr = ibool_arr.reshape((NGLLX, NGLLY, NGLLZ, NSPEC), order='F')
        for ispec in range(NSPEC):
            indices = ibool_arr[:, :, :, ispec] - 1
            vector_gll[iker, indices] = vector[:, :, :, ispec]
    gradient_arr = np.hstack(vector_gll)
    return gradient_arr

def get_hess_inv(base_dir, rank, model_num, NGLLX, NGLLY, NGLLZ, NSPEC, NGLOB, ibool_arr, kernel_list, dtype):
    """
    get the PRECOND gradient from all the different kernels type and combine together.
    Args:
        base_dir (str): the base directory ('AdjointFlows')
        rank (int): the rank index of the processor for multi-processing
        model_num (int): the model number, e.g. 0, 1, 2...
        NGLLX (int): the number of GLL points in the x direction
        NGLLY (int): the number of GLL points in the y direction
        NGLLZ (int): the number of GLL points in the z direction
        NSPEC (int): the number of spectral elements
        NGLOB (int): the number of global points
        ibool_arr (np.array): the array of the boolean index
        kernel_list (list): the list of the kernel names
        dtype (type): the type of the array (np.float32 or np.float64)
    Returns:
        gradient_arr (np.array): the gradient array combined from all the kernels
    """
    model_num = f"m{model_num:03d}"
    kernel_dir = os.path.join(base_dir, "TOMO", model_num, "KERNEL", "SMOOTH")
    vector_gll = np.zeros((len(kernel_list), NGLOB), dtype=dtype)
    hess_file = os.path.join(kernel_dir, f"proc{rank:06d}_hess_inv_kernel_smooth.bin")
    vector, _dummy = read_bin(hess_file, NGLLX, NGLLY, NGLLZ, NSPEC, dtype)
    vector = vector.reshape((NGLLX, NGLLY, NGLLZ, NSPEC), order='F')
    ibool_arr = ibool_arr.reshape((NGLLX, NGLLY, NGLLZ, NSPEC), order='F')
    for ispec in range(NSPEC):
        indices = ibool_arr[:, :, :, ispec] - 1    
        for iker in range(len(kernel_list)):
            vector_gll[iker, indices] = vector[:, :, :, ispec]
    hess_inv_arr = np.hstack(vector_gll)
    return hess_inv_arr

def get_model(base_dir, rank, model_num, NGLLX, NGLLY, NGLLZ, NSPEC, NGLOB, ibool_arr, model_list, dtype):
    """
    get the gradient from all the different kernels type and combine together.
    Args:
        base_dir (str): the base directory ('AdjointFlows')
        rank (int): the rank index of the processor for multi-processing
        model_num (int): the model number, e.g. 0, 1, 2...
        NGLLX (int): the number of GLL points in the x direction
        NGLLY (int): the number of GLL points in the y direction
        NGLLZ (int): the number of GLL points in the z direction
        NSPEC (int): the number of spectral elements
        NGLOB (int): the number of global points
        ibool_arr (np.array): the array of the boolean index
        model_list (list): the list of the model names (vp, vs, rho...)
        dtype (type): the type of the array (np.float32 or np.float64)
    Returns:
        model_arr (np.array): the model array combined from all the kernels
    """
    model_num = f"m{model_num:03d}"
    model_dir = os.path.join(base_dir, "TOMO", model_num, "DATABASES_MPI")
    vector_gll = np.zeros((len(model_list), NGLOB), dtype=dtype)
    for imod, model_name in enumerate(model_list):
        model_file = os.path.join(model_dir, f"proc{rank:06d}_{model_name}.bin")
        vector, _dummy = read_bin(model_file, NGLLX, NGLLY, NGLLZ, NSPEC, dtype)
        vector = vector.reshape((NGLLX, NGLLY, NGLLZ, NSPEC), order='F')
        ibool_arr = ibool_arr.reshape((NGLLX, NGLLY, NGLLZ, NSPEC), order='F')
        for ispec in range(NSPEC):
            indices = ibool_arr[:, :, :, ispec] - 1
            vector_gll[imod, indices] = vector[:, :, :, ispec]
    model_arr = np.hstack(vector_gll)
    return model_arr

def write_inner_product(g_dot_p, g_dot_g, p_dot_p, file_name):
    """
    Write the inner products into a file.
    Args:
        g_dot_p (float): the inner product of gradient and direction
        g_dot_g (float): the inner product of gradient and gradient
        p_dot_p (float): the inner product of direction and direction
        file_name (str): the path of the output file
    """
    with open(file_name, 'w') as f:
        f.write(f'g_dot_p: {g_dot_p:12.6e}\n')
        f.write(f'g_dot_g: {g_dot_g:12.6e}\n')
        f.write(f'p_dot_p: {p_dot_p:12.6e}\n')
        
def compute_inner_product(vector1, vector2):
    """
    Compute the inner products of two vectors.
    Args:
        vector1 (np.array): the first vector
        vector2 (np.array): the second vector
    """
    v1_v2 = np.sum(vector1 * vector2)
    v1_v1 = np.sum(vector1 * vector1)
    v2_v2 = np.sum(vector2 * vector2)
    return v1_v2, v1_v1, v2_v2

def find_minmax(vector):
    min_val = np.min(vector)
    max_val = np.max(vector)
    return min_val, max_val

def check_model_threshold(new_model_arr, vp_min, vp_max, vs_min, vs_max):
    vp = new_model_arr[0, :]
    vs = new_model_arr[1, :]
    rho = new_model_arr[2, :]
    
    vp = np.clip(vp, vp_min, vp_max)
    vs = np.clip(vs, vs_min, vs_max)
    
    new_model_arr[0, :] = vp
    new_model_arr[1, :] = vs
    new_model_arr[2, :] = rho
    
    return new_model_arr

def check_final_threshold(new_model_arr, vp_min, vp_max, vs_min, vs_max):
    vp = new_model_arr[0, :]
    vs = new_model_arr[1, :]
    rho = new_model_arr[2, :]

    # check if the model is in the range
    if np.any(vp < vp_min) or np.any(vp > vp_max):
        raise ValueError("Vp out of acceptable range. Stopping execution.")
    if np.any(vs < vs_min) or np.any(vs > vs_max):
        raise ValueError("Vs out of acceptable range. Stopping execution.")



def check_model_poisson_ratio(new_model_arr, nu_min=0.0, nu_max=0.45):
    vp = new_model_arr[0, :]
    vs = new_model_arr[1, :]
    rho = new_model_arr[2, :]

    nu = (vp**2 - 2 * vs**2) / (2 * (vp**2 - vs**2))
    # show the min/max of original poisson's ratio
    nu_min_orig, nu_max_orig = find_minmax(nu)
    # debug_logger.info(f"Original poisson's ratio min: {nu_min_orig}, max: {nu_max_orig}")

    if np.any(nu < nu_min) or np.any(nu > nu_max):
        raise ValueError("Poisson's ratio out of acceptable range. Stopping execution.")

    mask_low = nu < nu_min
    mask_high = nu > nu_max

    # correcting vp if nu < min
    vs_low = vs[mask_low]
    vp[mask_low] = vs_low * np.sqrt((2 * (1 - nu_min)) / (1 - 2 * nu_min)) 

    # correcting vp if nu < max
    vs_high = vs[mask_high]
    vp[mask_high] = vs_high * np.sqrt((2 * (1 - nu_max)) / (1 - 2 * nu_max))

    new_model_arr[0, :] = vp
    new_model_arr[1, :] = vs
    new_model_arr[2, :] = rho

    return new_model_arr


def check_model_vpvs_ratio(new_model_arr, min=1.15, max=2.8):
    vp = new_model_arr[0, :]
    vs = new_model_arr[1, :]

    vpvs = vp / vs
    vpvs_smaller = vpvs < min
    vpvs_greater = vpvs > max
    
    # correcting vs if vpvs < min
    vs[vpvs_smaller] = vp[vpvs_smaller] / min
    vs[vpvs_greater] = vp[vpvs_greater] / max

    new_model_arr[0, :] = vp
    new_model_arr[1, :] = vs

    return new_model_arr




def restore_vector(gll_arr, ibool_arr, NGLLX, NGLLY, NGLLZ, NSPEC, dtype):
    """
    Restore the gradient array back to the original vector format.
    
    Args:
        gradient_arr (np.array): The gradient array combined from all kernels.
        ibool_arr (np.array): The boolean index array.
        NGLLX (int): The number of GLL points in the x direction.
        NGLLY (int): The number of GLL points in the y direction.
        NGLLZ (int): The number of GLL points in the z direction.
        NSPEC (int): The number of spectral elements.
        kernel_list (list): The list of kernel names.
        dtype (type): The type of the array (np.float32 or np.float64).

    Returns:
        restored_vector (np.array): The restored vector in the shape (NGLLX, NGLLY, NGLLZ, NSPEC, len(kernel_list)).
    """
    restored_vector = np.zeros((NGLLX, NGLLY, NGLLZ, NSPEC), dtype=dtype)
    ibool_arr = ibool_arr.reshape((NGLLX, NGLLY, NGLLZ, NSPEC), order='F')
    for ispec in range(NSPEC):
        indices = ibool_arr[:, :, :, ispec] - 1
        restored_vector[:,:,:, ispec] = gll_arr[indices]
    
    return restored_vector