import numpy as np
import logging
import sys
import os

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
        logging.error(error_message)
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