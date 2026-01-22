from modules_anomaly import *
import numpy as np
import sys
import os


if __name__ == '__main__':
    
    # --------------------
    # Parameters
    # --------------------
    databases_dir = 'DATABASES_MPI'      
    vector = 'vs'
    nproc = 2
    output_dir = 'OUTPUT'
    # --------------------
    # main
    # --------------------
    for rank in range(nproc):
        print('processing rank:', rank)

        ibool_file = os.path.join(databases_dir, f"proc{rank:06d}_ibool.bin")
        # x_file = os.path.join(databases_dir, f"proc{rank:06d}_x.bin")
        # y_file = os.path.join(databases_dir, f"proc{rank:06d}_y.bin")
        # z_file = os.path.join(databases_dir, f"proc{rank:06d}_z.bin")
        vector_file = os.path.join(databases_dir, f"proc{rank:06d}_{vector}.bin")

        ibool_arr, ibool_data_type = read_binary_int32(ibool_file)
        vector_arr, vector_data_type = read_binary_float32(vector_file)
        print(vector_arr)
        vector_arr = add_constant_perturb(vector_arr, constant=-50.0)
        print(vector_arr)
        os.makedirs(output_dir, exist_ok=True)
        output_binary_file(data = vector_arr, 
                           outfile = f'{output_dir}/{os.path.basename(vector_file)}', 
                           data_type_int = ibool_data_type, 
                           data_type_float = vector_data_type)
        
        
