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

        # ----------------------------------------------------
        # defind the file names
        # ----------------------------------------------------
        ibool_file = os.path.join(databases_dir, f"proc{rank:06d}_ibool.bin")
        x_file = os.path.join(databases_dir, f"proc{rank:06d}_x.bin")
        y_file = os.path.join(databases_dir, f"proc{rank:06d}_y.bin")
        z_file = os.path.join(databases_dir, f"proc{rank:06d}_z.bin")
        vector_file = os.path.join(databases_dir, f"proc{rank:06d}_{vector}.bin")

        # ----------------------------------------------------
        # Read files and construct arrays
        # ----------------------------------------------------
        ibool_arr, ibool_data_type = read_binary_int32(ibool_file)
        x_arr, _ = read_binary_float32(x_file)
        y_arr, _ = read_binary_float32(y_file)
        z_arr, _ = read_binary_float32(z_file)
        vector_arr, vector_data_type = read_binary_float32(vector_file)

        x_arr = x_arr[ibool_arr - 1]
        y_arr = y_arr[ibool_arr - 1]
        z_arr = z_arr[ibool_arr - 1]
        

        # ----------------------------------------------------
        # Add anomaly
        # ----------------------------------------------------
        vector_arr = add_gaussian_perturb(data=vector_arr, 
                                          x_arr=x_arr, y_arr=y_arr, z_arr=z_arr, 
                                          center=(9.1e+05, 2.66e+06, -1.5e+04), 
                                          amplitude=0.15, 
                                          sigma=(3000, 3000, 3000))

        # ----------------------------------------------------
        # Output the file
        # ----------------------------------------------------
        os.makedirs(output_dir, exist_ok=True)
        output_binary_file(data = vector_arr, 
                           outfile = f'{output_dir}/{os.path.basename(vector_file)}', 
                           data_type_int = ibool_data_type, 
                           data_type_float = vector_data_type)
        
        
