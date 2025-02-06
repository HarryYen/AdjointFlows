from tools import ConfigManager, FileManager
from tools import GLOBAL_PARAMS
from kernel import ModelGenerator, ForwardGenerator
from workflow import WorkflowController
from mpi4py import MPI
import yaml
import os
import sys



def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    print(f"Hello from rank {rank} out of {size}")

    if rank == 0:
        print("Start the adjoint tomography workflow...")
        
        config = ConfigManager('config.yaml')
        config.load()
        
        workflow_controller = WorkflowController(config=config, global_params=GLOBAL_PARAMS)
        workflow_controller.setup()
        
        workflow_controller.move_to_other_directory(folder_to_move='specfem')
        workflow_controller.generate_model()
        
        workflow_controller.run_forward()
        # workflow_controller.misfit_check()
        # workflow_controller.create_misfit_kernel()
        # workflow_controller.move_to_other_directory(folder_to_move='iterate')
        
    
    
    
    
if __name__ == '__main__':
    
    main()	
