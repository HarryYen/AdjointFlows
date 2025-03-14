from tools import ConfigManager
from tools import GLOBAL_PARAMS
from workflow import WorkflowController
import sys
import logging

def setup_logging():

    # -----------------------------------------------------
    # Debug Logger (Recorded in both debug.log and terminal)
    # -----------------------------------------------------
    debug_logger = logging.getLogger("debug_logger")
    debug_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    debug_file_handler = logging.FileHandler("logger/debug.log", mode="w")
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    debug_file_handler.setFormatter(debug_file_formatter)

    debug_logger.addHandler(console_handler)
    debug_logger.addHandler(debug_file_handler)

    # -----------------------------------------------------
    # Result Logger (Only recorded in result.log)
    # -----------------------------------------------------
    result_logger = logging.getLogger("result_logger")
    result_logger.setLevel(logging.INFO)

    result_file_handler = logging.FileHandler("logger/result.log", mode="w")
    result_file_handler.setLevel(logging.INFO)
    result_file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    result_file_handler.setFormatter(result_file_formatter)

    result_logger.addHandler(result_file_handler)
    result_logger.addHandler(console_handler)
    result_logger.addHandler(debug_file_handler)


    debug_logger.propagate = False
    result_logger.propagate = False



def main():
    """
    1. if the current model number is the initial model number, do meshing.
       But meshing is only for the first round.
    """
    # -------------------------------
    # Settings
    # -------------------------------
    
    setup_logging()
    debug_logger = logging.getLogger("debug_logger")
    result_logger = logging.getLogger("result_logger")
    
    debug_logger.info("Start tuning the flexwin parameters...")
    config = ConfigManager('config.yaml')
    config.load()
    
    current_model_num = config.get('setup.model.current_model_num')
    stage_initial_model = config.get('setup.stage.stage_initial_model')
    do_mesh = current_model_num == stage_initial_model
    # -------------------------------
    
    workflow_controller = WorkflowController(config=config, global_params=GLOBAL_PARAMS)
    # workflow_controller.setup_dir()
    workflow_controller.move_to_other_directory(folder_to_move='specfem')
    # workflow_controller.generate_model(mesh_flag=do_mesh)
    workflow_controller.run_forward_for_tuning_flexwin(do_forward=False)
        
    
    
    
    
if __name__ == '__main__':
    
    main()	