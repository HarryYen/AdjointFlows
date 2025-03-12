from tools import ConfigManager
from tools import GLOBAL_PARAMS
from workflow import WorkflowController
from iterate import IterationProcess
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

    debug_file_handler = logging.FileHandler("logger/user.log", mode="w")
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    debug_file_handler.setFormatter(debug_file_formatter)

    debug_logger.addHandler(console_handler)
    debug_logger.addHandler(debug_file_handler)

    debug_logger.propagate = False



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
    
    debug_logger.info("Start User MAIN...")
    config = ConfigManager('config.yaml')
    config.load()
    
    current_model_num = config.get('setup.model.current_model_num')
    stage_initial_model = config.get('setup.stage.stage_initial_model')
    # -------------------------------
    
    iteration_process = IterationProcess(current_model_num=current_model_num, config=config)
    iteration_process.update_model(step_fac=0.05, lbfgs_flag=False)
    
    
    
    
if __name__ == '__main__':
    
    main()	
