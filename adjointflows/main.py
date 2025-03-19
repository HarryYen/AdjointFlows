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

    # --------------------------------------------------------------------------
    # Settings
    # --------------------------------------------------------------------------
    
    setup_logging()
    debug_logger = logging.getLogger("debug_logger")
    result_logger = logging.getLogger("result_logger")
    
    debug_logger.info("Start the adjoint tomography workflow...")
    config = ConfigManager('config.yaml')
    config.load()
    
    MAX_ATTEMPTS = config.get('inversion.max_fail')
    current_model_num = config.get('setup.model.current_model_num')
    stage_initial_model = config.get('setup.stage.stage_initial_model')
    which_step = config.get('setup.workflow.start_step')
    
    attempt = 0
    misfit_reduced = False
    # do_mesh = current_model_num == stage_initial_model
    do_mesh = bool(config.get('setup.model.do_mesh'))
    # ---------------------------------------------------------------------------
    
    workflow_controller = WorkflowController(config=config, global_params=GLOBAL_PARAMS)
    workflow_controller.setup_for_fail()

    # ---------------------------------------------------------------------------
    # determine which step to start
    # ---------------------------------------------------------------------------
    step_name_list = [
        'forward',
        'post_processing',
        'inversion',
    ]
    
    result_logger.info(f"Workflow: User choose to start from {step_name_list[which_step]}")
    # ---------------------------------------------------------------------------
    """
    Main loop
    FLOW:
        1. Run forward simulation and adjoint simulation.
        2. check misfit comparing to the previous model.
        3. If the following conditions are met, go to the next step.
            - Misfit is reduced.
            - Maximum attempts are not reached.
        4. create misfit kernel (post-processing).
        5. do inversion (Steepest descent or L-BFGS)
    """
    
    while not misfit_reduced and attempt < MAX_ATTEMPTS and which_step <= step_name_list.index('forward'):
        attempt += 1
        workflow_controller.setup_dir()
        workflow_controller.move_to_other_directory(folder_to_move='specfem')

        workflow_controller.generate_model(mesh_flag=do_mesh)
        
        workflow_controller.run_forward()
        if workflow_controller.misfit_check():
            misfit_reduced = True
        else:
            workflow_controller.reupdate_model_if_misfit_not_reduced()
        do_mesh = False
        
        
    if not misfit_reduced and attempt == MAX_ATTEMPTS:
        result_logger.warning("STOP: Reached max attempts without reducing misfit.")
        sys.exit(0)

    if which_step <= step_name_list.index('post_processing'):
        workflow_controller.create_misfit_kernel()
    if which_step <= step_name_list.index('inversion'):
        workflow_controller.move_to_other_directory(folder_to_move='adjointflows')
        workflow_controller.do_iteration()

        

if __name__ == '__main__':
    
    main()	
