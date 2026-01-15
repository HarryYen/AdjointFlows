from tools import ConfigManager
from tools import GLOBAL_PARAMS
from workflow import WorkflowController
import sys
import logging
import datetime

def setup_logging():

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # -----------------------------------------------------
    # Debug Logger (Recorded in both debug.log and terminal)
    # -----------------------------------------------------
    debug_logger = logging.getLogger("debug_logger")
    debug_logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)

    debug_file_handler = logging.FileHandler(f"logger/debug_{timestamp}.log", mode="w")
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

    result_file_handler = logging.FileHandler(f"logger/result_{timestamp}.log", mode="w")
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
    # (0a) Setting up parameters from config file
    # --------------------------------------------------------------------------
    
    setup_logging()
    debug_logger = logging.getLogger("debug_logger")
    result_logger = logging.getLogger("result_logger")
    
    debug_logger.info("Start the adjoint tomography workflow...")
    config = ConfigManager('config.yaml')
    config.load()
    
    MAX_ATTEMPTS = config.get('inversion.max_fail')
    
    run_mode = config.get('setup.workflow.run_mode')
    start_from_stage = config.get('setup.workflow.start_from_stage')
    end_at_stage = config.get('setup.workflow.end_at_stage')
    forward_stop_at = config.get('setup.workflow.forward_stop_at')
    do_wave_simulation = bool(config.get('setup.workflow.do_wave_simulation'))
    

    attempt = 0
    misfit_reduced = False
    do_mesh = bool(config.get('setup.model.do_mesh'))
    do_generate = bool(config.get('setup.model.do_generate'))

    # ---------------------------------------------------------------------------
    # (0b) Checking and modifying parameters
    # ---------------------------------------------------------------------------
    # Check the stage order
    stage_order = {
        "forward": 1,
        "postprocess": 2,
        "inversion": 3,
    }
    start_index = stage_order[start_from_stage]
    end_index = stage_order[end_at_stage]
    if start_index > end_index:
        result_logger.error(f'start_from_stage ({start_from_stage}) cannot be after end_at_stage ({end_at_stage}).')
        raise ValueError("start_from_stage cannot be after end_at_stage.")
    
    # Check the effectivity of forward_stop_at
    if end_at_stage != 'forward':
        forward_stop_at = 'full'
    
    do_measurement = forward_stop_at != 'synthetics'
    do_adjoint = forward_stop_at not in ('misfit', 'synthetics')
    
    # ---------------------------------------------------------------------------
    # (0c) Initiation
    # ---------------------------------------------------------------------------
    workflow_controller = WorkflowController(config=config, global_params=GLOBAL_PARAMS)
    workflow_controller.setup_for_fail()
    result_logger.info(f"Workflow: User choose to start from {start_from_stage}")
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
    
    while not misfit_reduced and attempt < MAX_ATTEMPTS and stage_order[start_from_stage] <= stage_order['forward']:
        
        attempt += 1
        # -------------------------------------------------------------------------
        # (1) Mesh and generate model  
        # -------------------------------------------------------------------------
        workflow_controller.move_to_other_directory(folder_to_move='specfem')
        if do_generate or attempt > 1:
            workflow_controller.generate_model(mesh_flag=do_mesh)
        
        # -------------------------------------------------------------------------
        # (2) Forward (Pipeline or tuning parameters from FLEXWIN)  
        # -------------------------------------------------------------------------
        if run_mode == 'flexwin_test':
            workflow_controller.run_forward_for_tuning_flexwin(do_forward=do_wave_simulation)
            return 0
        else:
            workflow_controller.run_forward(do_forward=do_wave_simulation, do_adjoint=do_adjoint, do_measurement=do_measurement)
        
        # -------------------------------------------------------------------------
        # (2a) Handle forward-stop modes:
        #   - 'gradient'  : stop forward after gradient computation (continue to next stage)
        #   - 'misfit'    : stop entire pipeline after misfit measurement
        #   - 'synthetics': stop after waveform modeling (SYN/ only)
        # -------------------------------------------------------------------------
        if forward_stop_at == 'gradient':
            result_logger.info("forward_stop_at='gradient': forward terminated after gradient computation.")
            break
        elif forward_stop_at == 'misfit':
            result_logger.info("Only compute misfit as requested by user. STOP!.")
            return 0
        elif forward_stop_at == 'synthetics':
            result_logger.info("Only create synthetic waveforms as requested by user. STOP!.")
            return 0
        
        # -------------------------------------------------------------------------
        # (3) Check whether the misfit is lower (if the run_mode == 'pipeline')
        # ------------------------------------------------------------------------- 
        if workflow_controller.misfit_check():
            misfit_reduced = True
        else:
            workflow_controller.reupdate_model_if_misfit_not_reduced()
        do_mesh = False
    
    # -------------------------------------------------------------------------
    # (4) Check if the attempt number exceeds MAX_ATTEPTS
    # -------------------------------------------------------------------------
    if not misfit_reduced and attempt == MAX_ATTEMPTS:
        result_logger.warning("STOP: Reached max attempts without reducing misfit.")
        return 0

    # -------------------------------------------------------------------------
    # (5) Post-processing: smoothing and summing up kernels
    # -------------------------------------------------------------------------
    if start_index <= stage_order['postprocess']:
        workflow_controller.move_to_other_directory(folder_to_move='specfem')
        workflow_controller.create_misfit_kernel()
        if forward_stop_at == 'gradient':
            result_logger.info("Only compute gradient as requested by user. Stop after post-processing step.")
            
            return 0
    # -------------------------------------------------------------------------
    # (6) Inversion
    # -------------------------------------------------------------------------
    if start_index <= stage_order['inversion']:
        workflow_controller.move_to_other_directory(folder_to_move='adjointflows')
        workflow_controller.do_iteration()

        workflow_controller.cleanup_after_inversion()

        

if __name__ == '__main__':
    
    main_status = main()
    sys.exit(main_status)
