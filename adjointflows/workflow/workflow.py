from tools import FileManager, ModelEvaluator
from tools.job_utils import remove_file, wait_for_launching
from kernel import ModelGenerator, ForwardGenerator, PostProcessing
from iterate import IterationProcess, StepLengthOptimizer
import os
import sys
import logging


class WorkflowController:
    def __init__(self, config, global_params):
        self.config = config
        self.base_dir = global_params['base_dir']
        self.adjointflows_dir = os.path.join(self.base_dir, 'adjointflows')
        self.specfem_dir = os.path.join(self.base_dir, 'specfem3d')
        self.flexwin_dir = os.path.join(self.base_dir, 'flexwin')
        self.iterate_dir = os.path.join(self.base_dir, 'iterate_inv')
        self.current_model_num   = self.config.get('setup.model.current_model_num')
        self.stage_initial_model = self.config.get('setup.stage.stage_initial_model')
        self.ichk                = self.config.get('preprocessing.ICHK')
        self.max_fail            = self.config.get('inversion.max_fail')
        
        self.debug_logger = logging.getLogger("debug_logger")
        
    def construct_misfit_list(self):
        """
        Construct a list to store the misfit values
        The first element is the misfit of previous model
        This is for the L-BFGS method
        """
        model_evaluator_tmp = ModelEvaluator(current_model_num=self.current_model_num, config=self.config)
        previous_misfit = model_evaluator_tmp.misfit_calculation(m_num=self.current_model_num - 1)
        self.misfit_list = [previous_misfit]
    
    def construct_step_length_list(self):
        """
        Construct a list to store the step length values
        The first element is 1
        This is for the L-BFGS method
        """
        self.step_length_list = [1.0]

    def setup_dir(self):
        """
        Setup files and directories for the following adjoint tomography processes
        """
        remove_flag = lambda num: num == 0
        clear_dir_flag = remove_flag(self.ichk)
        # if the L-BFGS fail number is NOT 0, remove the directories
        if self.lbfgs_fail_num != 0:
            clear_dir_flag = True
        
        file_manager = FileManager()
        file_manager.set_model_number(current_model_num=self.current_model_num)
        file_manager.setup_directory(clear_directories=clear_dir_flag)
        file_manager.make_symbolic_links()
    
    def setup_for_fail(self):
        """
        Setup misfit list and fail_num for preparing the situation 
        when the misfit is not reduced
        """
        self.reset_fail_num()

        if self.current_model_num != self.stage_initial_model:
            self.construct_misfit_list()
            self.construct_step_length_list()
        
    def generate_model(self, mesh_flag):
        """
        Prepare the model for the forward simulation
        """
        model_generator = ModelGenerator()
        model_generator.model_setup(mesh_flag=mesh_flag)
        
    def run_forward(self):
        """
        Run the adjoint tomography processes
        """
        forward_generator = ForwardGenerator(current_model_num=self.current_model_num, config=self.config)        
        forward_generator.preprocessing()
        forward_generator.output_vars_file()
        index_evt_last = forward_generator.check_last_event()
        forward_generator.process_each_event(index_evt_last)
        
    def run_forward_for_tuning_flexwin(self, do_forward):
        """
        Run the adjoint tomography processes
        """
        forward_generator = ForwardGenerator(current_model_num=self.current_model_num, config=self.config)        
        forward_generator.preprocessing()
        forward_generator.output_vars_file()
        index_evt_last = forward_generator.check_last_event()
        forward_generator.process_each_event_for_tuning_flexwin(index_evt_last=index_evt_last, do_forward=do_forward)
        
        
    def misfit_check(self):
        model_evaluator = ModelEvaluator(current_model_num=self.current_model_num, config=self.config)
        misfit = model_evaluator.misfit_calculation(m_num=self.current_model_num)
        
        if self.current_model_num != self.stage_initial_model:
            self.misfit_list.append(misfit)
        
        is_misfit_reduced = model_evaluator.is_misfit_reduced()
        
        if not is_misfit_reduced:
            if self.stage_initial_model == self.current_model_num - 1: 
                error_message = "STOP: [Steepest Descent] Misfit is not reduced!"
                self.debug_logger.error(error_message)
                raise ValueError(error_message)
            else:
                self.debug_logger.warning("[L-BFGS] Misfit is not reduced. Rollback the model.")
                self.add_fail_num()
                return False
        else:
            self.debug_logger.info("PASS: Misfit is reduced.")
            return True
    
    def create_misfit_kernel(self):
        """
        Sum up the event kernel and smooth it
        """
        post_processing = PostProcessing(current_model_num=self.current_model_num, config=self.config)
        post_processing.sum_and_smooth_kernels()
    
    def do_iteration(self):
        """
        Then do the preconditioning and calculate the direction
        """
        iteration_process = IterationProcess(current_model_num=self.current_model_num, config=self.config)
        iteration_process.save_params_json()
        iteration_process.hess_times_kernel()
        
        steplength_optimizer = StepLengthOptimizer(current_model_num=self.current_model_num, config=self.config)
        
        # --------------------------------------
        # Use Steepest Descent for inversion 
        # --------------------------------------
        if self.stage_initial_model == self.current_model_num:
            iteration_process.calculate_direction_sd()
            steplength_optimizer.run_line_search()
            step_fac = steplength_optimizer.get_current_best_step_length()
            iteration_process.update_model(step_fac=step_fac, lbfgs_flag=False)
        # --------------------------------------
        # Use L-BFGS for inversion 
        # --------------------------------------    
        else:
            step_fac = 1.
            iteration_process.calculate_direction_lbfgs()

            iteration_process.update_model(step_fac=step_fac, lbfgs_flag=True)
            
    def reupdate_model_if_misfit_not_reduced(self):
        
        rollback_model = self.current_model_num - 1
        re_iteration_process = IterationProcess(current_model_num=rollback_model, config=self.config)
        re_iteration_process.save_params_json()
        
        re_steplength_optimizer = StepLengthOptimizer(current_model_num=rollback_model, config=self.config)
        # --------------------------------------------------------------------------------------------------
        # take 3 (or 2 if the fail_num is 1) misfit values for finding a better step length
        # take 2 (or 1 if the fail_num is 1) step length values for finding a better step length
        # --------------------------------------------------------------------------------------------------
        misfit_selected = self.misfit_list[-3:]
        step_selected = self.step_length_list[-2:]
        g_dot_p = re_iteration_process.get_gradient_info(param_name='g_dot_p')
        re_iteration_process.save_polynomial_to_json(misfit_list = self.misfit_list, 
                                                     step_list = self.step_length_list, 
                                                     g_dot_p = g_dot_p)
        if self.lbfgs_fail_num == 1:
            new_step_length = re_steplength_optimizer.quadratic_interpolation(phi_o = misfit_selected[0], 
                                                                           phi0  = misfit_selected[1], 
                                                                           alpha = step_selected[0],
                                                                           phi_grad = g_dot_p)
        else:
            new_step_length = re_steplength_optimizer.cubic_interpolation(phi_o = misfit_selected[0], 
                                                                       phi0  = misfit_selected[1],
                                                                       phi1  = misfit_selected[2], 
                                                                       alpha0 = step_selected[0],
                                                                       alpha1 = step_selected[1],
                                                                       phi_grad = g_dot_p)        
        self.step_length_list.append(new_step_length)
        re_iteration_process.update_model(step_fac=new_step_length, lbfgs_flag=True)
        
        
        
    def move_to_other_directory(self, folder_to_move):
        """
        Move to the specified directory
        Args:
            folder_to_move (str): The folder name you want to move
        """
        folder_mapping = {
            'adjointflows': self.adjointflows_dir,
            'specfem': self.specfem_dir,
            'flexwin': self.flexwin_dir,
            'iterate': self.iterate_dir
        }
        if not folder_to_move in folder_mapping:
            error_message = f"Unknown process type: {folder_to_move}"
            self.debug_logger.error(error_message)
            raise ValueError(error_message)
        
        target_folder = folder_mapping[folder_to_move]
        try:
            os.chdir(target_folder)
            self.debug_logger.info(f'WORKDIR CHANGE: Moved to {folder_to_move} directory')
        except FileNotFoundError:
            error_message = f"Target folder does not exist: {target_folder}"
            self.debug_logger.error(error_message)
            raise FileNotFoundError(error_message)
            
    def reset_fail_num(self):
        self.lbfgs_fail_num = 0
        
    def add_fail_num(self):
        self.lbfgs_fail_num += 1
    