from tools import FileManager, ModelEvaluator
from tools.job_utils import remove_file, wait_for_launching
from kernel import ModelGenerator, ForwardGenerator, PostProcessing
import yaml
import os
import sys
import logging


class WorkflowController:
    def __init__(self, config, global_params):
        self.config = config
        self.base_dir = global_params['base_dir']
        self.specfem_dir = os.path.join(self.base_dir, 'specfem3d')
        self.flexwin_dir = os.path.join(self.base_dir, 'flexwin')
        self.iterate_dir = os.path.join(self.base_dir, 'iterate_inv')
        self.current_model_num = self.config.get('setup.model.current_model_num')
    
    def setup(self):
        """
        Setup files and directories for the following adjoint tomography processes
        """
        file_manager = FileManager()
        file_manager.set_model_number(current_model_num=self.current_model_num)
        file_manager.setup_directory(clear_directories=True)
        file_manager.make_symbolic_links()
    
    def generate_model(self):
        """
        Prepare the model for the forward simulation
        """
        model_generator = ModelGenerator(current_model_num=self.current_model_num)
        model_generator.model_setup(mesh_flag=True)
        
    def run_forward(self):
        """
        Run the adjoint tomography processes
        """
        forward_generator = ForwardGenerator(current_model_num=self.current_model_num, config=self.config)        
        forward_generator.do_forward()
        
    
    def misfit_check(self):
        model_evaluator = ModelEvaluator(current_model_num=self.current_model_num, config=self.config)
        is_misfit_reduced = model_evaluator.is_misfit_reduced()
        
        if not is_misfit_reduced:
            error_message = "STOP: Misfit is not reduced!"
            logging.error(error_message)
            raise ValueError(error_message)
    
    def create_misfit_kernel(self):
        """
        Sum up the event kernel and smooth it
        """
        finish_signal_file = f'{self.iterate_dir}/model_gradient_ready'
        
        post_processing = PostProcessing(current_model_num=self.current_model_num)
        remove_file(finish_signal_file)
        post_processing.sum_and_smooth_kernels()
        wait_for_launching(check_file=finish_signal_file,
                           message='ready to launch inversion!\n')
    
        
        

    
    def move_to_other_directory(self, folder_to_move):
        """
        Move to the specified directory
        Args:
            folder_to_move (str): The folder name you want to move
        """
        folder_mapping = {
            'specfem': self.specfem_dir,
            'flexwin': self.flexwin_dir,
            'iterate': self.iterate_dir
        }
        if not folder_to_move in folder_mapping:
            error_message = f"Unknown process type: {folder_to_move}"
            logging.error(error_message)
            raise ValueError(error_message)
        
        target_folder = folder_mapping[folder_to_move]
        try:
            os.chdir(target_folder)
            logging.info(f'WORKDIR CHANGE: Moved to {folder_to_move} directory')
        except FileNotFoundError:
            error_message = f"Target folder does not exist: {target_folder}"
            logging.error(error_message)
            raise FileNotFoundError(error_message)
            
        