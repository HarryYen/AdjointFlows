from tools import FileManager, ModelEvaluator
from tools.job_utils import remove_file, wait_for_launching, copy_files
from tools.dataset_loader import load_dataset_config, get_by_path, deep_merge
from kernel import ModelGenerator, ForwardGenerator, PostProcessing
from iterate import IterationProcess, StepLengthOptimizer
import os
import sys
import logging
import json
import yaml

class WorkflowController:
    def __init__(self, config, global_params):
        self.config = config
        self.base_dir = global_params['base_dir']
        self.adjointflows_dir = os.path.join(self.base_dir, 'adjointflows')
        self.specfem_dir = os.path.join(self.base_dir, 'specfem3d')
        self.flexwin_dir = os.path.join(self.base_dir, 'flexwin')
        self.iterate_dir = os.path.join(self.base_dir, 'iterate_inv')
        self.current_model_num   = int(self.config.get('setup.model.current_model_num'))
        self.stage_initial_model = int(self.config.get('setup.stage.stage_initial_model'))
        self.ichk                = int(self.config.get('preprocessing.ICHK'))
        self.max_fail            = int(self.config.get('inversion.max_fail'))
        self.max_model_update    = self.config.get('inversion.max_model_update')
        self.sd_runs_num         = int(self.config.get('inversion.sd_runs_num'))
        self.do_backtracking_ls  = int(self.config.get('inversion.do_backtracking_line_search'))
        
        self.precondition_flag   = bool(self.config.get('inversion.precondition_flag'))

        self.tomo_dir     = os.path.join(self.base_dir, 'TOMO', f'm{self.current_model_num:03d}')

        self.debug_logger = logging.getLogger("debug_logger")


        # initialize file_manager
        self.file_manager = FileManager()
        self.file_manager.set_model_number(current_model_num=self.current_model_num)
        self.determine_inversion_method()

        self.dataset_config = load_dataset_config(self.adjointflows_dir, logger=self.debug_logger)

        self.setup_dir()
        self.iteration_process = IterationProcess(current_model_num=self.current_model_num, config=self.config)
        # self.iteration_process.save_params_json()

    def determine_inversion_method(self):
        """
        Determine the inversion method based on the current model number
        """
        if self.current_model_num < self.stage_initial_model + self.sd_runs_num:
            self.inversion_method = 'SD'
        else:
            self.inversion_method = 'LBFGS'
            self.setup_for_fail()

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
        The first element is the step length when the current model was updated
        This is for the L-BFGS method
        """
        json_path = os.path.join(self.adjointflows_dir, 'step_length.json')
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            first_step_length = data.get('step_length')
        except FileNotFoundError:
            warning_message = f"File not found: {json_path}, we just use 1.0"
            self.debug_logger.error(warning_message)
            first_step_length = 1.0
        
        self.step_length_list = [first_step_length]

    def setup_dir(self):
        """
        Setup files and directories for the following adjoint tomography processes
        """
        self.file_manager.setup_directory()
        self.file_manager.make_symbolic_links()
    
    def setup_for_fail(self):
        """
        Setup misfit list and fail_num for preparing the situation 
        when the misfit is not reduced
        """
        self.reset_fail_num()

        if self.inversion_method == 'LBFGS':
            self.construct_misfit_list()
            self.construct_step_length_list()
        
    def generate_model(self, mesh_flag):
        """
        Prepare the model for the forward simulation
        """
        model_generator = ModelGenerator()
        model_generator.model_setup(mesh_flag=mesh_flag)
        
        self.iteration_process.update_specfem_params()
        self.iteration_process.save_params_json()

    def load_specfem_params_without_generation(self):
        """
        Load specfem parameters from existing files without generating a model.
        """
        self.debug_logger.info("Skipping model generation; loading specfem parameters from existing files.")
        self.iteration_process.update_specfem_params()
        self.iteration_process.save_params_json()

    def write_dataset_config_file(self, dataset_config):
        """Write a dataset-specific config file for FLEXWIN/MEASURE scripts."""
        dataset_name = get_by_path(dataset_config, "name", default="dataset")
        out_dir = os.path.join(self.adjointflows_dir, ".dataset_configs")
        os.makedirs(out_dir, exist_ok=True)
        config_path = os.path.join(out_dir, f"{dataset_name}.yaml")

        config_data = {
            "source": {
                "type": get_by_path(dataset_config, "source.type", default="cmt"),
                "force": {
                    "depth_km": get_by_path(dataset_config, "source.force.depth_km", default=0.0),
                },
            },
            "data": {
                "list": {
                    "evlst": get_by_path(dataset_config, "list.evlst", default=self.config.get("data.list.evlst")),
                    "stlst": get_by_path(dataset_config, "list.stlst", default=self.config.get("data.list.stlst")),
                    "evchk": get_by_path(dataset_config, "list.evchk", default=self.config.get("data.list.evchk")),
                },
                "seismogram": {
                    "tbeg": get_by_path(dataset_config, "seismogram.tbeg", default=self.config.get("data.seismogram.tbeg")),
                    "tend": get_by_path(dataset_config, "seismogram.tend", default=self.config.get("data.seismogram.tend")),
                    "tcor": get_by_path(dataset_config, "seismogram.tcor", default=self.config.get("data.seismogram.tcor")),
                    "dt": get_by_path(dataset_config, "seismogram.dt", default=self.config.get("data.seismogram.dt")),
                    "filter": {
                        "P1": get_by_path(dataset_config, "seismogram.filter.P1", default=self.config.get("data.seismogram.filter.P1")),
                        "P2": get_by_path(dataset_config, "seismogram.filter.P2", default=self.config.get("data.seismogram.filter.P2")),
                    },
                    "component": {
                        "COMP": get_by_path(dataset_config, "seismogram.component.COMP", default=self.config.get("data.seismogram.component.COMP")),
                        "EN2RT": get_by_path(dataset_config, "seismogram.component.EN2RT", default=self.config.get("data.seismogram.component.EN2RT")),
                    },
                },
            },
        }

        with open(config_path, "w") as f:
            yaml.safe_dump(config_data, f, sort_keys=False)
        return config_path

    def run_all_datasets(self, do_adjoint, do_measurement):
        """
        Run all datasets defined in dataset.yaml
        """
        default_settings = self.dataset_config.get("defaults", {})
        datasets = self.dataset_config.get("datasets", [])
        for dataset_entry in datasets:
            dataset_name = dataset_entry.get("name")
            if not dataset_name:
                self.debug_logger.error("Dataset entry missing 'name'. Skipping this dataset.")
                continue
            
            self.debug_logger.info(f"Processing dataset: {dataset_name}")
            # Merge default settings with dataset-specific settings
            merged_dataset = deep_merge(default_settings, dataset_entry)
            self.run_forward(merged_dataset, do_adjoint, do_measurement)
        
            

    def run_forward(self, dataset_config, do_adjoint, do_measurement):
        """
        Run the adjoint tomography processes
        """
        dataset_name = get_by_path(dataset_config, "name", default="dataset")
        do_forward = bool(get_by_path(dataset_config, "synthetics.do_wave_simulation", default=1))
        data_waveform_dir = get_by_path(dataset_config, "data.waveform_dir")
        syn_waveform_dir = get_by_path(dataset_config, "synthetics.waveform_dir")
        self.file_manager.ensure_dataset_dirs(dataset_name, syn_waveform_dir=syn_waveform_dir)
        self.file_manager.link_dataset_dirs(
            dataset_name,
            data_waveform_dir,
            syn_waveform_dir=syn_waveform_dir,
        )
        if not self.ichk:
            self.file_manager.clear_dataset_dirs(
                dataset_name,
                syn_waveform_dir=syn_waveform_dir,
                clear_syn=do_forward,
                clear_syn_intermediate=not do_forward,
                clear_measure=True,
                clear_kernel=True,
            )
        dataset_config_path = self.write_dataset_config_file(dataset_config)
        forward_generator = ForwardGenerator(
            current_model_num=self.current_model_num,
            config=self.config,
            dataset_config=dataset_config,
            dataset_config_path=dataset_config_path,
        )
        forward_generator.preprocessing()
        forward_generator.output_vars_file()
        index_evt_last = forward_generator.check_last_event()
        forward_generator.process_each_event(index_evt_last, do_forward, do_adjoint, do_measurement)
        
    def run_forward_for_tuning_flexwin(self, do_forward):
        """
        Run the adjoint tomography processes
        """
        forward_generator = ForwardGenerator(current_model_num=self.current_model_num, config=self.config)        
        forward_generator.preprocessing()
        forward_generator.output_vars_file()
        # index_evt_last = forward_generator.check_last_event()
        forward_generator.process_each_event_for_tuning_flexwin(index_evt_last=0, do_forward=do_forward)
        
        
    def misfit_check(self):
        model_evaluator = ModelEvaluator(current_model_num=self.current_model_num, config=self.config)
        misfit = model_evaluator.misfit_calculation(m_num=self.current_model_num)
        
        if self.inversion_method == 'LBFGS':
            self.misfit_list.append(misfit)
        
        is_misfit_reduced = model_evaluator.is_misfit_reduced()
        
        if not is_misfit_reduced:
            if self.stage_initial_model >= self.current_model_num - self.sd_runs_num: 
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
        post_processing.sum_and_smooth_kernels(precond_flag=self.precondition_flag)
    
    def do_iteration(self):
        """
        Then do the preconditioning and calculate the direction
        """
        # iteration_process = IterationProcess(current_model_num=self.current_model_num, config=self.config)
        # iteration_process.save_params_json()
        # iteration_process.hess_times_kernel()
        steplength_optimizer = StepLengthOptimizer(current_model_num=self.current_model_num, config=self.config)
        
        # --------------------------------------
        # Use Steepest Descent for inversion 
        # --------------------------------------
        if (self.stage_initial_model == self.current_model_num) or (self.inversion_method == 'SD'):
            self.iteration_process.calculate_direction_sd(precond_flag=self.precondition_flag)
            steplength_optimizer.run_line_search()
            step_fac = steplength_optimizer.get_current_best_step_length()
            self.iteration_process.update_model(step_fac=step_fac, lbfgs_flag=False)
        # --------------------------------------
        # Use L-BFGS for inversion 
        # --------------------------------------    
        else:
            self.iteration_process.calculate_direction_lbfgs(precond_flag=self.precondition_flag)
            step_fac = self.iteration_process.adjust_step_length_by_minmax(max_update_amount=self.max_model_update)
            
            if self.do_backtracking_ls:
                # do a backtracking line search for finding a better step length
                steplength_optimizer.run_backtracking_line_search(step_length_init=step_fac)
                step_fac = steplength_optimizer.get_current_best_step_length()

            self.iteration_process.save_last_step_length_to_json(step_fac)
            self.iteration_process.update_model(step_fac=step_fac, lbfgs_flag=True)
            
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
    

    def cleanup_after_inversion(self):
        self.file_manager.remove_files_after_inversion()
        copy_files(src_dir = self.adjointflows_dir,
                   dst_dir = self.tomo_dir, 
                   pattern = 'params.json')
        

    
