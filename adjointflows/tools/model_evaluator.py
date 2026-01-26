from .global_params import GLOBAL_PARAMS
from .dataset_loader import get_by_path
import json
import os
import logging
import pandas as pd

class ModelEvaluator:
    
    def __init__(self, current_model_num, config, dataset_config):
        self.config              = config
        self.dataset_config      = dataset_config
        self.base_dir            = GLOBAL_PARAMS['base_dir']
        # self.evlst               = f"{self.base_dir}/DATA/evlst/{config.get('data.list.evlst')}"
        self.current_model_num   = current_model_num
        self.previous_model_num  = current_model_num - 1
        self.stage_initial_model = self.config.get('setup.stage.stage_initial_model')
    
    def set_current_model_num(self, current_model_num):
        self.current_model_num = current_model_num
        
    def run_all_datasets_misfit_evaluation(self, m_num):
        """
        Run misfit evaluation for all datasets
        Return:
            total_misfit_among_datasets (float): The total average misfit among all datasets
        """
        total_misfit_among_datasets = 0.0
        total_weight = 0.0
        dataset_misfits = {}
        datasets = self.dataset_config.get("datasets", [])
        for dataset_entry in datasets:
            dataset_name = dataset_entry.get("name")
            weight = get_by_path(dataset_entry, "inversion.weight", 1.0)
            # Determine the evlst path
            evlst = get_by_path(dataset_entry, "list.evlst")
            if not evlst:
                raise ValueError("Missing list.evlst in dataset config.")
            if not os.path.isabs(evlst):
                evlst = os.path.join(self.base_dir, "DATA", "evlst", evlst)
            logging.info(f"Evaluating misfit for dataset: {dataset_name}")
            
            misfit = self.misfit_calculation(m_num, dataset_name, evlst)
            dataset_misfits[dataset_name] = misfit
            weighted_misfit = misfit * weight
            logging.info(f"Dataset: {dataset_name}, Misfit: {misfit}, Weight: {weight}, Weighted Misfit: {weighted_misfit}")
            total_misfit_among_datasets += weighted_misfit
            total_weight += weight
        self.write_dataset_misfit_summary(m_num, dataset_misfits)
        if total_weight == 0.0:
            logging.warning("Total dataset weight is 0; misfit set to 0.")
            return 0.0
        return total_misfit_among_datasets / total_weight

    def write_dataset_misfit_summary(self, m_num, dataset_misfits):
        """Write dataset misfit values to TOMO/m###/misfit_by_dataset.json."""
        if not dataset_misfits:
            logging.warning("No dataset misfits to write.")
            return
        model_dir = os.path.join(self.base_dir, "TOMO", f"m{m_num:03d}")
        os.makedirs(model_dir, exist_ok=True)
        output_path = os.path.join(model_dir, "misfit_by_dataset.json")
        try:
            with open(output_path, "w") as f:
                json.dump(dataset_misfits, f, indent=2, sort_keys=True)
        except OSError as exc:
            logging.warning(f"Failed to write {output_path}: {exc}")
            

    def misfit_calculation(self, m_num, dataset_name, evlst):
        """
        Calculate the misfit for the given model number
        Args:
            m_num (int): The model number
            dataset_name (str): The name of the dataset
            evlst (str): The path to the event list file
        Return:
            average_misfit (float): The misfit value
        """
        measure_dir = f'{self.base_dir}/TOMO/m{m_num:03d}/MEASURE_{dataset_name}/adjoints'
    
        evt_df = pd.read_csv(evlst, header=None, sep=r'\s+')
        chi_df = pd.DataFrame()
        for evt in evt_df[0]:
            adjoints_dir = f'{measure_dir}/{evt}'
            chi_file = f'{adjoints_dir}/window_chi'
            if not os.path.isfile(chi_file):
                logging.warning(f"Skip {evt}: missing {chi_file}")
                continue
            try:
                tmp_df = pd.read_csv(chi_file, header=None, sep=r'\s+')
            except Exception as exc:
                logging.warning(f"Skip {evt}: failed to read {chi_file} ({exc})")
                continue
            chi_df = pd.concat([chi_df, tmp_df])
        
        if chi_df.empty:
            logging.warning("No window_chi files found; misfit set to 0.")
            return 0.0
        chi_filtered_df = chi_df[(chi_df[28] != 0.) | (chi_df[29] != 0.)]
        win_num = len(chi_filtered_df)
        if win_num == 0:
            logging.warning("No valid windows; misfit set to 0.")
            return 0.0
        total_misfit = chi_filtered_df[28].sum()
        average_misfit = round(total_misfit / win_num, 5)
        logging.info(f'Misfit: {average_misfit}')
        return average_misfit
    
    def is_misfit_reduced(self):
        """
        Determine if the misfit is reduced.
        Note:
            If the current model is the first model in this stage (current_model_num == stage_initial_model),
            then the misfit evaluation is skipped.
        Return:
            is_reduced (bool): True for PASS, False otherwise
        """
        
        current_misfit = self.run_all_datasets_misfit_evaluation(self.current_model_num)
        if self.current_model_num == self.stage_initial_model:
            logging.info(f"SKIP the misfit evaluation: Current model number is the first model in this stage")
            logging.info(f"""Current misfit: {current_misfit}""")
            return True
        
        previous_misfit = self.run_all_datasets_misfit_evaluation(self.previous_model_num)
        if current_misfit < previous_misfit:
            logging.info(f"PASS the misfit evaluation: Current misfit: {current_misfit}, previous misfit: {previous_misfit}")
            return True
        else:
            logging.warning(f"FAIL in the misfit evaluation: Current misfit: {current_misfit}, previous misfit: {previous_misfit}")
            return False
        
