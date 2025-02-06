from .global_params import GLOBAL_PARAMS
import os
import logging
import pandas as pd

class ModelEvaluator:
    
    def __init__(self, current_model_num, config):
        self.config              = config
        self.base_dir            = GLOBAL_PARAMS['base_dir']
        self.evlst               = f"{self.base_dir}/DATA/evlst/{config.get('data.list.evlst')}"
        self.current_model_num   = current_model_num
        self.previous_model_num  = current_model_num - 1
        self.stage_initial_model = self.config.get('setup.stage.stage_initial_model')
        

    def misfit_calculation(self, m_num):
        """
        Calculate the misfit for the given model number
        Args:
            m_num (int): The model number
        Return:
            misfit (float): The misfit value
        """
        measure_dir = f'{self.base_dir}/TOMO/m{m_num:03d}/MEASURE/adjoints'
    
        evt_df = pd.read_csv(self.evlst, header=None, sep=r'\s+')
        chi_df = pd.DataFrame()
        for evt in evt_df[0]:
            adjoints_dir = f'{measure_dir}/{evt}'
            chi_file = f'{adjoints_dir}/window_chi'
            tmp_df = pd.read_csv(chi_file, header=None, sep=r'\s+')
            chi_df = pd.concat([chi_df, tmp_df])
        
        misfit = round(chi_df[28].sum() / len(chi_df), 5)
        logging.info(f'Misfit: {misfit}')
        return misfit
    
    def is_misfit_reduced(self):
        """
        Determine if the misfit is reduced.
        Note:
            If the current model is the first model in this stage (current_model_num == stage_initial_model),
            then the misfit evaluation is skipped.
        Return:
            is_reduced (bool): True for PASS, False otherwise
        """
        
        current_misfit = self.misfit_calculation(self.current_model_num)
        if self.current_model_num == self.stage_initial_model:
            logging.info(f"SKIP the misfit evaluation: Current model number is the first model in this stage")
            logging.info(f"""Current misfit: {current_misfit}""")
            return True
        
        previous_misfit = self.misfit_calculation(self.previous_model_num)
        if current_misfit < previous_misfit:
            logging.info(f"PASS the misfit evaluation: Current misfit: {current_misfit}, previous misfit: {previous_misfit}")
            return True
        else:
            logging.warning(f"FAIL in the misfit evaluation: Current misfit: {current_misfit}, previous misfit: {previous_misfit}")
            return False
        