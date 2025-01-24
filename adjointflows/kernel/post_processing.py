from tools.job_utils import wait_for_launching, check_path_is_correct
import logging
import subprocess

class PostProcessing:
    def __init__(self, current_model_num):
        self.current_model_num = current_model_num
    
    def sum_and_smooth_kernels(self):
        """
        Sum and smooth the kernels
        Note:
            Currently, we just call the sum_smooth_kernel.bash script to do the job.
        """
        logging.info(f"Start smoothing the kernels for the {self.current_model_num:03d} model...") 
        subprocess.run(['qsub', 'sum_smooth_kernel.bash'])
        
    