from .job_utils import clean_and_initialize_directories
from .global_params import GLOBAL_PARAMS
import os
import shutil
import logging

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")

class FileManager:
    
    def __init__(self):
        self.base_dir = GLOBAL_PARAMS['base_dir']
        self.current_model_dir = None
        self.next_model_dir = None
    
    
    def set_model_number(self, current_model_num):
        """
        Import the current model number and set the current and next model directories.
        Args:
            current_model_num (int): The current model number
        """
        mcur_dir = f"m{current_model_num:03d}"
        mnxt_dir = f"m{current_model_num + 1:03d}"
        self.current_model_dir = os.path.join(self.base_dir, 'TOMO', mcur_dir)
        self.next_model_dir    = os.path.join(self.base_dir, 'TOMO', mnxt_dir)
    
    
    def remove_path(self, directories):
        """
        Remove the specified directories or files, including symbolic links.
        Args:
            directories (list): A list of directories or files to remove.
        """
        for directory in directories:
            try:
                if not os.path.exists(directory):
                    logging.warning(f"Path does not exist: {directory}")
                    continue
                
                if os.path.islink(directory): 
                    os.unlink(directory)
                    logging.debug(f"Unlinked symbolic link: {directory}")
                elif os.path.isdir(directory): 
                    shutil.rmtree(directory)
                    logging.debug(f"Removed directory: {directory}")
                elif os.path.isfile(directory): 
                    os.remove(directory)
                    logging.debug(f"Removed file: {directory}")
                else:
                    logging.warning(f"Unknown path type, skipped: {directory}")
            except Exception as e:
                logging.error(f"Failed to remove {directory}: {e}")
                    
            
    def setup_directory(self, clear_directories=True):
        """
        Create a series of directories for inversion.
        it will call clean_and_initialize_directories to remove all files in the directories
        if the clear_directories is True.
        Args:
            clear_directories (bool): If True, remove all files in the directories
        """
        if not self.current_model_dir or not self.next_model_dir:
            message = "Model numbers not set. Please call set_model_number before setup_directory."
            logging.error(message)
            raise ValueError(message)
        
        dirs = [
            f"{self.current_model_dir}/DATABASES_MPI",
            f"{self.current_model_dir}/SYN",
            f"{self.current_model_dir}/MEASURE/windows",
            f"{self.current_model_dir}/MEASURE/adjoints",
            f"{self.current_model_dir}/KERNEL/SMOOTH",
            f"{self.current_model_dir}/KERNEL/SUM",
            f"{self.current_model_dir}/KERNEL/VTK",
            f"{self.current_model_dir}/KERNEL/PRECOND",
            f"{self.current_model_dir}/KERNEL/DATABASE",
            f"{self.current_model_dir}/MOD",
            f"{self.next_model_dir}"
    ]
    
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)
        
        if clear_directories:
            clean_and_initialize_directories(dirs[1:])
        
    def make_symbolic_links(self):
        """
        We need to create the symbolic links for the SEM simulation.
        Here, it will link 
        """
        if not self.current_model_dir or not self.next_model_dir:
            message = "Model numbers not set. Please call set_model_number before setup_directory."
            logging.error(message)
            raise ValueError(message)
        
        specfem_dir = os.path.join(self.base_dir, 'specfem3d')
        link_directories = [
            f'{specfem_dir}/DATABASES_MPI', 
            f'{specfem_dir}/KERNEL', 
            f'{specfem_dir}/MOD',
            f'{self.base_dir}/SYN', 
            f'{self.base_dir}/flexwin/PACK', 
            f'{self.base_dir}/measure_adj/PACK'
        ]
        target_directories = [
            f"{self.current_model_dir}/DATABASES_MPI",
            f"{self.current_model_dir}/KERNEL",
            f"{self.current_model_dir}/MOD",
            f"{self.current_model_dir}/SYN",
            f"{self.current_model_dir}/MEASURE/windows",
            f"{self.current_model_dir}/MEASURE/adjoints",
        ]
        
        self.remove_path(link_directories)
        
        if len(link_directories) != len(target_directories):
            message = "The length of link_directories and target_directories should be the same"
            logging.error(message)
            raise ValueError(message)
        
        for index in range(len(target_directories)):
            target = target_directories[index]
            link = link_directories[index]
            os.symlink(target, link)