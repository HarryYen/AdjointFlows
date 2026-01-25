from .job_utils import copy_files, clean_and_initialize_directories, remove_path, remove_files_with_pattern
from .global_params import GLOBAL_PARAMS
import os
import shutil
import logging

# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FileManager:
    
    def __init__(self):
        self.base_dir = GLOBAL_PARAMS['base_dir']
        self.current_model_dir = None
        self.next_model_dir = None
        
        self.debug_logger = logging.getLogger("debug_logger")
    
    
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
                    
            
    def setup_directory(self):
        """
        Create a series of directories for inversion.
        it will call clean_and_initialize_directories to remove all files in the directories
        if the clear_directories is True.
        Args:
            clear_directories (bool): If True, remove all files in the directories
        """
        if not self.current_model_dir or not self.next_model_dir:
            message = "Model numbers not set. Please call set_model_number before setup_directory."
            self.debug_logger.error(message)
            raise ValueError(message)
        
        dirs = [
            f"{self.current_model_dir}/DATABASES_MPI",
            f"{self.current_model_dir}/MOD",
            f"{self.next_model_dir}"
    ]
    
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)
    

    def ensure_dataset_dirs(self, dataset_name):
        """Create dataset-specific directories for the current model."""
        if not self.current_model_dir:
            message = "Model numbers not set. Please call set_model_number before ensure_dataset_dirs."
            self.debug_logger.error(message)
            raise ValueError(message)

        kernel_base = f"{self.current_model_dir}/KERNEL_{dataset_name}"
        measure_base = f"{self.current_model_dir}/MEASURE_{dataset_name}"
        syn_dir = f"{self.current_model_dir}/SYN_{dataset_name}"

        dirs = [
            syn_dir,
            f"{measure_base}/windows",
            f"{measure_base}/adjoints",
            f"{kernel_base}/SMOOTH",
            f"{kernel_base}/SUM",
            f"{kernel_base}/VTK",
            f"{kernel_base}/PRECOND",
            f"{kernel_base}/UPDATE",
            f"{kernel_base}/DATABASE",
        ]
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)

    def link_dataset_dirs(self, dataset_name, data_waveform_dir):
        """Relink dataset-specific directories to standard paths.

        Args:
            dataset_name (str): Dataset name used for suffixing TOMO directories.
            data_waveform_dir (str): Directory under DATA/ to link as DATA/wav.
        """
        if not self.current_model_dir:
            message = "Model numbers not set. Please call set_model_number before link_dataset_dirs."
            self.debug_logger.error(message)
            raise ValueError(message)
        if not data_waveform_dir:
            raise ValueError("data.waveform_dir is required but missing.")
        if os.path.normpath(data_waveform_dir) == "wav":
            raise ValueError("data.waveform_dir cannot be 'wav'; it conflicts with DATA/wav symlink.")

        links = {
            f"{self.current_model_dir}/SYN": f"{self.current_model_dir}/SYN_{dataset_name}",
            f"{self.current_model_dir}/MEASURE": f"{self.current_model_dir}/MEASURE_{dataset_name}",
            f"{self.current_model_dir}/KERNEL": f"{self.current_model_dir}/KERNEL_{dataset_name}",
        }
        data_target = os.path.join(self.base_dir, "DATA", data_waveform_dir)
        if not os.path.isdir(data_target):
            raise FileNotFoundError(f"Data waveform dir not found: {data_target}")
        links[f"{self.base_dir}/DATA/wav"] = data_target
        remove_path(list(links.keys()))
        for link, target in links.items():
            os.symlink(target, link)

    def clear_dataset_dirs(self, dataset_name, clear_syn=False, clear_measure=True, clear_kernel=True):
        """Clear dataset-specific directories."""
        if not self.current_model_dir:
            message = "Model numbers not set. Please call set_model_number before clear_dataset_dirs."
            self.debug_logger.error(message)
            raise ValueError(message)

        dirs = []
        if clear_syn:
            dirs.append(f"{self.current_model_dir}/SYN_{dataset_name}")
        if clear_measure:
            dirs.extend([
                f"{self.current_model_dir}/MEASURE_{dataset_name}/windows",
                f"{self.current_model_dir}/MEASURE_{dataset_name}/adjoints",
            ])
        if clear_kernel:
            dirs.extend([
                f"{self.current_model_dir}/KERNEL_{dataset_name}/SMOOTH",
                f"{self.current_model_dir}/KERNEL_{dataset_name}/SUM",
                f"{self.current_model_dir}/KERNEL_{dataset_name}/VTK",
                f"{self.current_model_dir}/KERNEL_{dataset_name}/PRECOND",
                f"{self.current_model_dir}/KERNEL_{dataset_name}/UPDATE",
                f"{self.current_model_dir}/KERNEL_{dataset_name}/DATABASE",
            ])
        if dirs:
            clean_and_initialize_directories(dirs)
        
    def make_symbolic_links(self):
        """
        We need to create the symbolic links for the SEM simulation.
        Here, it will link 
        """
        if not self.current_model_dir or not self.next_model_dir:
            message = "Model numbers not set. Please call set_model_number before setup_directory."
            self.debug_logger.error(message)
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
        
        remove_path(link_directories)
        
        if len(link_directories) != len(target_directories):
            message = "The length of link_directories and target_directories should be the same"
            self.debug_logger.error(message)
            raise ValueError(message)
        
        for index in range(len(target_directories)):
            target = target_directories[index]
            link = link_directories[index]
            os.symlink(target, link)

    def remove_files_after_inversion(self):
        
        remove_files_with_pattern(f'{self.current_model_dir}/DATABASES_MPI/*vtk')
        remove_files_with_pattern(f'{self.current_model_dir}/DATABASES_MPI/*absorb_field.bin')
