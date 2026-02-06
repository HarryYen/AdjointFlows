from .job_utils import copy_files, clean_and_initialize_directories, remove_path, remove_files_with_pattern
from .global_params import GLOBAL_PARAMS
from pathlib import Path
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
    

    def _resolve_syn_dir(self, dataset_name, syn_waveform_dir):
        syn_dir_name = syn_waveform_dir or f"SYN_{dataset_name}"
        syn_dir_norm = os.path.normpath(syn_dir_name)
        if os.path.isabs(syn_dir_norm):
            if os.path.normpath(syn_dir_norm) == os.path.normpath(os.path.join(self.current_model_dir, "SYN")):
                raise ValueError("synthetics.waveform_dir cannot be the same as the SYN symlink.")
            return syn_dir_norm

        first_segment = syn_dir_norm.split(os.sep)[0]
        if first_segment == "SYN":
            raise ValueError(
                "synthetics.waveform_dir cannot start with 'SYN'; it conflicts with the SYN symlink."
            )
        return os.path.join(self.current_model_dir, syn_dir_name)

    def _resolve_tool_path(self, tool_dir, source_path, default_name=None):
        if not source_path:
            return None
        source = Path(source_path).expanduser()
        if not source.is_absolute():
            source = Path(tool_dir) / source
        if source.is_dir() and default_name:
            source = source / default_name
        return str(source)

    def _link_tool_file(self, target_path, source_path, label=None):
        if not source_path:
            return
        target = Path(target_path)
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Tool file not found: {source}")
        if target.exists() and not target.is_symlink():
            if target.is_dir():
                raise ValueError(f"{label or 'Target'} is a directory; cannot replace: {target}")
            try:
                if os.path.realpath(source) == os.path.realpath(target):
                    return
            except OSError:
                pass
            self.debug_logger.info(f"{label or 'Target'} exists; replacing with symlink: {target}")
            target.unlink()
        if target.is_symlink():
            try:
                if os.path.realpath(source) == os.path.realpath(target):
                    return
            except OSError:
                pass
            target.unlink()
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source, target)

    def link_measurement_tools(
        self,
        flexwin_bin=None,
        flexwin_par=None,
        measure_adj_bin=None,
        measure_adj_par=None,
    ):
        flexwin_dir = os.path.join(self.base_dir, "flexwin")
        measure_dir = os.path.join(self.base_dir, "measure_adj")

        flexwin_bin_path = self._resolve_tool_path(flexwin_dir, flexwin_bin, default_name="flexwin")
        flexwin_par_path = self._resolve_tool_path(flexwin_dir, flexwin_par)
        measure_bin_path = self._resolve_tool_path(measure_dir, measure_adj_bin, default_name="measure_adj")
        measure_par_path = self._resolve_tool_path(measure_dir, measure_adj_par)

        self._link_tool_file(
            os.path.join(flexwin_dir, "flexwin"),
            flexwin_bin_path,
            label="flexwin binary",
        )
        self._link_tool_file(
            os.path.join(flexwin_dir, "PAR_FILE"),
            flexwin_par_path,
            label="flexwin PAR_FILE",
        )
        self._link_tool_file(
            os.path.join(measure_dir, "measure_adj"),
            measure_bin_path,
            label="measure_adj binary",
        )
        self._link_tool_file(
            os.path.join(measure_dir, "MEASUREMENT.PAR"),
            measure_par_path,
            label="measure_adj MEASUREMENT.PAR",
        )

    def ensure_dataset_dirs(self, dataset_name, syn_waveform_dir=None):
        """Create dataset-specific directories for the current model."""
        if not self.current_model_dir:
            message = "Model numbers not set. Please call set_model_number before ensure_dataset_dirs."
            self.debug_logger.error(message)
            raise ValueError(message)

        kernel_base = f"{self.current_model_dir}/KERNEL_{dataset_name}"
        measure_base = f"{self.current_model_dir}/MEASURE_{dataset_name}"
        syn_dir = self._resolve_syn_dir(dataset_name, syn_waveform_dir)

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

    def link_dataset_dirs(self, dataset_name, data_waveform_dir, syn_waveform_dir=None):
        """Relink dataset-specific directories to standard paths.

        Args:
            dataset_name (str): Dataset name used for suffixing TOMO directories.
            data_waveform_dir (str): Directory under DATA/ to link as DATA/wav.
            syn_waveform_dir (str): Directory name used as the SYN target.
        """
        if not self.current_model_dir:
            message = "Model numbers not set. Please call set_model_number before link_dataset_dirs."
            self.debug_logger.error(message)
            raise ValueError(message)
        if not data_waveform_dir:
            raise ValueError("data.waveform_dir is required but missing.")
        if os.path.normpath(data_waveform_dir) == "wav":
            raise ValueError("data.waveform_dir cannot be 'wav'; it conflicts with DATA/wav symlink.")

        syn_dir = self._resolve_syn_dir(dataset_name, syn_waveform_dir)
        links = {
            f"{self.current_model_dir}/SYN": syn_dir,
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

    def clear_syn_intermediate_files(self, syn_dir):
        """Remove SAC and TOMO files from a synthetic directory tree."""
        if not os.path.isdir(syn_dir):
            return
        for pattern in ("*.sac", "*.tomo"):
            for file in Path(syn_dir).rglob(pattern):
                try:
                    file.unlink()
                except Exception as e:
                    self.debug_logger.warning(f"Failed to remove {file}: {e}")

    def clear_dataset_dirs(
        self,
        dataset_name,
        syn_waveform_dir=None,
        clear_syn=False,
        clear_syn_intermediate=False,
        clear_measure=True,
        clear_kernel=True,
    ):
        """Clear dataset-specific directories."""
        if not self.current_model_dir:
            message = "Model numbers not set. Please call set_model_number before clear_dataset_dirs."
            self.debug_logger.error(message)
            raise ValueError(message)

        dirs = []
        syn_dir = self._resolve_syn_dir(dataset_name, syn_waveform_dir)
        if clear_syn:
            dirs.append(syn_dir)
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
        if clear_syn_intermediate and not clear_syn:
            self.clear_syn_intermediate_files(syn_dir)
        
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
