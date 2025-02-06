import logging
from pathlib import Path

def get_par_from_specfem_parfile(specfem_dir, par_name):
    """
    Read a given parameter from Par_file.
    
    Args:
        specfem_dir (str): The directory containing `DATA/Par_file`.
        par_name (str): The parameter name to search for.
    
    Returns:
        str: The value of the parameter if found.
    
    Raises:
        FileNotFoundError: If `Par_file` does not exist.
        ValueError: If the parameter is not found.
    """
    par_file_path = Path(specfem_dir) / "DATA/Par_file"

    if not par_file_path.exists():
        logging.error(f"Par_file not found: {par_file_path}")
        raise FileNotFoundError(f"Par_file not found: {par_file_path}")

    try:
        with open(par_file_path, "r") as f:
            for line in f:
                if par_name in line:
                    return line.split("=")[-1].strip()

        logging.error(f"Cannot find the parameter {par_name} in {par_file_path}")
        raise ValueError(f"Cannot find the parameter {par_name} in {par_file_path}")

    except OSError as e:
        logging.error(f"Error opening file {par_file_path}: {e}")
        raise
