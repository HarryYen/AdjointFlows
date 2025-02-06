from pathlib import Path
import time
import os
import logging
import shutil
import glob


def wait_for_launching(check_file, message):
    """
    Wait for the launching of the job
    Because the job may take some time to be launched, we need to wait for the job to be launched
    :: check_file: the file that indicates the job is launched
    :: message: the message that will be printed when the job is launched
    """
    while not os.path.isfile(check_file):
        time.sleep(3)
    logging.info(message)

def remove_file(file):
    """
    Remove the file
    """
    try:
        dir = Path(file)
        dir.unlink()
    except Exception as e:
        logging.warning(f"Failed to remove {file}: The path does not exist or is not a file.")

def remove_files_with_pattern(pattern):
    """
    Remove the files with the given pattern
    Args:
        pattern (str): The pattern to match the files to remove (e.g. '*.txt')
    """
    parent_dir = Path(pattern).parent
    pattern_only = Path(pattern).name 

    if parent_dir.exists():
        for file in parent_dir.glob(pattern_only):
            try:
                file.unlink()
            except FileNotFoundError:
                logging.info(f"File does not match the given pattern: {pattern}")
            
            
def clean_and_initialize_directories(directories):
    """
    Remove all files and subdirectories in the given directories, and recreate them as empty.
    Args:
        directories (list): The list of directories to clean up and reinitialize.
    """
    for directory in directories:
        try:
            if os.path.exists(directory):
                if os.path.islink(directory): 
                    os.unlink(directory)
                    logging.debug(f"Unlinked symbolic link: {directory}")
                else:
                    shutil.rmtree(directory)
                    logging.debug(f"Removed directory: {directory}")
            
            os.makedirs(directory, exist_ok=True) 
            logging.debug(f"Recreated directory: {directory}")
        except Exception as e:
            logging.error(f"Failed to clean up {directory}: {e}")
            

def clean_symlink_target(symlink_path):
    """
    clean up the whole files and directories in a given synlink.
    Args:
        symlink_path (str): The path to the symlink to clean up.
    """
    if os.path.islink(symlink_path):
        target_path = os.readlink(symlink_path)
        target_abs_path = os.path.abspath(os.path.join(os.path.dirname(symlink_path), target_path))
        
        if os.path.isdir(target_abs_path):
            try:
                shutil.rmtree(target_abs_path)
                os.makedirs(target_abs_path)
                logging.info(f"Clear all contents in: {target_abs_path}")
            except Exception as e:
                logging.warning(f"Failed to clear contents in {target_abs_path}: {e}")
            
        else:
            logging.warning(f"Target path is not a directory: {target_abs_path}")
    else:
        logging.warning(f"Path is not a symbolic link: {symlink_path}")
        
        

def check_if_directory_not_empty(directory):
    """
    Check if the directory is not empty.
    Args:
        directory (str): The directory to check.
    Returns:
        bool: True if the directory is not empty, False otherwise.
    """
    if not os.path.isdir(directory):
        return False
    
    if len(os.listdir(directory)) == 0:
        return False
    
    return True

def check_path_is_correct(expected_dir):
    """
    Comparing the current directory with the expected directory
    Retrun:
        bool: True if the current directory is the same as the expected directory, False otherwise
    """
    current_dir = os.path.abspath(os.getcwd())
    error_message = f"Current directory {current_dir} does not match expected directory {expected_dir}"
    if current_dir != expected_dir:
        logging.error(error_message)
        return False
    else:
        return True
    

def move_files(src_dir, dst_dir, pattern):
    """
    Move files from the source directory to the destination directory with the given pattern.
    
    Args:
        src_dir (str): The source directory to move files from.
        dst_dir (str): The destination directory to move files to.
        pattern (str): The pattern to match the files to move (e.g. '*.txt').
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if not src_path.exists():
        logging.error(f"Source directory does not exist: {src_path}")
        return
    
    dst_path.mkdir(parents=True, exist_ok=True)

    for file in src_path.glob(pattern): 
        try:
            shutil.move(str(file), str(dst_path / file.name))
            logging.info(f"Moved file: {file} to {dst_path}")
        except Exception as e:
            logging.error(f"Failed to move {file} to {dst_path}: {e}")

    