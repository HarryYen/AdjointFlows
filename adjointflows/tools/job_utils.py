from pathlib import Path
import time
import os
import logging
import shutil
import glob


debug_logger = logging.getLogger("debug_logger")

def wait_for_launching(check_file, message):
    """
    Wait for the launching of the job
    Because the job may take some time to be launched, we need to wait for the job to be launched
    :: check_file: the file that indicates the job is launched
    :: message: the message that will be printed when the job is launched
    """
    while not os.path.isfile(check_file):
        time.sleep(3)
    debug_logger.info(message)

def remove_file(file):
    """
    Remove the file
    """
    try:
        dir = Path(file)
        dir.unlink()
    except Exception as e:
        debug_logger.warning(f"Failed to remove {file}: The path does not exist or is not a file.")

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
                debug_logger.info(f"File does not match the given pattern: {pattern}")
            
def make_symlink(src, dst):
    """
    Make a symbolic link from the source to the destination.
    Args:
        src (str): The source path to link from.
        dst (str): The destination path to link to.
    """
    if not os.path.exists(src):
        print(f"Warning: Target {src} does not exist! The symlink may be broken.")

    if os.path.exists(dst) or os.path.islink(dst):
        os.unlink(dst)
    
    try:
        os.symlink(src, dst)
        debug_logger.info(f"Created symbolic link: {dst} -> {src}")
    except Exception as e:
        debug_logger.error(f"Failed to create symbolic link {dst} -> {src}: {e}")
        
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
                    debug_logger.debug(f"Unlinked symbolic link: {directory}")
                else:
                    shutil.rmtree(directory)
                    debug_logger.debug(f"Removed directory: {directory}")
            
            os.makedirs(directory, exist_ok=True) 
            debug_logger.debug(f"Recreated directory: {directory}")
        except Exception as e:
            debug_logger.error(f"Failed to clean up {directory}: {e}")
            

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
                debug_logger.info(f"Clear all contents in: {target_abs_path}")
            except Exception as e:
                debug_logger.warning(f"Failed to clear contents in {target_abs_path}: {e}")
            
        else:
            debug_logger.warning(f"Target path is not a directory: {target_abs_path}")
    else:
        debug_logger.warning(f"Path is not a symbolic link: {symlink_path}")
        
        

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
        debug_logger.error(error_message)
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
        debug_logger.error(f"Source directory does not exist: {src_path}")
        return
    
    dst_path.mkdir(parents=True, exist_ok=True)

    for file in src_path.glob(pattern): 
        try:
            shutil.move(str(file), str(dst_path / file.name))
            debug_logger.debug(f"Moved file: {file} to {dst_path}")
        except Exception as e:
            debug_logger.error(f"Failed to move {file} to {dst_path}: {e}")
            
def copy_files(src_dir, dst_dir, pattern):
    """
    Copy files from the source directory to the destination directory with the given pattern.
    
    Args:
        src_dir (str): The source directory to copy files from.
        dst_dir (str): The destination directory to copy files to.
        pattern (str): The pattern to match the files to copy (e.g. '*.txt').
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)

    if not src_path.exists():
        debug_logger.error(f"Source directory does not exist: {src_path}")
        return
    
    dst_path.mkdir(parents=True, exist_ok=True)

    for file in src_path.glob(pattern): 
        try:
            shutil.copy2(str(file), str(dst_path / file.name))  
            debug_logger.debug(f"Copied file: {file} to {dst_path}")
        except Exception as e:
            debug_logger.error(f"Failed to copy {file} to {dst_path}: {e}")
            
def check_dir_exists(dir_name):
    """
    Check if the directory exists, if not, create it.
    Args:
        dir_name (str): The directory name to check.
    """
    os.makedirs(dir_name, exist_ok=True)


def remove_path(directories):
    """
    Remove the specified directories or files, including symbolic links.
    Args:
        directories (list): A list of directories or files to remove.
    """
    for directory in directories:
        try:
            if not os.path.lexists(directory):
                debug_logger.warning(f"Path does not exist: {directory}")
                continue
            
            if os.path.islink(directory): 
                os.unlink(directory)
                debug_logger.debug(f"Unlinked symbolic link: {directory}")
            elif os.path.isdir(directory): 
                shutil.rmtree(directory)
                debug_logger.debug(f"Removed directory: {directory}")
            elif os.path.isfile(directory): 
                os.remove(directory)
                debug_logger.debug(f"Removed file: {directory}")
            else:
                debug_logger.warning(f"Unknown path type, skipped: {directory}")
        except Exception as e:
            debug_logger.error(f"Failed to remove {directory}: {e}")

