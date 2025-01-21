import time
import os
import logging
import shutil


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
        os.remove(file)
    except Exception as e:
        logging.warning(f"Failed to remove {file}: The path does not exist or is not a file.")
        
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