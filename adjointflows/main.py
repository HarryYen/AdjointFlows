from tools import ConfigManager, FileManager
from tools import GLOBAL_PARAMS
from kernel import ModelGenerator, ForwardGenerator
import yaml
import os
import sys


def main():
    
    config = ConfigManager('config.yaml')
    config.load()
    
    # get parameters from config file
    mrun = config.get('setup.model.current_model_num')
    # get parameters from global_params.py
    base_dir = GLOBAL_PARAMS['base_dir']
    
    # other variables
    specfem_dir = os.path.join(base_dir, 'specfem3d')
    
    file_manager = FileManager()
    file_manager.set_model_number(current_model_num = mrun)
    file_manager.setup_directory(clear_directories = True)
    file_manager.make_symbolic_links()

    os.chdir(specfem_dir)
    model_generator = ModelGenerator(current_model_num = mrun)
    model_generator.model_setup(mesh_flag = True)

    forward_generator = ForwardGenerator(current_model_num = mrun, config = config)
    forward_generator.do_forward()

if __name__ == '__main__':
    
    main()	
