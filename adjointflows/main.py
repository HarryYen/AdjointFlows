from tools import ConfigManager, FileManager
from kernel import ModelGenerator
import yaml
import os


def main():
    #
    mrun = 0
    #
    
    current_dir = os.getcwd()
    work_dir = os.path.abspath(os.path.join(current_dir, ".."))
    
    config = ConfigManager('config.yaml')
    config.load()
    
    file_manager = FileManager(base_dir=work_dir)
    file_manager.set_model_number(current_model_num=mrun)
    file_manager.setup_directory(clear_directories=True)
    file_manager.make_symbolic_links()

    model_generator = ModelGenerator(base_dir=work_dir, current_model_num=mrun)
    model_generator.model_setup(mesh_flag=True)


if __name__ == '__main__':
    
    main()	
