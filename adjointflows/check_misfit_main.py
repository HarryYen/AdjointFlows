#%%
from visualizer.misfits_reduction import plot_misfits_reduction
from tools import ConfigManager
from tools import GLOBAL_PARAMS

def main():

    model_beg = 0
    model_end = 2

    
    config = ConfigManager('config.yaml')
    config.load()
    
    plot_misfits_reduction(model_beg=model_beg, model_end=model_end, config=config)


if __name__ == '__main__':
    main()
# %%
