from tools import ModelEvaluator
import matplotlib.pyplot as plt


def plot_misfits_reduction(model_beg, model_end, config):
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title('Misfits Reduction')
    ax.set_xlabel('Model Number')
    ax.set_ylabel('Misfit')
    
    for model_num in range(model_beg, model_end+1):
        model_evaluator = ModelEvaluator(model_num, config)    
        misfit = model_evaluator.misfit_calculation(model_num)
        print(misfit)
        ax.plot(model_num, misfit, 'ro')
    
    plt.show()