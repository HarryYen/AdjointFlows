#%%
import pandas as pd
import numpy as np
import pygmt
import os
import sys

def get_chi(evt_arr, model_num, line_search_num):
    global tomo_dir
    
    chi_list_tmp = []
    for evt in evt_arr:
        print(f'==============={evt}===================')
        if line_search_num == 0:
            misfit_dir = f'{tomo_dir}/m{model_num:03d}/MEASURE/adjoints/{evt}'
        else:
            misfit_dir = f'{tomo_dir}/mtest{model_num:03d}/MEASURE{line_search_num:1d}/{evt}'
        
        misfit_file = f'{misfit_dir}/window_chi'
        if not os.path.exists(misfit_file):
            print(f'{misfit_file} does not exist')
            continue
        df = pd.read_csv(misfit_file, sep='\s+', header=None)
        df = df[(df.iloc[:, 28] != 0) | (df.iloc[:, 29] != 0)]
        chi_list = df[28].values
        chi_list_tmp.append(chi_list)

    chi_arr = np.concatenate(chi_list_tmp)

    chi_arr = np.sum(chi_arr) / len(chi_arr)
    print(f'chi_arr: {chi_arr}')
    
    return chi_arr

def get_misfit_list(model_num, ls_beg, ls_end):
    misfit_list = []
    for ls_num in range(ls_beg, ls_end+1):
        chi_arr = get_chi(evt_arr, model_num, ls_num)
        misfit_list.append(chi_arr)
    return misfit_list

def pygmt_begin():
    fig = pygmt.Figure()
    pygmt.config(FONT_LABEL='16p',
                 FONT_ANNOT_PRIMARY='15p',)
    
    return fig

def plot_misfit(fig, model_beg, model_end, misfit_list):
    
    model_list = np.arange(model_beg, model_end+1)
    fig.basemap(
        region=[model_beg-0.9, model_end+0.9, 0.7, 1.09],
        projection="X8c/10c",
        frame=["WSne+tMisfit Reduction", "x1a1f+lModel Number", "y+lMisfit"],
    )
    fig.plot(
        x = model_list,
        y = misfit_list,
        pen = '1p,black'
    )
    fig.plot(
        x = model_list,
        y = misfit_list,
        style = "c0.2c",
        pen = "0.5p,black",
        fill = "red",
    )
    
    
    return fig

def check_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir

if __name__ == '__main__':
    
    # -----------------PARAMETERS----------------- #
    tomo_dir = '/home/harry/Work/AdjointFlows/TOMO'
    # event_file = '/home/harry/Work/AdjointFlows/DATA/evlst/test.txt'
    event_file = '/home/harry/Work/AdjointFlows/DATA/evlst/fwi_new_cat_version4_18.txt'
    # event_file = '/home/harry/Work/AdjointFlows/DATA/evlst/fwi_new_cat_87_version3.txt'
    out_dir = '/home/harry/Work/AdjointFlows/TOMO/OUTPUT'
    line_search_n_list = [0, 9]
    model_num = 14
    
    # -------------------------------------------- #
    
    evt_df = pd.read_csv(event_file, sep='\s+', header=None)
    evt_arr = evt_df[0].values

    misfit_list = get_misfit_list(model_num=model_num, ls_beg=line_search_n_list[0], ls_end=line_search_n_list[1])
    misfit_arr = np.array(misfit_list)
    misfit_arr = misfit_arr / misfit_arr[0]
    fig = pygmt_begin()
    fig = plot_misfit(fig=fig, model_beg=line_search_n_list[0], model_end=line_search_n_list[1], misfit_list=misfit_arr)
    
    fig.show()
# %%
# 