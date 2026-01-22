#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pygmt
import os
import sys

import random
import colorsys
from matplotlib.colors import to_hex

def determine_which_leg(cluster_list, model_list):
    groups = np.digitize(model_list, cluster_list)
    return groups

def calculating_misfit(mrun):
    global evt_file, tomo_dir
    measure_dir = f'{tomo_dir}/m{mrun:03d}/MEASURE/adjoints'
    
    evt_df = pd.read_csv(evt_file, header=None, delimiter='\s+')
    
    chi_df = pd.DataFrame()
    for evt in evt_df[0]:
        adjoints_dir = f'{measure_dir}/{evt}'
        chi_file = f'{adjoints_dir}/window_chi'
        tmp_df = pd.read_csv(chi_file, header=None, delimiter='\s+')
        chi_df = pd.concat([chi_df, tmp_df])
    win_num = len(chi_df)
    misfit = round(chi_df[28].sum() / len(chi_df), 5)
    return misfit, win_num

def pygmt_begin():
    fig = pygmt.Figure()
    pygmt.config(FONT_LABEL='24p,4',
                 FONT_ANNOT_PRIMARY='12p,4',)
    
    return fig

def basemap_setting(fig, mbeg, mend):
    fig.basemap(region=[mbeg-0.9, mend+0.9, 5000, 7500], projection='X10i/4i', 
                frame=['xa1f1+literation(model)', 'nES'])
    return fig
    
def plot_leg_background_color(leg_list, leg_color):
    global leg_period_band, sigma_v_list, sigma_h_list
    for i in range(len(leg_list)):
        large_num = 99999
        leg_list[0] = -large_num
        fig.plot(
            x = [leg_list[i], leg_list[i], large_num, large_num, leg_list[i]],
            y = [-large_num, large_num, large_num, -large_num, -large_num],
            fill = leg_color[i],
            label = f"Stage{i+1}: @~\s@~@-v@- / @~\s@~@-h@- = {sigma_v_list[i]} / {sigma_h_list[i]}",
        )
    fig.legend(position='jBL+o0.2c', box='+gwhite+p1p')
    return fig
    
def pygmt_plot_misfit(fig, misfit_list, model_name_list, mbeg, mend):
    for stage in range(len(misfit_list)):
        model_arr = model_name_list[stage]
        misfit_arr = misfit_list[stage]

    #     misfit_df_group = misfit_df[misfit_df.leg == group]
    #     misfit_arr_normalized = misfit_df_group.misfit.values / misfit_df_group.misfit.values.max()
        misfit_arr_normalized = misfit_arr / np.max(misfit_arr)
        print(mbeg, mend)
        print(len(model_arr), len(misfit_arr_normalized))
        
        fig.plot(
            region=[mbeg-0.9, mend+0.9, 0.6, 1.1],
            x=model_arr, 
            y=misfit_arr_normalized,
            style='c0.2c',
            fill='#307ce1',
            pen='1p,black',
            frame = ['ya0.2f0.1+lnormalized misfit', 'W'],
        )
    fig.plot(
        x = 99999, y = 99999, style='c0.35c', pen='2p,black', fill='#307ce1', label = 'misfit'
    )
    return fig

def pygmt_plot_win_num(fig, misfit_df):

    fig.plot(
        x=misfit_df.model.values,
        y=misfit_df.win_num.values,
        style='d0.35c',
        fill='purple@70',
        pen = '1.5p,black',
        label = 'total window number'
    )
    return fig

def pygmt_legend_for_symbol(fig):
    fig.legend(position='jBR+o0.2c', box='+gwhite+p1p')
    return fig


def get_pastel_colors(n=5, min_dist=0.25):
    colors = []
    rgbs   = []
    while len(colors) < n:
        h = random.random()
        s = random.uniform(0.3, 0.6)   # saturation
        l = random.uniform(0.9, 0.95)    # Lightness  
        r, g, b = colorsys.hls_to_rgb(h, l, s)

        if all(((r - r0)**2 + (g - g0)**2 + (b - b0)**2)**0.5 >= min_dist for r0, g0, b0 in rgbs):
            colors.append(to_hex([r, g, b]))
            rgbs.append((r, g, b))
    return colors


if __name__ == '__main__':
    
    model_beg, model_final = 0, 26
    # group_startpoint_list = [0, 5, 7, 11, 14, 16, 20, 23]
    leg_list = [0, 5, 7, 11, 14, 16, 20, 23]
    # leg_period_band = ['10-30s', '8-30s', '5-30s']
    sigma_v_list = [8, 5, 3, 3, 2, 2, 1.5, 1]
    sigma_h_list = [16, 8, 6, 6, 4, 3, 2, 2]
    misfit_list = [
        [1.297, 1.280, 1.263, 1.182, 1.115, 1.090],
        [1.045, 1.017, 0.953],
        [0.962, 0.933, 0.883, 0.868, 0.866],
        [0.952, 0.940, 0.899, 0.887],
        [0.886, 0.839, 0.789],
        [2.388, 2.152, 2.098, 2.064, 2.059],
        [2.004, 1.819, 1.749, 1.715],
        [1.323, 1.288, 1.256, 1.229]
    ]
    model_name_list = [
        [0, 1, 2, 3, 4, 5],
        [5, 6, 7],
        [7, 8, 9, 10, 11],
        [11, 12, 13, 14],
        [14, 15, 16],
        [16, 17, 18, 19, 20],
        [20, 21, 22, 23],
        [23, 24, 25, 26]
    ]

    # leg_color = ['#c3f1ff', '#c3ffd3', '#fffccd', '#ffdfcd', '']
    leg_color = get_pastel_colors(n=len(leg_list), min_dist=0.05)
    # tomo_dir = '/home/harry/Work//Adjoint_tomography_old_dataset/FWI_result_gutenberg_1/output_from_ADJOINT_TOMO'
    # evt_file = '/home/harry/Work/Adjoint_tomography_old_dataset/FWI_result_gutenberg_1/rmt_g10.txt'
    output_dir = '/home/harry/Work/AdjointFlows_mesh_2/TOMO'
    

   
    # for model_n in range(model_beg, model_final + 1):


    #     misfit_list.append(misfit)
    #     win_num_list.append(win_num)
    #     model_name_list.append(model_n)


    # leg_arr = determine_which_leg(group_startpoint_list, model_name_list)
    
    # misfit_df = pd.DataFrame({
    #     'model': model_name_list,
    #     'misfit': misfit_list,
    # })
    
    
    fig = pygmt_begin()
    fig = basemap_setting(fig, model_beg, model_final)
    fig = plot_leg_background_color(leg_list, leg_color)
    # fig = pygmt_plot_win_num(fig, misfit_df)
    fig = pygmt_plot_misfit(fig, misfit_list, model_name_list, model_beg, model_final)
    fig = pygmt_legend_for_symbol(fig)
    # fig = pygmt_plot_win_num(fig, misfit_df, model_beg, model_final)
    fig.show()
    # fig.savefig(f'{output_dir}/misfit_with_iterations.png', dpi=300, transparent=True)
    # fig.savefig()
    

# %%
