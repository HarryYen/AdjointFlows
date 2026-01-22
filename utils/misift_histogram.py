#%%
import pandas as pd
import numpy as np
import pygmt
import os
import sys

def generate_misfit_array(evt_arr, model_num):
    global tomo_dir
    
    tt_misfit_list, amp_misfit_list = [], []
    chi_list_tmp = []
    for evt in evt_arr:
        print(f'==============={evt}===================')
        misfit_dir = f'{tomo_dir}/{model_num}/MEASURE/adjoints/{evt}'
        misfit_file = f'{misfit_dir}/window_chi'
        if not os.path.exists(misfit_file):
            print(f'{misfit_file} does not exist')
            continue
        df = pd.read_csv(misfit_file, sep='\s+', header=None)

        df = df[(df.iloc[:, 28] != 0) | (df.iloc[:, 29] != 0)]
        # df = df[(df.iloc[:, 28] != 0)]
        # print(f'df shape: {df.shape}')
        # tt_misfit_list.append(df[28].values)
        # amp_misfit_list.append(df[29].values)
        
        mt_dt_list = df[12].values
        mt_dlna_list = df[13].values
        xc_dt_list = df[14].values
        xc_dlna_list = df[15].values
        chi_list = df[28].values
        
        dt_list = np.where(mt_dt_list != 0, mt_dt_list, xc_dt_list)
        dlna_list = np.where(mt_dt_list != 0, mt_dlna_list, xc_dlna_list)

        tt_misfit_list.append(dt_list)
        amp_misfit_list.append(dlna_list)
        chi_list_tmp.append(chi_list)
    tt_misfit_arr = np.concatenate(tt_misfit_list)
    amp_misfit_arr = np.concatenate(amp_misfit_list)
    chi_arr = np.concatenate(chi_list_tmp)

    chi_arr = np.sum(chi_arr) / len(chi_arr)
    print(f'chi_arr: {chi_arr}')
    
    return tt_misfit_arr, amp_misfit_arr

def pygmt_begin():
    fig = pygmt.Figure()
    pygmt.config(FONT_LABEL='24p',
                 FONT_ANNOT_PRIMARY='15p',)
    
    return fig

def pygmt_plot_histogram_dt(fig, data1, data2, interval, x_range):
    global model_num_1, model_num_2
    num_1, num_2 = len(data1), len(data2)
   # Create histogram for data02 by using the combined data set
    fig.histogram(
        region=[x_range[0], x_range[1], 0, 0],
        projection="X13c",
        frame=["WSne+tMisfit Histogram: dt", "xaf10+ltravel-time misfit(sec)", "yaf500+lCounts"],
        data=data2,
        series=interval,
        fill="skyblue",
        pen="1p,blue,solid",
        histtype=0,
        # The combined data set appears in the final histogram visually
        # as data set data02
        label=f"{model_num_2}(N={num_2})",
    )

    # Create histogram for data01
    # It is plotted on top of the histogram for data02
    fig.histogram(
        data=data1,
        series=interval,
        pen="2p,gray,solid",
        fill='orange',
        histtype=0,
        label=f"{model_num_1}(N={num_1})",
    )
    
    fig.histogram(
        data=data2,
        series=interval,
        pen="2p,blue,solid",
        histtype=0,
    )
    
    fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p1p")
    
    
    return fig

def pygmt_plot_histogram_dlnA(fig, data1, data2, interval, x_range):
    
    fig.shift_origin(xshift="16c")
    
    num_1, num_2 = len(data1), len(data2)
   # Create histogram for data02 by using the combined data set
    fig.histogram(
        region=[x_range[0], x_range[1], 0, 0],
        projection="X13c",
        frame=["WSne+tMisfit Histogram: dlnA", "xaf10+ldlnA", "yaf500+lCounts"],
        data=data2,
        series=interval,
        fill="skyblue",
        pen="1p,blue,solid",
        histtype=0,
        # The combined data set appears in the final histogram visually
        # as data set data02
        label=f"M_final(N={num_2})",
    )

    # Create histogram for data01
    # It is plotted on top of the histogram for data02
    fig.histogram(
        data=data1,
        series=interval,
        pen="2p,gray,solid",
        fill='orange',
        histtype=0,
        label=f"M_init(N={num_1})",
    )
    
    fig.histogram(
        data=data2,
        series=interval,
        pen="2p,blue,solid",
        histtype=0,
    )
    
    fig.legend(position="JTR+jTR+o0.2c", box="+gwhite+p1p")
    
    
    return fig

def check_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    return output_dir

if __name__ == '__main__':
    
    # -----------------PARAMETERS----------------- #
    tomo_dir = '/home/harry/Work/AdjointFlows/TOMO/'
    # event_file = '/home/harry/Work/AdjointFlows/DATA/evlst/test.txt'
    event_file = '/home/harry/Work/AdjointFlows/DATA/evlst/fwi_new_cat_version4.txt'
    # event_file = '/home/harry/Work/AdjointFlows/DATA/evlst/fwi_new_cat_87_version3.txt'
    out_dir = f'{tomo_dir}/OUTPUT'
    model_n_list = [15, 16]
    use_limit_rows = True
    head_rows_num = 60
    # -------------------------------------------- #
    
    evt_df = pd.read_csv(event_file, sep='\s+', header=None)
    if use_limit_rows:
        evt_df = evt_df.head(head_rows_num)
    evt_arr = evt_df[0].values

    
    model_num_1 = f'm{model_n_list[0]:03d}'
    model_num_2 = f'm{model_n_list[1]:03d}'
    

    
    tt_misfit_arr1, amp_misfit_arr1 = generate_misfit_array(evt_arr, model_num_1)
    # tomo_dir = f'{tomo_dir}/Results_from_last_stage'
    tt_misfit_arr2, amp_misfit_arr2 = generate_misfit_array(evt_arr, model_num_2)
    
    fig = pygmt_begin()
    fig = pygmt_plot_histogram_dt(fig, tt_misfit_arr1, tt_misfit_arr2, 0.25, [-5, 5])
    # fig = pygmt_plot_histogram_dt(fig, tt_misfit_arr1, tt_misfit_arr2, 0.005, [0, 0.1])
    # fig = pygmt_plot_histogram_dt(fig, tt_misfit_arr1, tt_misfit_arr2, , [0, 1E-07])
    
    fig = pygmt_plot_histogram_dlnA(fig, amp_misfit_arr1, amp_misfit_arr2, 0.2, [-3, 3])
    
    output_dir = check_output_dir(f'{out_dir}/misfit_histogram')
    output_file = f'{output_dir}/{model_num_1}_{model_num_2}.png'
    # fig.savefig(output_file, dpi=300, transparent = True)
    fig.show()
# %%
# 