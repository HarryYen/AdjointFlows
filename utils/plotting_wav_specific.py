#%%
"""
This script is used to plot waveforms with azimuth information.
We can check the FWI results between two models (e.g., M00, M16) by comparing the synthetics and observed data.

Preparation:
    - Waveform data (synthetics and observed) in SAC format
        - Synthetics directory: ${result_dir}/m???/SYN
        - Data directory: ${result_dir}/m???/OBS
    - Window index file in each event directory
        - Window index file: ${final_dir}/m???/MEASURE/adjoints/${event}/window_index
    - Event file with moment tensor information (the same file used in adjtomo)
        - Set the file path as ${evt_file}

Usage:
    - Decide the `model_ref` and `model_final` you want to compare (e.g., m000, m016).
    - Modify the PARAMETERS section in the script.
    - Run the script with: `python plot_wav_with_azi_comp.py`

Output:
    - The output will be saved in ${output_dir}/m???_m???/
        e.g., /home/harry/Work/FWI_result/waveform_with_azi/m010_m016/
    - Each figure represents the waveform comparison of one event and one component (maximum 6 stations per figure).
      This script arranges stations by azimuth and groups them into figures with 6 stations per plot.
"""

import matplotlib.pyplot as plt
from obspy import read
from obspy import UTCDateTime
import numpy as np
import pandas as pd
import glob
import os
import sys
import pygmt

def create_meca_dataframe(file):
    df = pd.read_csv(file, sep="\s+", header=None, usecols=range(20), names=[
        'formatted_datetime', 'date', 'time', 'long', 'lat', 'depth', 
        'strike1', 'dip1', 'rake1', 'strike2', 'dip2', 'rake2',
        'Mw', 'MR', 'mrr', 'mtt', 'mpp', 'mrt', 'mrp', 'mtp'
    ])
    return df

def modify_meca_format(df, evt):
    
    df_sort = df[df.formatted_datetime == float(evt)]
    df_meca = df_sort[['long', 'lat', 'depth', 'strike1', 'dip1', 'rake1', 'Mw']]
    # df_meca.columns = ['longitude','latitude', 'depth',
                    # 'strike', 'dip', 'rake', 'magnitude']
    meca_info = np.array(list(df_meca.iloc[0]))

    return meca_info
    
        
def get_evt_win_info(evt, sta, comp, win_dir):

    win_file = f'{win_dir}/{evt}/window_index'
    df = pd.read_csv(win_file, delimiter='\s+', header=None, 
                        names=['net', 'sta', 'comp', 'u1', 'u2', 'u3', 'u4', 't1', 't2'])
    win_info = df[(df['sta'] == sta) & (df['comp'] == comp)][['t1', 't2']]
    t1_list = win_info['t1'].tolist()
    t2_list = win_info['t2'].tolist()
    
    return t1_list, t2_list


def Pygmt_config():
    font = 0
    pygmt.config(MAP_FRAME_TYPE="plain",
                 FORMAT_GEO_MAP="ddd.x",
                 FONT = f'24p, {font}',
                 FONT_TITLE = f'24p, 1',
                 MAP_TITLE_OFFSET="0.1c")
    


def wav_preprocessing(tr):
    global period_min, period_max
    tr.detrend(type='demean')
    tr.detrend(type='linear')
    tr.filter('bandpass', freqmin=1/period_max, freqmax=1/period_min, corners=4, zerophase=False)
    tr.taper(0.05, type='hann')
    tr.detrend(type='demean')
    tr.detrend(type='linear')
    return tr
    
        
def read_sac_syn(sac, t0, t1, time):
    try:
        stream = read(sac)
        st = stream.copy()
        tr = st[0]
        tr = wav_preprocessing(tr)
        tr.trim(time+t0, time+t1)
        wav = tr.data     
        times = tr.times()
    except FileNotFoundError:
        wav = np.nan
        times = np.nan
    return wav, times

def read_sac_data(sac, t0, t1, time):
    try:
        stream = read(sac)
        st = stream.copy()
        tr = st[0]
        tr = wav_preprocessing(tr)
        tr.trim(time+t0, time+t1)
        wav = tr.data     
        times = tr.times()
    except FileNotFoundError:
        wav = np.nan
        times = np.nan
    return wav, times

        
def ensure_directory_exists(path):
    os.makedirs(path, exist_ok=True)


if __name__ == '__main__':
    """
    PARAMETERS
    
    - model_ref, model_final
        The model number you want to compare. e.g., 10, 16
    - map_region
        The region you want to plot. e.g., [119, 123, 21, 26]
    - result_dir
        The directory where you put the waveform files (both synthetics and data).
        Note that the path need to be ${result_dir}/m???/SYN and ${result_dir}/m???/OBS
    - evt_file
        The event file with moment tensor information.
    - waveform_time_range
        The time range you want to plot. e.g., [0, 120]
    - wav_start_time
        You need to check the start time of your sac files.
        e.g. -30 means your sac file starts from -30 seconds
    - output_dir
        The directory where you want to save the output figures.
    
    """
    
    # ---------------- PARAMETER -----------------#
    model_ref, model_final = 0, 14
    period_min, period_max = 5, 30
    map_region = [119, 123, 21, 26]
    result_dir = '/home/harry/Work/AdjointFlows/TOMO'
    data_dir = '/home/harry/Work/AdjointFlows/DATA/wav'
    evt_file = '/home/harry/Work/AdjointFlows/DATA/evlst/fwi_new_cat_version4.txt'
    sta_file = '/home/harry/Work/AdjointFlows/DATA/stlst/st_new_remove_western_only_new.txt'
    waveform_time_range = [0, 150]
    wav_start_time = -30.
    output_dir = '/home/harry/Work/AdjointFlows/TOMO/OUTPUT'
    chunksize = 6
    

    event_list = ['202109252221', '202203222030', '202207280016', '202306100531', '202309050930', '201903122019']
    sta_list = ['SBCB', 'TWGB', 'NTS', 'LATB', 'HOPB', 'ETLH']
    comp_list = ['BHN', 'BHE', 'BHN', 'BHN', 'BHE', 'BHN']
    # --------------------------------------------#
    
    win_dir = f'{result_dir}/m{model_final:03d}/MEASURE/adjoints'

    evt_meca_df = create_meca_dataframe(evt_file)
    
    model_ref = f'm{model_ref:03d}'
    model_final = f'm{model_final:03d}'
    final_win_dir = f'{result_dir}/{model_final}/MEASURE/adjoints'

    ref_wav_dir = data_dir
    ref_syn_wav_dir = f'{result_dir}/{model_ref}/SYN'
    final_syn_wav_dir = f'{result_dir}/{model_final}/SYN'

    sta_df = pd.read_csv(sta_file, sep='\s+', header=None, names=['sta', 'stlo', 'stla', 'ele'])    


    # plot #
    fig = pygmt.Figure()
    fig.basemap(region=map_region, projection="M8i", frame=['a2f1', 'WSne'])
    fig.coast(shorelines=True)


    pygmt.makecpt(cmap='jet', series=[0, 200], reverse=True)
    for index, evt in enumerate(event_list):
        
        evt_info = evt_meca_df[evt_meca_df.formatted_datetime == float(evt)].iloc[0]
        meca_info = modify_meca_format(evt_meca_df, evt)
        evlon, evlat, evdep, evmag = meca_info[0], meca_info[1], meca_info[2], meca_info[6]

        sta = sta_list[index]
        sta_info = sta_df[sta_df.sta == sta].iloc[0]
        stlon, stlat = sta_info.stlo, sta_info.stla

        fig.plot(x=[stlon, evlon], y=[stlat, evlat], pen='0.8p,black')    


        
        fig.plot(x = stlon, y = stlat, style='t0.4c', fill='blue', pen='black')
        
        fig.meca(
            spec = meca_info,
            convention = 'aki',
            cmap = True,
            scale = "1.0c"
        )
    

        
        fig.text(x=stlon, y=stlat - 0.15, text=sta, 
                font='24p,1', justify='CM', fill='#ffffaa')
        
    fig.colorbar(cmap = True, position = 'x0.5c/0.5c+w7c/0.6c+m+h', frame = ['a20f10','+LDepth (km)'])  
    fig.shift_origin(xshift="22c")

    with fig.subplot(nrows=chunksize, ncols=2, subsize=('17c', '5c'), margins=["0.6c", "0.6c"], 
                     sharex='b', sharey='r'):
        for i, evt in enumerate(event_list):
            evt_time = UTCDateTime(evt_meca_df[evt_meca_df.formatted_datetime == float(evt)].iloc[0].date + 'T' + evt_meca_df[evt_meca_df.formatted_datetime == float(evt)].iloc[0].time)
            
            sta = sta_list[i]
            channel = comp_list[i]
            
            
            comp = channel[1:]
            comp_single = channel[2]
            ref_syn_sac = f'{ref_syn_wav_dir}/{evt}/*.{sta}.BX{comp_single}.semv.convolved.sac'
            final_syn_sac = f'{final_syn_wav_dir}/{evt}/*.{sta}.BX{comp_single}.semv.convolved.sac'            
            ref_data_name = f'{ref_wav_dir}/{evt}/{sta}.H{comp}*.sac'
            
            try:
                ref_data_sac = glob.glob(ref_data_name)[0]
            except IndexError:
                print(f"{ref_data_name} doesn't exist!")
                continue
            
            syn_sac_list = [ref_syn_sac, final_syn_sac]
            title_list = [model_ref, model_final]        
                        
            for j, syn_sac in enumerate(syn_sac_list): 
                index = i * 2 + j  
                # get waveform
                wav_data, data_time  = read_sac_data(ref_data_sac, waveform_time_range[0], waveform_time_range[1], evt_time)
                wav_syn, syn_time    = read_sac_syn(syn_sac, waveform_time_range[0], waveform_time_range[1], evt_time)
                
                max_val = np.max(np.abs(np.hstack([wav_data, wav_syn])))
                
                # normalize
                wav_data = wav_data / max_val
                wav_syn = wav_syn / max_val
                formatted_title = f'{sta} {channel} {title_list[j]}'

                with fig.set_panel(panel=index):
                    fig.basemap(region = [waveform_time_range[0], waveform_time_range[1], -1.3, 1.3], 
                                frame = ['xa20f10', 'yf1',f'+t{formatted_title}'], projection = 'X?')
                    
                    #get window info
                    try:
                        t1_list, t2_list = get_evt_win_info(evt, sta, channel, win_dir)

                        for win_i in range(len(t1_list)):
                            t1, t2 = t1_list[win_i], t2_list[win_i]
                            fig.plot(x=[t1, t2, t2, t1, t1], y=[-9999, -9999, 9999, 9999, -9999], pen='1p,black', fill='pink@50')
                    except KeyError:
                        pass
                    fig.plot(x = data_time, y = wav_data, pen='2.5p,black')
                    fig.plot(x = syn_time, y = wav_syn, pen='2.5p,red')



    fig.show()




# %%
