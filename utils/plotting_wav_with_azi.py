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
    
        

def generate_evt_win_dict(win_dir):
    '''
    read window index file from final model
    here we output 2 dictonaries: evt_dict and win_dict_evt
    1. evt_dict: {evt: win_dict_evt}
    2. win_dict_evt: {(sta, comp): [(t1, t2)]}
    '''
    evt_dict = {}
    for dir in glob.glob(f'{win_dir}/*'):
        evt = dir.split('/')[-1]
        win_file = f'{dir}/window_index'
        df = pd.read_csv(win_file, delimiter='\s+', header=None, 
                         names=['net', 'sta', 'comp', 'u1', 'u2', 'u3', 'u4', 't1', 't2'])
       
        win_dict_evt = {
            (sta, comp): list(zip(group['t1'], group['t2']))
            for (sta, comp), group in df.groupby(['sta', 'comp'])
        }
        
        evt_dict[evt] = win_dict_evt
    
    return evt_dict

def generate_sta_dataframe(obs_wav_dir, win_dict_evt, comp):
            
    sta_list, azi_list = [], []
    stlo_list, stla_list = [], []
    evlo_list, evla_list = [], []
    
    win_dict_evt = {key: value for key, value in win_dict_evt.items() if key[1] == comp}
    for key in win_dict_evt.keys():
        sta = key[0]
        try:
            obs_sac = glob.glob(f'{obs_wav_dir}/{sta}.TW.{comp}.sac.tomo')[0]
        except IndexError:
            print(f"{obs_wav_dir}/{sta}.TW.{comp}.sac.tomo' doesn't exist!")
            continue
        
        stream = read(obs_sac)
        st = stream.copy()
        tr = st[0]
        try:
            azi = tr.stats.sac.az
        except:
            azi = 0.
            print(f'WARNING: {sta} has no Azi info. Use 0')

        stlo, stla = tr.stats.sac.stlo, tr.stats.sac.stla
        evlo, evla = tr.stats.sac.evlo, tr.stats.sac.evla
        
        sta_list.append(sta)
        azi_list.append(azi)
        stlo_list.append(stlo)
        stla_list.append(stla)
        evlo_list.append(evlo)
        evla_list.append(evla)
    
    sta_azi_df = pd.DataFrame({'sta': sta_list, 'azi': azi_list, 'stlo': stlo_list, 'stla': stla_list, 'evlo': evlo_list, 'evla': evla_list})
    sta_azi_df = sta_azi_df.drop_duplicates()
    sta_azi_df = sta_azi_df.sort_values(by='azi')
    
    return sta_azi_df

def chunck_dataframe(df, chunksize=10):
    df_splits = [df.iloc[i: i + chunksize] for i in range(0, len(df), chunksize)]
    return df_splits

def Pygmt_config():
    font = 0
    pygmt.config(MAP_FRAME_TYPE="plain",
                 FORMAT_GEO_MAP="ddd.x",
                 FONT = f'24p, {font}',
                 FONT_TITLE = f'24p, 1',
                 MAP_TITLE_OFFSET="0.1c")
    


def plot_map(map_region, sta_df, meca_info):
    fig = pygmt.Figure()
    
    evlon, evlat, evdep, evmag = meca_info[0], meca_info[1], meca_info[2], meca_info[6]
    fig.basemap(region=map_region, projection="M8i", frame=['a2f1', f'WSne+tlon.: {evlon}, lat: {evlat}, dep.: {evdep}, Mw: {evmag}'])
    fig.coast(shorelines=True)
    
    x_coords = []
    y_coords = []
    sta_list = []
    for line in sta_df.itertuples():
        x_coords.extend([line.stlo, line.evlo, np.nan])
        y_coords.extend([line.stla, line.evla, np.nan])
        sta_list.append(line.sta)
        
    fig.plot(x=x_coords, y=y_coords, pen='0.8p,black')    
    fig.plot(x = sta_df.stlo, y = sta_df.stla, style='t1c', fill='blue', pen='black')
    
    pygmt.makecpt(cmap='jet', series=[0, 200], reverse=True)
    
    fig.meca(
        spec = meca_info,
        convention = 'aki',
        cmap = True,
        scale = "1.6c"
    )
    
    fig.text(x=sta_df.stlo, y=sta_df.stla - 0.15, text=sta_df.sta, 
             font='24p,1', justify='CM', fill='#ffffaa')
    
    fig.colorbar(cmap = True, position = 'x0.5c/0.5c+w7c/0.6c+m+h', frame = ['a20f10','+LDepth (km)'])  
    
    return fig

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

def plot_waveforms(fig, evt, evt_time, channel, waveform_time_range, chunk_df, win_dict_evt, ref_syn_wav_dir, final_syn_wav_dir, ref_wav_dir):
    global wav_start_time, chunksize, model_ref, model_final
    
    fig.shift_origin(xshift="22c")
    with fig.subplot(nrows=chunksize, ncols=2, subsize=('17c', '5c'), margins=["0.6c", "0.6c"], 
                     sharex='b', sharey='r'):
        for i in range(len(chunk_df)):
            sta_info = chunk_df.iloc[i]
            sta = sta_info.sta
            
            comp = channel[1:]
            comp_single = channel[2]
            # ref_syn_sac = f'{ref_syn_wav_dir}/{evt}/{sta}.TW.{channel}.semv.sac.tomo'
            # final_syn_sac = f'{final_syn_wav_dir}/{evt}/{sta}.TW.{channel}.semv.sac.tomo'
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
                    
                    # get window info
                    try:
                        win_values = win_dict_evt[(sta, channel)]
                        for win in win_values:
                            t1, t2 = win
                            fig.plot(x=[t1, t2, t2, t1, t1], y=[-9999, -9999, 9999, 9999, -9999], pen='1p,black', fill='pink@50')
                    except KeyError:
                        pass
                    # fig.plot(x = data_time + wav_start_time, y = wav_data, pen='2.5p,black')
                    # fig.plot(x = syn_time + wav_start_time, y = wav_syn, pen='2.5p,red')
                    fig.plot(x = data_time, y = wav_data, pen='2.5p,black')
                    fig.plot(x = syn_time, y = wav_syn, pen='2.5p,red')
    return fig
        
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
    model_ref, model_final = 0, 16
    period_min, period_max = 5, 30
    map_region = [119, 123, 21, 26]
    result_dir = '/home/harry/Work/AdjointFlows/TOMO'
    data_dir = '/home/harry/Work/AdjointFlows/DATA/wav'
    evt_file = '/home/harry/Work/AdjointFlows/DATA/evlst/fwi_new_cat_version4.txt'
    waveform_time_range = [0, 150]
    wav_start_time = -30.
    output_dir = '/home/harry/Work/AdjointFlows/TOMO/OUTPUT'
    chunksize = 6
    # --------------------------------------------#
    
    evt_meca_df = create_meca_dataframe(evt_file)
    
    model_ref = f'm{model_ref:03d}'
    model_final = f'm{model_final:03d}'
    final_win_dir = f'{result_dir}/{model_final}/MEASURE/adjoints'
    # ref_wav_dir = f'{result_dir}/{model_ref}/OBS'
    ref_wav_dir = data_dir
    ref_syn_wav_dir = f'{result_dir}/{model_ref}/SYN'
    final_syn_wav_dir = f'{result_dir}/{model_final}/SYN'
    
    output_dir = f'{output_dir}/{model_ref}_{model_final}'
    ensure_directory_exists(output_dir)
    
    
    evt_dict = generate_evt_win_dict(final_win_dir)
    evt_list = list(evt_dict.keys())
    
    # SET GMT CONFIG
    Pygmt_config()
    
    total_evt_num = len(evt_list)
    
    for evt, win_dict_evt in evt_dict.items():
        out_dir_evt = f'{output_dir}/{evt}'
        ensure_directory_exists(out_dir_evt)
        print(f'Start plotting event {evt} ({evt_list.index(evt)+1}/{total_evt_num})...')
        obs_wav_dir = f'{ref_wav_dir}/{evt}'
        evt_info = evt_meca_df[evt_meca_df.formatted_datetime == float(evt)].iloc[0]
        evt_time = UTCDateTime(evt_info.date + 'T' + evt_info.time)
        
        for comp in ['BHZ', 'BHN', 'BHE']:
            sta_azi_df = generate_sta_dataframe(obs_wav_dir, win_dict_evt, comp)

            chunk_list = chunck_dataframe(df = sta_azi_df, chunksize=chunksize)
            
            meca_info = modify_meca_format(evt_meca_df, evt)
            for ii, chunk_df in enumerate(chunk_list):
                fig = plot_map(map_region, chunk_df, meca_info)
                fig = plot_waveforms(fig, evt, evt_time, comp, waveform_time_range, chunk_df, win_dict_evt, 
                            ref_syn_wav_dir, final_syn_wav_dir, ref_wav_dir)

                # fig.savefig(f'{out_dir_evt}/{evt}_{comp}_{ii+1:02d}.jpg')



# %%
