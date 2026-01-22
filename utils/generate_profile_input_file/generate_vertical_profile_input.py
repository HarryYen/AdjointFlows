#%%
import numpy as np
import pygmt
import pandas as pd

"""
docstring:
    This code is used for generating the input file for the vertical profile.
    Here you can also check the profile on the map (profiles_check.png).
output:
    1. profiles_check.png: the map with profiles
    2. profile.input.tmp: the input file for the vertical profile
"""

def check_map_begin(clon, clat, angle, interval):
    global map_range
    
    pygmt.config(MAP_FRAME_TYPE="plain")
    fig = pygmt.Figure()
    fig.basemap(region=map_range, projection='M4.5i', frame=["neWS","a2f1"])
    fig.coast(shorelines=True, resolution='h')
    
    return fig

def output_input_file(center_df, angle, profile_half_length):
    num = len(center_df)
    lon_arr = center_df.r.values
    lat_arr = center_df.s.values
    azi_arr = np.full(num, angle)
    lmin_arr = np.full(num, profile_half_length[0])
    lmax_arr = np.full(num, profile_half_length[1])
    index_arr = np.arange(num) + 1
    df = pd.DataFrame({
        'clon': lon_arr,
        'clat': lat_arr,
        'angle': azi_arr,
        'lmin': lmin_arr,
        'lmax': lmax_arr,
        'index': index_arr
    })
    df.to_csv('profile.input.tmp', index=False, header=False, sep=' ')
    

if __name__ == '__main__':
    """
    Parameters:
    - map_range (list): [lon_min, lon_max, lat_min, lat_max]
    - interval(float, int): the interval between each profile
    - clon, clat (float, int): the starting point
    - angle (float, int): the direction among the profiles
        * PLEASE note that the angle here is not the azimuth of profiles themselves.
        * e.g. If the angle here is 105, the azimuth of profiles will be 15 (profile angle = angle here - 90).
        * Finally, the profiles will be arranged along the angle (105 degree) and the angle itself is 15 degree.
    - range_length (float, int): the length of the range along [angle]
    - profile_length (float, int): the length of the profile itself
    - output_fig_flag (bool): whether to output the map with
    """
    # -------- Parameters --------
    map_range = [119, 123.5, 20.5, 26.5]
    interval = 20
    clon, clat = 121.4, 23.8
    angle = 195
    range_length = 300
    profile_half_length = [-60, 40] 
    output_fig_flag = True
    # ----------------------------

    range_half_len = range_length / 2
    range_df = pygmt.project(
            center = [clon, clat],
            azimuth = angle,
            length = [-range_half_len, range_half_len],
            unit = True,
            generate = interval,
    )

    fig = check_map_begin(clon, clat, angle, interval)
    
    azi = angle - 90
    # half_len = profile_length / 2
    for ii in range(len(range_df)):
        lon, lat = range_df.iloc[ii].r, range_df.iloc[ii].s
        profile_df = pygmt.project(
            center = [lon, lat],
            azimuth = azi,
            length = profile_half_length,
            unit = True,
            generate = 1,
        )
        fig.plot(
            x = [profile_df.r.values[0], profile_df.r.values[-1]],
            y = [profile_df.s.values[0], profile_df.s.values[-1]],
            pen='1.5p,blue')
        fig.plot(x = lon, y = lat, style='c0.2c', fill='red')
        
    output_input_file(range_df, azi, profile_half_length)
    
    if output_fig_flag:
        # fig.show()
        fig.savefig('profiles_check.png')



    
# %%
