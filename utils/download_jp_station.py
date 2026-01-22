#%%
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy import Stream
import pandas as pd
import os
import sys

evt_file = '/home/harry/Work/AdjointFlows/DATA/evlst/fwi_new_cat_87_version3.txt'

df = pd.read_csv(evt_file, sep="\s+", header=None, usecols=range(20), names=[
    'formatted_datetime', 'date', 'time', 'long', 'lat', 'depth', 
    'strike1', 'dip1', 'rake1', 'strike2', 'dip2', 'rake2',
    'Mw', 'MR', 'mrr', 'mtt', 'mpp', 'mrt', 'mrp', 'mtp'
])
print(df)

sys.exit()


# 設定 FDSN client，這裡使用 IRIS 資料中心
client = Client("IRIS")

# 設定下載時間區間（你可以依需求修改）
starttime = UTCDateTime("2017-02-10T17:00:00")
endtime = UTCDateTime("2017-02-10T18:00:00")  # 1 小時資料

# 設定台站資訊
network = "JP"      
station = "YOJ"
location = "*"    
channel = "BH?"    

# 下載波形資料
st = client.get_waveforms(network=network, station=station,
                          location=location, channel=channel,
                          starttime=starttime, endtime=endtime)


# 下載台站的 metadata（包括儀器響應）
inventory = client.get_stations(network=network, station=station,
                                location=location, channel=channel,
                                starttime=starttime, endtime=endtime,
                                level="response")
print(st)
print(inventory)

# # 去除儀器響應（轉換為地動速度，單位 m/s）
# st.remove_response(inventory=inventory, output="VEL",
#                    pre_filt=(0.001, 0.002, 8.0, 10.0))
# st.taper(0.05, type='hann')
# st.plot()
# # # 儲存處理後的波形
# st.write("YOJ_corrected.sac", format="SAC")
# %%
