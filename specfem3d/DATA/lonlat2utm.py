import pandas as pd
import utm
import numpy as np

tomo_file = "tomo_files/tomography.xyz"

df = pd.read_csv(tomo_file, sep='\s+', skiprows=4, 
                 names=['lon', 'lat', 'dep', 'vp', 'vs', 'rho'])

lon_arr = np.array(df.lon)
lat_arr = np.array(df.lat)

# print(lat_arr)

dep_arr = np.array(df.dep) * 1e+03
results = utm.from_latlon(lat_arr, lon_arr)
# print(results)

df_new = pd.DataFrame({
    'utm_x': results[0].round(3),
    'utm_y': results[1].round(3),
    'depth': (dep_arr * -1.0).round(3),
    'vp': df.vp.round(3),
    'vs': df.vs.round(3),
    'rho': df.rho.round(3)
})

print(np.min(df_new['vp']))
df_new[df_new['vp'] < 2600] = 2600
df_new[df_new['vs'] < 1500] = 1500
print(np.min(df_new['vp']))


# df_new = pd.DataFrame(zip(results[0], results[1], dep_arr*-1.0, df.vp, df.vs, df.rho))
# print(df_new)
#df_new.to_csv('test.txt', index=False, sep=' ', header=False)
