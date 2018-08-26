from process import *
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from pandas import DataFrame,Series
from sklearn.cluster import DBSCAN

import random

n = 0
file_list = gci("/Users/kueen/Downloads/trajectory dataset/trucks/sub")
#所有轨迹
all_trajs = []
min_lat = np.inf
max_lat = -np.inf
min_lon = np.inf
max_lon = -np.inf

for i in range(len(file_list)):
    with open(file_list[i]) as f:
        t = f.readlines()
        traj = [x[1:-2].split(" ") for x in t]
        df = DataFrame(traj)
        # print(df)
        col = df[0].astype('int')
        df[0] = col
        col = df[3].astype('float64')

        #时间商界
        max_time = df[1]+""

        #一条轨迹里最小的纬度
        min_val = col.min()
        if min_val < min_lat:
            min_lat = min_val
        # 一条轨迹里最小的纬度
        max_val = col.max()
        if max_val > max_lat:
            max_lat = max_val
        df[3] = col

        col = df[4].astype('float64')
        #一条轨迹里最小的经度
        min_val = col.min()
        if min_val < min_lon:
            min_lon = min_val
        # 一条轨迹里最小的经度
        max_val = col.max()
        if max_val > max_lon:
            max_lon = max_val
        df[4] = col
        all_trajs.append(df)
print(type(all_trajs[0].iloc[1,3]))
print("The MIN_LAT is %f,\n The MAX_LAT is %f,\n The MIN_LON is %f,\n The MAX_LON is %f"
      %(min_lat,max_lat,min_lon,max_lon))

