import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from process import *


n = 0
file_list = gci("/Users/kueen/Downloads/trajectory dataset/trucks/distance")
print(file_list)
all_trajs = []

with open(file_list[1]) as f:
        t = f.readlines()
        traj = [x.split(",") for x in t]
        df = DataFrame(traj)
        # print(df)
        col = df[1].astype('float64')
        df[1] = col
        all_trajs.append(df)

df = df.sort(columns=[1])
print(df)