import numpy as np
import matplotlib.pyplot as plt
from process import *
import pandas as pd
from pandas import DataFrame,Series

file_list = gci("/Users/kueen/Downloads/trajectory dataset/trucks/sub")
with open("/Users/kueen/Downloads/trajectory dataset/trucks/Outliers_k10") as f:
    outlier_file = f.readlines()
outlier_file = ['/Users/kueen/Downloads/trajectory dataset/trucks/sub/'+x.strip('\n') for x in outlier_file]
print(outlier_file)

normal_file = list(set(file_list).difference(set(outlier_file)))
print(len(normal_file))

normal_trajs = []
n = len(normal_file)
m = len(outlier_file)
for i in range(n):
    with open(normal_file[i]) as f:
        t = f.readlines()
        traj = [x[1:-2].split(" ") for x in t]
        df = DataFrame(traj)
        col = df[0].astype('int')
        df[0] = col
        col = df[3].astype('float64')
        df[3] = col
        col = df[4].astype('float64')
        df[4] = col

        normal_trajs.append(df)

outlier_trajs = []
for i in range(m):
    with open(outlier_file[i]) as f:
        t = f.readlines()
        traj = [x[1:-2].split(" ") for x in t]
        df = DataFrame(traj)
        col = df[0].astype('int')
        df[0] = col
        col = df[3].astype('float64')
        df[3] = col
        col = df[4].astype('float64')
        df[4] = col

        outlier_trajs.append(df)


fig, ax = plt.subplots()
for i in range(n):
    ax.plot(normal_trajs[i][3], normal_trajs[i][4], 'b')

ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
ax.plot(outlier_trajs[3][3], outlier_trajs[3][4], 'r', linewidth=4)
ax.set_title('id = 1017')
plt.show()