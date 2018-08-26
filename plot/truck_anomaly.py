import numpy as np
import matplotlib.pyplot as plt
from process import *
import pandas as pd
from pandas import DataFrame,Series

n = 0
file_list = gci("/Users/kueen/Downloads/trajectory dataset/trucks/discritized")
#画出轨迹网格离散后的效果

compare = [1,99]

all_trajs = []
for _ in compare:
    with open(file_list[_]) as f:
        t = f.readlines()
        # traj = [x[1:-2].split(" ") for x in t]
        traj = [x.split(',') for x in t]
        df = DataFrame(traj)
        # print(df)
        for i in range(6):
            col = df[i].astype('int')
            df[i] = col

    # col = df[0].astype('int')
    # df[0] = col
    # col = df[3].astype('float64')
    # df[3] = col
    # col = df[4].astype('float64')
    # df[4] = col

        all_trajs.append(df)
print(all_trajs[0])
fig, ax = plt.subplots()
ax.set_title('id = 1')
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
r1 = ax.plot(all_trajs[0][0].values, all_trajs[0][1].values, '*-')
r2 = ax.plot(all_trajs[1][0].values, all_trajs[1][1].values, 'ro-')
ax.legend((r1[0],r2[0]), ('id = 1','id = 99'), loc = 'upper left')
plt.show()


