from process import *
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
import smopy

n = 0
file_list = gci("/Users/kueen/Downloads/trajectory dataset/trucks/sub")
'''
truck clustering
'''
all_trajs = []
# for i in range(len(file_list)):
#     with open(file_list[i]) as f:
#         t = f.readlines()
#         traj = [x[1:-2].split(" ") for x in t]
#         df = DataFrame(traj)
#         # print(df)
#         col = df[0].astype('int')
#         df[0] = col
#         col = df[3].astype('float64')
#         df[3] = col
#         col = df[4].astype('float64')
#         df[4] = col
#         all_trajs.append(df)
#
# labels = np.loadtxt("../truck_labels")
# labels = labels.astype("int32")
# unique = list(set(list(labels)))
cdict = {0:'#9ff113', 1:'#5fbb44', 3:'#f5f329', 4:'#e50b32'}

with open(file_list[1]) as f:
    t = f.readlines()
    traj = [x[1:-2].split(" ") for x in t]
    df = DataFrame(traj)
    # print(df)
    col = df[0].astype('int')
    df[0] = col
    col = df[3].astype('float64')
    df[3] = col
    col = df[4].astype('float64')
    df[4] = col
    all_trajs.append(df)
plt.title('id = 1')
plt.xlabel('longitude')
plt.ylabel('latitude')
for i in range(len(all_trajs)):

    plt.plot(all_trajs[i][3],all_trajs[i][4],color=cdict[1])

plt.show()

