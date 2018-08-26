from process import *
import pandas as pd
from pandas import DataFrame,Series
import numpy as np

file_list = gci("../syntheticData")
labels = np.loadtxt("synthetic_labels")
cdict = {0:'#9ff113', 1:'#5fbb44', 2:'#f5f329', 3:'#e50b32'}
#45条噪声轨迹
noise = []
for i in range(45):
    traj = np.random.normal(2.5,2,size=(6,2))
    print(traj)
    noise.append(traj)

plt.subplot(111)
# plt.axis('off')
frame = plt.gca()
# y 轴不可见
frame.axes.get_yaxis().set_visible(False)
# x 轴不可见
frame.axes.get_xaxis().set_visible(False)

for _ in noise:
    plt.plot(_[:,0], _[:,1], color='grey')

for i in range(len(file_list)):
    t = np.load(file_list[i])
    plt.plot(t[:, 0], t[:, 1], color=cdict[labels[i]])
plt.show()