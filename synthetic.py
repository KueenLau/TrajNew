import numpy as np
import matplotlib.pyplot as plt
from process import *
import pandas as pd
from pandas import DataFrame,Series

def gen_syn_trajs():
    cand_1 = np.array([
        [1,8],
        [0,4],
        [3,0.5],
    ],dtype=float)

    cand_2 = np.array([
        [5,6],
        [4.5,4],
        [4,1.5]
    ],dtype=float)

    cand_3 = np.array([
        [0,4],
        [3,3],
        [6,5]
    ],dtype=float)
    np.save(r"syntheticData/tc1",cand_1)
    np.save(r"syntheticData/tc2",cand_2)
    np.save(r"syntheticData/tc3",cand_3)

    R = 1
    for i in range(99):
        t1 = cand_1.copy()
        for _ in t1:
            disturb = R * np.random.uniform(-0.2,0.2,2)
            _ += disturb
        np.save(r"syntheticData/tc1_"+str(i), t1)
        t2 = cand_2.copy()
        for _ in t2:
            disturb = R * np.random.uniform(-0.2,0.2,2)
            _ += disturb
        np.save(r"syntheticData/tc2_"+str(i), t2)
        t3 = cand_3.copy()
        for _ in t3:
            disturb = R * np.random.uniform(-0.2,0.2,2)
            _ += disturb
        np.save(r"syntheticData/tc3_" + str(i), t3)

# gen_syn_trajs()

file_list = gci("syntheticData")
all_trajs = []
for i in range(len(file_list)):
    df = DataFrame(np.load(file_list[i]))
    all_trajs.append(df)
start = time.time()
sim_mat_picked = get_sim_mats(all_trajs, len(file_list), [0,1], "dtw")
print(sim_mat_picked)

cutoff = 0.7
# print("Shape is " + str(sim_mat.shape))
cluster_model = init_clust(sim_mat_picked,cutoff)
end = time.time()
execution_time = end - start
print("Execution time is ", execution_time)
clusters = cluster_model.finalClusters
modularity = cluster_model.modality
print(modularity)
labels = np.zeros(sim_mat_picked.shape[0])

for i in range(len(clusters)):
    elements = clusters[i].elements
    for e in elements:
        labels[e.ID] = i
labels = labels.astype("int32")
np.savetxt("synthetic_labels",labels)
print("Labels are " + str(labels))
sse_old = sse_compute(sim_mat_picked,labels)
print("Old sse is ", sse_old)