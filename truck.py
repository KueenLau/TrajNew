import pandas as pd
from process import gci
import time

'''
   truck data format：
   befor:obj-id, traj-id, date(dd/mm/yyyy), time(hh:mm:ss), lat, lon, x, y
   after:obj-id, date(dd/mm/yyyy), time(hh:mm:ss), lat, lon
   '''
# data = pd.read_csv("/Users/kueen/Downloads/trajectory dataset/trucks/Trucks.txt",delimiter=';',
#                    header=None, usecols=[0,2,3,4,5]).dropna()
#
# obj_set = list(set(data[0].values))
# for i in range(len(obj_set)):
#     tmp = data[data[0]==obj_set[i]]
#     tmp.to_csv("/Users/kueen/Downloads/trajectory dataset/trucks/"+str(obj_set[i])+".csv")


# print(file_list)

# total = 0
# for i in range(len(file_list)):
#     data = pd.read_csv(file_list[i],header=None)
#     data = data.values[1:,1:]
#     # print(data)
#     t = data[:,1] + " " + data[:,2]
#     n = 0
#     obj_id = data[0,0]
#     print("object ID is " + str(obj_id))
#     split_head = 0
#     for j in range(len(t)):
#         if j == len(t)-1:
#             break
#         time_arr = time.strptime(t[j],"%d/%m/%Y %H:%M:%S")
#         timestamp_front = time.mktime(time_arr)
#         time_arr = time.strptime(t[j+1],"%d/%m/%Y %H:%M:%S")
#         timestamp_after = time.mktime(time_arr)
#         if abs(timestamp_after - timestamp_front) > 900:
#             print(split_head,j)
#             tmpd = data[split_head:j].copy()
#             print(tmpd)
#             with open("/Users/kueen/Downloads/trajectory dataset/trucks/sub/"+str(obj_id)+
#                                    "_"+str(n)+".txt","w") as f:
#                 for _ in tmpd:
#                     f.write(str(_))
#                     f.write('\r\n')
#             n += 1
#             split_head = j + 1
#
#
#
#     print(file_list[i] + str(n) + " subtrajs")
#     total += n
# print("-----" , total)
import os
# path = "/Users/kueen/Downloads/trajectory dataset/trucks/sub"
# for i in os.listdir(path):
#     path_file = os.path.join(path,i)
#     if os.path.getsize(path_file) < 1024:
#         os.remove(path_file)


from process import *
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from pandas import DataFrame,Series
from sklearn.cluster import DBSCAN

import random

n = 0
file_list = gci("/Users/kueen/Downloads/trajectory dataset/trucks/sub")
'''
truck clustering
'''
def EuclideanDistances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A,BT)
    # print(vecProd)
    SqA =  A**2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0
    ED = np.sqrt(SqED)
    return ED
# ----------------count points and trajs--------------
# print("Total traj number is ", len(file_list))
#
# point_count = 0
# # all_trajs = []
# for i in range(len(file_list)):
#     with open(file_list[i]) as f:
#         t = f.readlines()
#         traj = []
#         for x in t:
#             # tmp = x[1:-2].split(" ")
#             # traj.append(tmp)
#             point_count += 1
#         # df = DataFrame(traj)
#         # # print(df)
#         # col = df[0].astype('int')
#         # df[0] = col
#         # col = df[3].astype('float64')
#         # df[3] = col
#         # col = df[4].astype('float64')
#         # df[4] = col
#         # all_trajs.append(df)
# print("Total point number is ", point_count)
# file_list = random.sample(file_list, int(len(file_list)))
all_trajs = []
for i in range(len(file_list)):
    with open(file_list[i]) as f:
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
print(type(all_trajs[0].iloc[1,3]))
start = time.time()
sim_mat_picked = get_sim_mats(all_trajs, len(file_list), [3,4], "dtw")
# # np.savetxt("truck_sim",sim_mat_picked)
#
# ###直接load相似矩阵
# # # sim_mat = np.loadtxt("truck_sim")
# cutoff = np.percentile(sim_mat_picked,50)
# print("Shape is " + str(sim_mat_picked.shape))
# cluster_model = init_clust(sim_mat_picked,cutoff)
# end = time.time()
# execution_time = end - start
# print("Execution time is ", execution_time)
# clusters = cluster_model.finalClusters
# modularity = cluster_model.modality
# print(modularity)
# labels = np.zeros(sim_mat_picked.shape[0])
#
# #获取label
# for i in range(len(clusters)):
#     elements = clusters[i].elements
#     for e in elements:
#         labels[e.ID] = i
# labels = labels.astype("int32")
#
# distance = np.linalg.norm(sim_mat_picked[0] - sim_mat_picked[1])
# print(distance)
# dis = EuclideanDistances(sim_mat_picked, sim_mat_picked)
# print(dis)
#
#
#
# #DBSCAN聚类
# db = DBSCAN(eps=0.8, min_samples=20).fit(sim_mat_picked)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
# print(labels)
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# print('Estimated number of clusters: %d' % n_clusters_)
# print('Number of clusters plus anomalies: ', len(set(labels)))

# np.savetxt("truck_labels",labels)
# print("Labels are " + str(labels))
# sse_old = sse_compute(sim_mat_picked,labels)
# print("Old sse is ", sse_old)
# c_num = len(set(list(labels)))

# 维数太高，直接lmnn内存溢出
# pca = PCA(0.9)
# pca.fit(sim_mat)
# print(pca.explained_variance_ratio_)
# fea_vectors = pca.transform(sim_mat)
# sim_mat_pca = np.divide(1,1+euclidean_distances(fea_vectors, fea_vectors))

# lmnn = LMNN(k=3, min_iter=10, learn_rate=1e-8)
# lmnn.fit(sim_mat_picked, labels)
# s_final = lmnn.transform(sim_mat_picked)
# np.savetxt("ml_mat",s_final)
#
# cutoff = np.percentile(s_final,50)
# clusters_newmodel = init_clust(s_final,cutoff)
# clusters_new = clusters_newmodel.finalClusters
# print(clusters_newmodel.modality)
# labels_new = []
#
# for i in range(len(clusters_new)):
#     elements = clusters_new[i].elements
#     for e in elements:
#         labels_new.append(e.clusterID)
#
# labels_new = np.array(labels_new)
# print("Labels are " + str(labels_new))
# sse_new = sse_compute(s_final, labels_new)
# print("New sse is ", sse_new)
#
# count = 0
# for x, y in zip(labels,labels_new):
#     if x != y:
#         count += 1
# print(count)
#
#
#
