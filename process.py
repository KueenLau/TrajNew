import pandas as pd
import numpy as np
import os
import time
import math

import smopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import distance
import time
from rdp import rdp
import traj_dist.distance as dist
from MDL_clustering import XMeans,Network
from sklearn.decomposition import PCA
from metric_learn import LMNN

def gci(path):
    Const_Format = ["plt","csv","txt","npy"]
    file_list = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.split(".")[1] in Const_Format:
                file_list.append(os.path.join(root, name))
    return file_list

def get_sim_mats(data, n, xy_cols, dist_measure, t_col = None):
    '''
    path:input path
    dist_measure:{   1. 'sspd'

        Computes the distances using the Symmetrized Segment Path distance.

    2. 'dtw'

        Computes the distances using the Dynamic Path Warping distance.

    3. 'lcss'

        Computes the distances using the Longuest Common SubSequence distance

    4. 'hausdorf'

        Computes the distances using the Hausdorff distance.

    5. 'frechet'

        Computes the distances using the Frechet distance.

    6. 'discret_frechet'

        Computes the distances using the Discrete Frechet distance.

    7. 'sowd_grid'

        Computes the distances using the Symmetrized One Way Distance.

    8. 'erp'

        Computes the distances using the Edit Distance with real Penalty.

    9. 'edr'

        Computes the distances using the Edit Distance on Real sequence.
}
    '''
    # file_list = gci(path)
    # n = len(file_list)
    # print(n)
    # data = [pd.read_csv(x, skiprows=skip_rows, usecols=cols, header=0).dropna() for x in file_list]

    def get_space_sim(d):
        coors = [rdp(t[xy_cols].values, 1e-4) for t in d]
        spatial_mat = np.identity(n)
        spatial_dist = dist.pdist(coors, dist_measure)
        min = spatial_dist.min()
        spatial_dist = np.divide(spatial_dist - min, spatial_dist.max() - min)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                spatial_mat[i, j] = spatial_mat[j, i] = spatial_dist[k]
                k += 1
        # print(spatial_dist)
        return np.divide(1,1+spatial_mat)

    def get_time_sim(d, t_col=t_col):
        temporal_dist = []
        temporal_mat = np.identity(n)
        for i in range(n):
            t0 = d[i]
            time0 = t0[t_col]
            for j in range(i + 1, n):
                t1 = d[j]
                time1 = t1[t_col]
                dis = distance.time_dis(time0.values, time1.values)[0]
                temporal_dist.append(dis)
        temporal_dist = np.asarray(temporal_dist)
        min = temporal_dist.min()
        temporal_dist = np.divide(temporal_dist - min, temporal_dist.max() - min)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                temporal_mat[i, j] = temporal_mat[j, i] = temporal_dist[k]
                k += 1
        # print(temporal_mat)
        return np.divide(1,1+temporal_mat) #转换成相似度

    if t_col == None:
        return get_space_sim(data)
    sim_mat = np.multiply(get_space_sim(data), get_time_sim(data))
    return sim_mat

def init_clust(data, cutoff=0.7):
    net = Network.Network(network=data, cutoff=cutoff, flag=True)
    model = XMeans.XMeans(net, 2, 0)
    model.run(restructure=False)
    return model

def sse_compute(X, labels):
    sse = 0.0
    unique = list(set(list(labels)))
    c_num = len(unique)
    n_sample = len(labels)
    n = 0
    for k in range(c_num):
        ci = 0
        sum = 0.0
        for i in range(n_sample):
            if labels[i] == unique[k]:
                ci += 1
                for j in range(n_sample):
                    if labels[j] == unique[k]:
                        # print(X[i], X[j])
                        sum += np.sum((X[i] - X[j])**2)
                        # print(sum)
        sse += np.divide(sum,2*ci)
        # print("added kth cluster's squared error", sse)
    return sse


if __name__ == '__main__':
    '''
    whole procedure
    '''
    # s = get_sim_mats("/Users/kueen/Downloads/trajectory dataset/animal movement/vertebrate taxa",
    #                      ["location-long","location-lat"],0,["location-long","location-lat"],"dtw")
    # # pca = PCA(n_components=3)
    # # pca.fit(s)
    # # print(pca.explained_variance_ratio_)
    # # s_new = pca.transform(s)
    # clusters = init_clust(s)
    # labels = np.zeros((len(s),1))
    #
    # for i in range(len(clusters)):
    #     elements = clusters[i].elements
    #     for e in elements:
    #         labels[e.id] == e.clusterID
    #
    # print(labels)

    # lmnn = LMNN(k=2, min_iter=500, learn_rate=1e-6)
    # lmnn.fit(s_new, labels)
    # s_final = lmnn.transform(s_new)


    # sim_mat = np.loadtxt("Sim.txt")

    data = np.array([[1,1,1],[1.1,1.1,1.1],[2,2,2],[2.1,2.1,2.1],[2.2,2.2,2.2]])
    labels = [0,0,1,1,1]
    sse = sse_compute(data,labels)
    print(sse)
