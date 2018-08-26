# -*- coding: utf-8 -*-
__author__ = 'Administrator'
import numpy as np
import math

from MDL_clustering import KMeans, Dataobject, Network


class XMeans(object):
    def __init__(self, n, kMin, cost, init = True, labels = None):
        self.n = n
        self.kMax = n.m
        self.k = kMin
        self.minNumElements = 2
        self.modality = 0
        self.clusterCount = 0
        self.verbose = True
        self.restructure = False
        self.finalClusters = []
        self.cutPoints = []
        self.init = init
        if labels is None:
            self.labels = []
        else:self.labels = labels
        if cost == 0:
            self.deltaS = 1
        else:self.deltaS = cost

    def run(self,restructure):
        self.restructure = restructure

        if self.verbose:
            print("INITIAL CLUSTERING...")
        km = KMeans.KMeans(k=self.k, n=self.n, restructure=restructure, init=self.init, labels=self.labels)
        km.run()

        if self.verbose:
            print("STARTING ITERATIVE CLUSTER SPLITTING WITH k = %d\n"%self.k)
        self.splitClusters(km=km)

        if restructure:
            self.n.restructured = self.restructureNetwork()

        if self.verbose:
            print("\n\n %d CLUSTERS WERE FOUND"%len(self.finalClusters))
            for i in range(len(self.finalClusters)):
                print("Cluster %d : %f -+ %f     %d"%(i,self.finalClusters[i].mean,self.finalClusters[i].sd,self.finalClusters[i].numElements))

        self.modality = self.getModularity()

    def splitClusters(self,km):
        l = len(km.clusters)
        for i in range(l):
            cluster = km.clusters[i]
            print(cluster.numElements, self.kMax)
            if cluster.mean != 0.0 and cluster.numElements > self.minNumElements:
                mean = [cluster.mean + cluster.sd,cluster.mean - cluster.sd]
                sd = [cluster.sd/2,cluster.sd/2]
                netw = self.initNetwork(cluster, km.n.network)

                kTmp = 2
                kmSplit = KMeans.KMeans(k=kTmp, n=netw, flag=True, mean=mean, sd=sd)
                kmSplit.run()

                mdl = self.getMDL(cluster.elements)
                print("unseparated cluster cost = ", mdl)
                mdlSplit = self.getMDL(kmSplit.n.db)
                print("L(D|M) = ", mdlSplit)
                p = cluster.numElements / float(self.kMax)
                print(p)
                # model_cost = math.log(self.deltaS, 2) + math.log(self.clusterCount + 1, 2) \
                #              - 0.5 * self.kMax * (p * math.log(p, 2) + (1 - p) * math.log((1 - p), 2))
                model_cost = 64
                print("L(M) = ", model_cost)
                mdlSplit += model_cost
                print("L(D|M) + L(M) = ", mdlSplit)

                if self.verbose:
                    print("MDL = %f ; MDLSPLIT = %f ; k = %d \n"%(mdl,mdlSplit,self.k))

                if abs(mdlSplit) < abs(mdl) and self.k < self.kMax and not(kmSplit.clusters[0].mean == 0.0 or kmSplit.clusters[1].mean == 0.0) and kmSplit.n.m > self.minNumElements:
                    self.k += 1
                    if self.verbose:
                        print("SPLITTING DATA k = %d \n"%self.k)
                    self.splitClusters(kmSplit)
                else:
                    self.addFinalClusters(cluster)
            else:
                self.addFinalClusters(cluster)

    def initNetwork(self,cls,net):
        network = Network.Network(network = np.zeros((cls.numElements, cls.numElements)), db =[None] * cls.numElements)
        counterj = 0
        for j in cls.elements:
            network.db[counterj] = Dataobject.Dataobject(counterj, j.ID)
            network.db[counterj].trueClusterID = j.trueClusterID
            counterl = 0
            for l in cls.elements:
                network.network[counterj,counterl] = net[j.number,l.number]
                counterl += 1
            counterj += 1
        return network

    def getMDL(self,db):
        mdl = 0.0
        for d in db:
            mdl += d.cost
        return mdl

    def addFinalClusters(self,cluster):
        cluster.ID = self.clusterCount
        for o in cluster.elements:
            o.clusterID = cluster.ID
        self.finalClusters.append(cluster)
        self.clusterCount += 1

    def restructureNetwork(self):
        restructured = np.zeros((self.n.m,self.n.n))
        newIndex = []
        self.cutPoints = []

        for i in range(self.k):
            for obj in self.finalClusters[i].elements:
                newIndex.append(obj.ID)
            if i > 0 and i < self.k-1:
                self.cutPoints.append(self.cutPoints[i-1] + self.finalClusters[i].numElements)
            elif i < self.k - 1:
                self.cutPoints.append(self.finalClusters[i].numElements)

        counteri = 0
        for i in newIndex:
            counterj = 0
            for j in newIndex:
                restructured[counteri,counterj] = self.n.network[i,j]
                counterj += 1
            counteri += 1

        return restructured

    def getModularity(self):##修改了权重之后权重趋于相等，用modularity来衡量不太合适
        mod = 0.0
        L = 0.0
        for i in range(self.n.n):
            for j in range(self.n.n):
                if i != j:
                    if self.n.network[i,j] != 0 or not np.isnan(self.n.network[i,j]):
                        L += self.n.network[i,j]
        L /= 2.0

        for i in range(self.k):
            ls = 0.0
            ds = 0.0
            for obji in self.finalClusters[i].elements:
                for objj in self.n.db:
                    if obji.ID != objj.ID and not np.isnan(self.n.network[obji.ID,objj.ID]):
                        ds += self.n.network[obji.ID,objj.ID]

                for objj in self.finalClusters[i].elements:
                    if obji.ID != objj.ID and not np.isnan(self.n.network[obji.ID,objj.ID]):
                        ls += self.n.network[obji.ID,objj.ID]

            ls /= 2.0
            ds /= 2.0
            mod += (ls/L - np.power(ds/L,2))
        return mod

if __name__ == '__main__':
    data = np.loadtxt(r'/Users/kueen/Documents/Traj_DM/MDL_clustering/MatrixR.txt', delimiter=' ')
    net = Network.Network(network=data, cutoff=0.7,flag=True)
    model = XMeans(net, 2, 0, True)
    model.run(restructure=False)
    print(model.finalClusters)

