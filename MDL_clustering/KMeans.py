__author__ = 'Administrator'
import math

import numpy as np

from MDL_clustering import Cluster


class KMeans(object):
    def __init__(self, k, n, iterationMax = 10, xmeans = False, verbose = False, restructure = False, iteration = 0,
                 flag = False, init = True, labels = None, **kw):
        self.k = k
        self.n = n
        self.iteration = iteration
        self.iterationMax = iterationMax
        self.xmeans = xmeans
        self.verbose = verbose
        self.restructure = restructure
        self.clusters = [None]*k
        self.lastClusterings = [[],[]]
        self.maxClusterCost = []
        self.initial = init
        for i in range(k):
            self.maxClusterCost.append(float(-np.inf))
        # self.update_representative()
        if self.initial == True:
            self.init()
        else:
            for k in range(self.k):
                self.clusters[k] = Cluster.Cluster(id = k, numElements = 0)
            for i in range(self.n.m):
                clusterID = labels[i]
                # print ", %d" %clusterID
                self.n.db[i].clusterID = clusterID
                self.clusters[clusterID].addElement(self.clusters[clusterID].numElements+1,self.n.db[i])
        if flag == True:
            self.xmeans = True
            for i in range(k):
                self.clusters[i].mean = kw.get('mean')[i]
                sd = kw.get('sd')[i]
                self.clusters[i].sd = sd
                self.clusters[i].var = math.pow(sd,2)


    def init(self):
        for i in range(self.n.m):
            if(i < self.k):
                self.n.db[i].clusterID = i
                self.clusters[i] = (Cluster.Cluster(id = i, numElements = 1, element = self.n.db[i]))
            else:
                clusterID = np.random.randint(0,self.k)
                # print ", %d" %clusterID
                self.n.db[i].clusterID = clusterID
                self.clusters[clusterID].addElement(self.clusters[clusterID].numElements+1,self.n.db[i])


    def run(self):
        done = False
        if(self.xmeans):
            self.assign_points()
        self.update_representative()

        while (not done) and (self.iteration < self.iterationMax):
            if self.verbose:
                self.printCls()

            clusterChanged = self.assign_points()
            self.update_representative()

            self.iteration += 1
            if not clusterChanged:
                done = True

        for i in self.n.db:
            if i.cost == float("inf"):
                i.cost = self.maxClusterCost[i.clusterID]

        if self.restructure:
            self.n.restructured = self.restructure_network()

        if self.verbose:
            self.printCls()


    def update_representative(self):
        for i in range(self.k):
            diag = 0.0
            diagC = 0
            cElements = self.clusters[i].elements
            for p in range(self.clusters[i].numElements):
                for q in range(p+1,self.clusters[i].numElements):
                    if self.n.network[cElements[p].number,cElements[q].number] != 0:
                        diag = diag + self.n.network[cElements[p].number,cElements[q].number]
                        diagC = diagC + 1
            mean = 0.0
            if diagC!=0:
                mean = float(diag)/diagC

            sdiag = 0.0
            for p in range(self.clusters[i].numElements):
                for q in range(p+1,self.clusters[i].numElements):
                    if self.n.network[cElements[p].number,cElements[q].number] != 0:
                        sdiag = sdiag + math.pow(mean - self.n.network[cElements[p].number,cElements[q].number],2)

            if diagC == 0 or (sdiag/diagC) < 1e-30:
                var = 1e-30
            else:var = sdiag/diagC

            if diagC == 0 or math.sqrt(sdiag/diagC) < 1e-30:
                sd = 1e-15
            else:sd = math.sqrt(var)
            self.clusters[i].mean = mean
            self.clusters[i].sd = sd
            self.clusters[i].var = var

            # print "Mean:%f Sd:%f"%(mean,sd)

    def assign_points(self):#assign point to cluster with the minimum cost
        clusterChanged = False
        elems = []
        numElements = []

        for i in range(self.k):
            elems.append([])
            numElements.append(0)

        for i in range(self.k):
            self.maxClusterCost[i] = 0

        for i in range(self.n.m):
            minCost = float("inf")
            aktClustID = 0
            for j in range(self.k):
                aktCost = self.get_cost(self.n.db[i],self.clusters[j])
                if aktCost < minCost:
                    minCost = aktCost
                    aktClustID = j
            if self.verbose and minCost == float("inf"):
                print("%d %f"%(self.n.db[i].ID,minCost))
            if minCost != float("inf") and minCost > self.maxClusterCost[aktClustID]:
                self.maxClusterCost[aktClustID] = minCost
            if self.n.db[i].clusterID != aktClustID:
                clusterChanged = True
            self.n.db[i].clusterID = aktClustID
            self.n.db[i].cost = minCost

            elems[aktClustID].append(self.n.db[i])
            numElements[aktClustID] = numElements[aktClustID] + 1

        for i in range(self.k):
            self.clusters[i].elements = elems[i]
            self.clusters[i].numElements = numElements[i]

        return clusterChanged

    def get_cost(self,d,cl):
        degree = 0
        for i in range(self.n.n):
            if self.n.network[d.number,i] != 0:
                degree = 1 + degree
        cost = 0.0
        countEdges = 0
        for i in range(cl.numElements):
            weight = self.n.network[d.number,cl.elements[i].number]
            if d.number != cl.elements[i].number and weight != 0:
                rest = 1/(math.sqrt(2*math.pi)*cl.sd)
                exp = math.exp(-math.pow(weight - cl.mean,2)/(2*cl.var))
                cost = cost + rest*exp
                countEdges = 1 + countEdges

        if countEdges == 0:
            return float("inf")

        if cl.numElements - 1==0:
            cost = 1e-30
        else:
            cost = cost / (cl.numElements - 1)
        if cost == 0:
            cost = 1e-30

        # print "Cost = %f,Degree = %d,Edges = %d"%(cost,degree,countEdges)
        return -math.log(cost,2) - 0.5*math.log(countEdges,2) + 0.5*math.log(degree,2)

    def check_cls_sim(self):
        i = 0
        clusterCount = 0
        for cluster in self.clusters:
            if cluster.numElements == self.lastClusterings[0][i].numElements:
                objectCount = 0
                for objecti in cluster.elements:
                    if objecti in self.lastClusterings[0][i].elements:
                        objectCount = 1+ objectCount
                if objectCount == cluster.numElements:
                    clusterCount = 1 + clusterCount
            i += 1

        if clusterCount == self.k:
            mdlCurrentClustering = 0
            for cluster in self.clusters:
                for obj in cluster.elements:
                    if obj.cost == float("inf"):
                        obj.cost = self.maxClusterCost[obj.clusterID]
                    mdlCurrentClustering += obj.cost
            mdlLastClustering = 0
            for cluster in self.lastClusterings[1]:
                for obj in cluster.elements:
                    if obj.cost == float("inf"):
                        obj.cost = self.maxClusterCost[obj.clusterID]
                    mdlLastClustering += obj.cost
            if mdlCurrentClustering > mdlLastClustering:
                self.clusters = self.lastClusterings[1]
                return False
            else:
                self.lastClusterings[0] = self.lastClusterings[1]
                self.lastClusterings[1] = self.clusters
                return True

    def restructure_network(self):
        restructured = np.zeros((self.n.m,self.n.n))
        newIndex = []

        for i in range(self.k):
            for obj in self.clusters[i].elements:
                newIndex.append(obj.number)

        # print newIndex

        counteri = 0
        for i in newIndex:
            counterj = 0
            for j in newIndex:
                restructured[counteri,counterj] = self.n.network[i,j]
            counteri += 1

        return restructured

    # def printCls(self):
    #     print "%d.iteration \n"%(self.iteration+1)
    #     for i in range(len(self.clusters)):
    #         print "Cluster %d:"%i
    #         for j in range(self.clusters[i].numElements):
    #             print "%d, "%self.clusters[i].elements[j].ID
    #         print "\r\n"
    #         print "%f +- %f    %d"%(self.clusters[i].mean,self.clusters[i].sd,self.clusters[i].numElements)
    #         print "\r\n"
