__author__ = 'Administrator'
import time
import numpy as np
from metric_learn import itml,lmnn
from sklearn.metrics import adjusted_mutual_info_score

from MDL_clustering import Network, XMeans


class Adaptive_cluster(object):
    def __init__(self,simMat,constraints,restructured=False,max_iters=1000,beta=1e-1,verbose=False,
                 data=None,init_cls_cnt=2,cutoff=None,true_labels=None):
        self.original_Mat = simMat
        self.simMat = self.original_Mat.copy()
        self.n = len(simMat)
        self.max_iters = max_iters
        self.verbose = verbose
        self.lastLabels = np.zeros((self.n,1))
        self.trueLabels = true_labels
        self.restructured = restructured
        self.metric_paras = {
            'Mat':np.eye(self.n),
            'beta':beta,
            'data':data,
            'constraints':constraints
        }
        self.cls_paras = {
            'init_cluster_cnt':init_cls_cnt,
            'cutoff':cutoff,
        }
        if cutoff == None:
            self.cls_paras['cutoff'] = 0.7
        if data == None:
            self.metric_paras['data'] = []

    def run(self):
        start = time.time()
        print("This turn's cutoff is : %f"%self.cls_paras['cutoff'])
        distance = self.simMat.copy()
        labels = self.clustering(similarity=distance,init_cls_flg=True)
        end = time.time()
        print("CLUSTERING COSTS " + str(end-start) + " seconds.......")
        labels = np.asarray(labels.flat)
        print("Labels are : \n", labels, "\n", np.unique(labels), "\n")
        if self.trueLabels != None:
            print("AMI IS:%f"%adjusted_mutual_info_score(labels,self.trueLabels))

        iters = 0
        labelChanged = True
        while(iters < self.max_iters and labelChanged):
            #------------itml------------------------#
            # Ma = self.metric_learn(data,labels)
            # for i in range(self.n):
            #     distance[i,i]=1
            #     for j in range(i+1,len(data)):
            #         distance[j,i]=distance[i,j]=np.exp(-(data[i]-data[j]).dot(Ma).dot(data[i]-data[j]))
            #----------------------------------------#

            start = time.time()
            self.simMat = self.nonVec_metric(self.simMat,labels)
            end = time.time()
            print("UPDATING COSTS " + str(end-start) + " seconds.......")
            model_cost = np.sum(np.absolute(self.simMat - self.original_Mat))  # to caculate |D0-D1| for model cost
            # self.cls_paras['cutoff'] = 0.8
            print("This turn's cutoff is : %f"%self.cls_paras['cutoff'])
            start = time.time()
            distance = self.simMat.copy()
            newLabels = self.clustering(distance, model_cost, labels, False)
            end = time.time()
            print("CLUSTERING COSTS " + str(end-start) + ".......")

            newLabels = np.asarray(newLabels.flat)
            print("Labels are : \n", newLabels, "\n", np.unique(newLabels), '\n')
            if self.trueLabels != None:
                print("AMI IS:%f"%adjusted_mutual_info_score(newLabels,self.trueLabels))

            if np.array_equal(newLabels,labels):
                labelChanged = False
            labels = newLabels
            iters += 1
        self.lastLabels = labels


    def metric_learn(self,data,labels):
        learner = itml.ITML_Supervised()
        model = learner.fit(data,labels)
        Ma = model.A
        return Ma

    def nonVec_metric(self,simMat,labels):
        simMat = simMat
        length = simMat.shape[0]
        dim = simMat.shape[1]-1

        deletion = range(0,length**2,length+1)
        vecMat = np.delete(simMat,deletion).reshape((length,dim))

        deltaMat = np.zeros((length,dim))
        labelMat = np.zeros((length,length))

        for i in range(length):
            for j in range(length):
                if labels[i] == labels[j]:
                    labelMat[i,j] = 1
                else:labelMat[i,j] = 0
        deletion = range(0,length**2,length+1)
        labelMat = np.delete(labelMat,deletion).reshape((length,dim))

        num = np.sum(labelMat,axis=1)

        for i in range(length):
            diff = np.tile(vecMat[i],length).reshape((length,dim)) - vecMat
            clusterDiff = diff * labelMat
            sumDim = np.sum(clusterDiff,axis=0)
            sinValue = np.sin(sumDim)
            if num[i] > 1:
                deltaMat[i] = 1/float(num[i])*sinValue
        vecMat = vecMat + deltaMat
        insertion = range(0,length*dim+1,length)
        simMat = np.insert(vecMat,insertion,1.0).reshape((length,length))
        return simMat

    def restructureNetwork(self,km):
        restructured = np.zeros((self.n.m,self.n.n))
        newIndex = []
        km.cutPoints = []

        for i in range(km.k):
            for obj in km.finalClusters[i].elements:
                newIndex.append(obj.ID)
            if i > 0 and i < km.k-1:
                km.cutPoints.append(km.cutPoints[i-1] + km.finalClusters[i].numElements)
            elif i < km.k - 1:
                km.cutPoints.append(km.finalClusters[i].numElements)

        counteri = 0
        for i in newIndex:
            counterj = 0
            for j in newIndex:
                restructured[counteri,counterj] = km.n.network[i,j]
                counterj += 1
            counteri += 1

        return restructured

    def clustering(self, similarity, mdl_cost = 0, oldLabels = None, init_cls_flg = False):
        model_cost = mdl_cost
        if init_cls_flg == True:
            cutoff = self.cls_paras['cutoff']
            net = Network.Network(similarity, cutoff=cutoff, flag=True)
            km = XMeans.XMeans(net, self.cls_paras['init_cluster_cnt'], model_cost)
            km.run(False)
            # print("First turn's modularity equals:"+str(km.modality))
            self.cls_paras['init_cluster_cnt'] = km.clusterCount
            labels = np.zeros((1,self.n),dtype=int)
             # print km.clusterCount
            for i in range(km.clusterCount):
                for j in range(km.finalClusters[i].numElements):
                    labels[0,km.finalClusters[i].elements[j].number] = i
        else:
            cutoff = self.cls_paras['cutoff']
            net = Network.Network(similarity, cutoff=cutoff, flag=True)
            km = XMeans.XMeans(net, self.cls_paras['init_cluster_cnt'], model_cost, init=False, labels=oldLabels)
            km.run(False)
            # print("This turn's modularity equals:"+str(km.modality))
            self.cls_paras['init_cluster_cnt'] = km.clusterCount
            labels = np.zeros((1,self.n),dtype=int)
            for i in range(km.clusterCount):
                for j in range(km.finalClusters[i].numElements):
                    labels[0,km.finalClusters[i].elements[j].number] = i
        return labels

    @staticmethod
    def get_similarity(data,delta):
        distance = np.eye(len(data))
        for i in range(len(data)):
            for j in range(i+1,len(data)):
                distance[j,i]=distance[i,j]=np.exp(-np.linalg.norm(data[i]-data[j])/2*delta**2)
        return distance

    @staticmethod
    def get_rbf(data,delta):
        return np.exp(-data/2*delta**2)

if __name__ == '__main__':
    data = np.loadtxt(r'/Users/kueen/Documents/Traj_DM/MDL_clustering/MatrixR.txt', delimiter=' ')
    # simMat = Adaptive_cluster.get_rbf(data,1)
    mdl = Adaptive_cluster(simMat = data,constraints=1,cutoff=0.7)
    mdl.run()
