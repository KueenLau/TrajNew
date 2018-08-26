__author__ = 'Administrator'
import numpy as np

from MDL_clustering import Dataobject


class Network(object):
    def __init__(self,network,db=None,cutoff=float(-np.inf),flag=False):
        self.cutoff = cutoff
        self.network = network
        self.n = len(self.network)
        self.m = len(self.network)
        if db == None:
            self.db = []
        else:self.db = db
        self.restructured = []
        self.flag = flag

        if self.flag == True:
            for i in range(self.n):
                for j in range(self.m):
                    value = float(self.network[i,j])
                    if value > self.cutoff or value < -self.cutoff:
                        self.network[i,j] = value
                    else:self.network[i,j] = 0
                self.db.append(Dataobject.Dataobject(num=i, id=i))

        # print "Matrix[%dx%d]" %(self.n,self.m)