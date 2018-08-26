__author__ = 'Administrator'
import numpy as np
import sys

class Dataobject(object):
    def __init__(self,num=0,id=0,trueClusterID=0,cost=float(np.inf),clusterID=0):
        self.number = num
        self.clusterID = clusterID
        self.trueClusterID = trueClusterID
        self.cost = cost
        self.ID = id