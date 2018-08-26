__author__ = 'Administrator'

import numpy as np
import math

class Cluster(object):
    def __init__(self,id=0,numElements=0,element=None):
        self.ID = id
        self.mean = 0.0
        self.sd = 0.0
        self.var = math.pow(self.sd,2)
        self.numElements = numElements
        self.elements = []
        if element != None:
            self.elements.append(element)

    def addElement(self,numElements,element):
        self.numElements = numElements
        self.elements.append(element)