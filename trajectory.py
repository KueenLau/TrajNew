import pandas as pd
import numpy as np
from pandas import DataFrame,Series

class Traj(object):
    def __init__(self, coor, time):
        self.coordinates = coor
        self.time = time
        self.time_dtr


