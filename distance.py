import numpy as np
import traj_dist.distance as dist
import pandas as pd
from pandas import DataFrame,Series
import math

def dtw_dis(t0, t1):
    return dist.e_dtw(t0, t1)


def time_dis(t0, t1):
    time_seq1 = [x.split(":")[0] for x in Series(t0)]
    time_seq2 = [x.split(":")[0] for x in Series(t1)]

    time_dtr1 = np.zeros((24,1)) + np.spacing(1)
    time_dtr2 = np.zeros((24,1)) + np.spacing(1)
    for _ in time_seq1:
        if _[0] == '0':
            idx = int(_[1])
            time_dtr1[idx] += 1
        else:
            idx = int(_)
            time_dtr1[idx] += 1

    for _ in time_seq2:
        if _[0] == '0':
            idx = int(_[1])
            time_dtr2[idx] += 1
        else:
            idx = int(_)
            time_dtr2[idx] += 1

    time_dtr1 /= time_dtr1.sum()
    time_dtr2 /= time_dtr2.sum()
    # print(time_dtr1,time_dtr2)
    return KLD(time_dtr1,time_dtr2)

def KLD(p,q):
    # p,q=zip(*filter(lambda x,y: x!=0 or y!=0, zip(p,q)))
    p=p+np.spacing(1)
    q=q+np.spacing(1)
    # print(p,q)
    return sum([_p * math.log(_p/_q,2) for (_p,_q) in zip(p,q)])

#JSD=1/2*KL(P||(P+Q)/2) +1/2*KL(Q||(P+Q)/2)
def JSD_core(p,q):
    M = [0.5*(_p+_q) for _p,_q in zip(p,q)]
    p=p+np.spacing(1)
    q=q+np.spacing(1)
    M=M+np.spacing(1)
    return 0.5*KLD(p,M)+0.5*KLD(q,M)








