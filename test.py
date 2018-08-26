import smopy
from metric_learn import LMNN
from rdp import rdp
import pandas as pd
from pandas import DataFrame,Series
import matplotlib.pyplot as plt
import numpy as np
from process import gci

# file_list = gci("/Users/kueen/Downloads/trajectory dataset/animal movement/vertebrate taxa/Movement syndromes across vertebrate taxa (data from Abrahms et al. 2017)-gps.csv")
# print(file_list[12],file_list[20])
# data = pd.read_csv("/Users/kueen/Downloads/trajectory dataset/animal movement/vertebrate taxa/Movement syndromes across vertebrate taxa (data from Abrahms et al. 2017)-gps.csv")
# id = data["individual-local-identifier"].values
# id = list(set(id))
# for i in range(len(id)):
#     tmp = data[data["individual-local-identifier"]==id[i]]
#     tmp.to_csv("/Users/kueen/Downloads/trajectory dataset/animal movement/vertebrate taxa/"
#     + id[i] + ".csv")
# data = pd.read_csv("/Users/kueen/Downloads/trajectory dataset/animal movement/vertebrate taxa/Movement syndromes across vertebrate taxa (data from Abrahms et al. 2017)-gps.csv"
#                    , usecols=["timestamp", "location-long", "location-lat", "individual-local-identifier"])
# data = data[data["comments"]=="LI06_LH364"]
# coors = data[["location-long", "location-lat"]].values
# plt.plot(coors[:,0],coors[:,1])
# plt.show()
# file_list = gci("/Users/kueen/Downloads/trajectory dataset/animal movement/vertebrate taxa/")
# file_list = [x for x in file_list
#              if x.split("/")[7].split(" ")[0] == "jackal" or
#              x.split("/")[7].split(" ")[0] == "elephant" or
#              x.split("/")[7].split(" ")[0] == "springbok"]
# for i in range(len(file_list)):
#     data = pd.read_csv(file_list[i], usecols=["timestamp", "location-long", "location-lat"])
#     data = data.dropna(axis=0, how="any")
#     coors = data[["location-long", "location-lat"]].values
#     if file_list[i].split("/")[7].split(" ")[0] == "jackal":
#         plt.plot(coors[:,0],coors[:,1],color="blue")
#     elif file_list[i].split("/")[7].split(" ")[0] == "elephant":
#         plt.plot(coors[:,0],coors[:,1],color="red")
mat = np.loadtxt("truck_sim")
lmnn = LMNN(k=2, min_iter=100, learn_rate=1e-6)

label = np.zeros(mat.shape[0])
print(label)