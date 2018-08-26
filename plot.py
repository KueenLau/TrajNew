import matplotlib.pyplot as plt
import pandas as pd
from rdp import rdp
import os
import smopy

if __name__ == "__main__":
    file = "/Users/kueen/Downloads/plotdata"
    file_list = os.listdir(file)
    file_list = [file + "/" + x for x in file_list if x.split(".")[1] != "DS_Store"]
    print(file_list)

    data = [pd.read_csv(x, skiprows=6, usecols=[0,1,6], header=None) for x in file_list]

    coors = [rdp(x[[0,1]].values, 1e-4) for x in data]

    hz = smopy.Map((30.0, 115.0, 41.55, 122.5), z=10)
    ax = hz.show_mpl()
    cnames = {0:"red",1:"blue"}
    for i in range(len(coors)):
        x,y = hz.to_pixels(coors[i][:,0], coors[i][:,1])
        ax.plot(x,y,color=cnames[i])
    plt.show()
