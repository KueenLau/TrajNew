import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

N = 4
ind = np.arange(start=1,stop=5)
width = 0.2
gap = 0.1

hmm = [31643.215283, 83796.19076, 183742.02834, 265169.51099]
reg = [35728.356323, 84329.098374, 203425.2572178, 283242.6523661]
_mine = [29860.0289612, 73697.2749371, 139449.076722, 226607.913023]
catc = [30129.44633, 77453.1799394, 174691.87125, 243907.23635781]


fig, ax = plt.subplots()

rects0 = ax.bar(ind+gap, hmm, width, color='g')
rects1 = ax.bar(ind+gap+width, reg, width, color='r')
rects2 = ax.bar(ind+gap+2*width, _mine, width, color='y')
rects3 = ax.bar(ind++gap+3*width, catc, width, color='b')


frame = plt.gca()
frame.spines['top'].set_visible(False)
frame.spines['right'].set_visible(False)

ax.set_xticks(ind+gap+2*width)
ax.set_xticklabels(('30%','50%','70%','100%'))
plt.ylabel('Average SSE')
plt.xlabel('Data percentage')
ax.legend((rects0[0], rects1[0], rects2[0], rects3[0]), ('HMM','Regression model', 'TCBML', 'CATC'),loc='upper left')

plt.show()