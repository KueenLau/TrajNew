import matplotlib.pyplot as plt
import numpy as np

N = 5
ind = np.arange(N)
width = 0.35

y1 = [267002.531,266607.913,245715.723,257217.38,259677.652]
# y2 = []
# diff = 1000
# for _ in y1:
#     y2.append(_ + diff)
#     diff += diff

fig, ax = plt.subplots()

rects1 = ax.bar(ind, y1, width, color='r')
# rects2 = ax.bar(ind+width, y2, width, color='y')

frame = plt.gca()
frame.spines['top'].set_visible(False)
frame.spines['right'].set_visible(False)

ax.set_xticks(ind+width/2)
ax.set_xticklabels(('0.55','0.65','0.7','0.75','0.8'))
plt.ylabel('Average SSE')
plt.xlabel('Cutoff')
# ax.legend((rects1[0], rects2[0]), ('TCBML', 'Regression model'),loc='upper left')

plt.show()