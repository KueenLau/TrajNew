import matplotlib.pyplot as plt
import numpy as np

result1 = np.array([0,3.614381075,22.5730319,58.253288,115.4299521,223.3563])
result2 = np.array([0,8.235523,54.32425,117.76223,200.532144364,478.235462])
result3 = result2*2
result4 = result1*3
result5 = result3+result1
delta = np.array([0,5,10,15,20,25])
fig, ax = plt.subplots()

# plt.grid(linestyle = "--")

# plt.xlim(0,30)
# plt.ylim(0.1,0.23)

frame = plt.gca()
frame.spines['top'].set_visible(False)
frame.spines['right'].set_visible(False)
ax.set_ylabel('Runtime(seconds)')
ax.set_xlabel('Data percentage')
ax.set_xticklabels(('','10%','30%','50%','70%','100%'))
r1 = ax.plot(delta, result1, color='black',marker='*')
r2 = ax.plot(delta, result2, color='black',marker='o')
r3 = ax.plot(delta, result3, color='black',marker='s')
r4 = ax.plot(delta, result4, color='black',marker='+')
r5 = ax.plot(delta, result5, color='black',marker='v')


ax.legend((r1[0],r2[0],r3[0],r4[0],r5[0]), ('K=5', 'K=10', 'K=15', 'K=20', 'K=25'),loc='upper left')
plt.show()