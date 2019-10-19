import numpy as np
import matplotlib.pyplot as plt

outFile = "heat_eqs.eqs"

file = open(outFile, "rb")
mat = np.load(file, allow_pickle=True)
file.close()

A = np.reshape(mat[0][1], (100,50))

plt.imshow(A, cmap='hot', interpolation='nearest', aspect='auto')

plt.xticks([0,24,49], [0, 0.125, 0.25])
plt.yticks([0, 49, 99], [0,format(np.pi/2, '.4f'),format(np.pi, '.4f')])

plt.show()