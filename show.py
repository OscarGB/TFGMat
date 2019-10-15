import numpy as np
import matplotlib.pyplot as plt

outFile = "heat_eqs"

file = open(outFile, "rb")
mat = np.load(file)
file.close()

A = np.reshape(mat[0][1], (256,50))

plt.imshow(A, cmap='hot', interpolation='nearest', aspect='auto')

plt.xticks([0,24,49], [0, 0.125, 0.25])

plt.show()