import numpy as np
import matplotlib.pyplot as plt
outFile = "heat_eqs.eqs"
nTime = 50


file = open(outFile, "rb")
mat = np.load(file, allow_pickle=True)
file.close()

A = np.reshape(mat[0][1], (100,50))

tim = [(-1+np.exp(n/nTime))/20 for n in range(0,nTime)]

#plt.imshow(A, cmap='RdGy',extent=[0, 0.05, 0, np.pi], aspect='auto')

a = [10*i for i in range(5)]
plt.contourf(A, 20)
plt.xticks(a, [format(tim[b], '.3f') for b in a])
#plt.yticks([0, 49, 99], [format(np.pi, '.4f'),format(np.pi/2, '.4f'),0])

plt.show() 
