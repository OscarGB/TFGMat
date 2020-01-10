import numpy as np
import matplotlib.pyplot as plt
import sys

L = np.pi # Length
nTime = 50
nPoints = 100 # Points of the grid
tim = [(-1+np.exp(n/nTime))/20 for n in range(0,nTime)]
slices = np.arange(0, L, L/(nPoints-1)).tolist()
slices.append(L)

file = "resultados/" + sys.argv[1] + ".res"

data = np.load(file, allow_pickle=True)

fig, axes = plt.subplots(nrows=2, ncols=1)

a = [1,3,6,10,15,21,28,36,47]

plt.setp(axes, xticks=a, xticklabels=[format(tim[b], '.3f') for b in a],
        yticks=[0, 50, 100], yticklabels=[0, 1.5708, 3.1416])
fig.autofmt_xdate(rotation=45)

im = axes.flat[0].contourf(data[0][1], np.arange(np.min(np.minimum(data[0][1], data[1])), np.max(np.maximum(data[0][1], data[1])), 0.1))
im = axes.flat[1].contourf(data[1], np.arange(np.min(np.minimum(data[0][1], data[1])), np.max(np.maximum(data[0][1], data[1])), 0.1))
axes.flat[0].set_title("Exact solution")
axes.flat[1].set_title("Neural Network solution")

axes.flat[0].grid(True)
axes.flat[1].grid(True)

fig.colorbar(im, ax=axes.flat)

plt.show()

fig, axes = plt.subplots(nrows=1, ncols=1)

plt.setp(axes, xticks=a, xticklabels=[format(tim[b], '.3f') for b in a],
        yticks=[0, 50, 100], yticklabels=[0, 1.5708, 3.1416])
fig.autofmt_xdate(rotation=45)

error = np.log(np.abs(data[0][1] - data[1]))

im = axes.contourf(error, np.arange(np.min(error)+0.1, np.max(error)+0.1, 0.1))
axes.set_title("Log of absolute error")

axes.grid(True)

fig.subplots_adjust(right=0.9)
fig.colorbar(im, ax=axes)

plt.show()


#Check things after
error  = np.abs(data[0][1] - data[1])
print(np.max(error))
print(np.where(error == np.max(error)))
print(np.mean(error))
