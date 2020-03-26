# import numpy as np
# import matplotlib.pyplot as plt
# import sys

# file = "resultados/" + sys.argv[1] + ".res"

# data = np.load(file, allow_pickle=True)

# fig, axes = plt.subplots(nrows=2, ncols=1)

# plt.setp(axes, xticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90], xticklabels=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
#         yticks=[0, 50, 100], yticklabels=[0, 1.5708, 3.1416])
# fig.autofmt_xdate(rotation=45)

# im = axes.flat[0].contourf(data[0][1], np.arange(np.min(np.minimum(data[0][1], data[1])), np.max(np.maximum(data[0][1], data[1])), 0.2))
# im = axes.flat[1].contourf(data[1], np.arange(np.min(np.minimum(data[0][1], data[1])), np.max(np.maximum(data[0][1], data[1])), 0.2))
# axes.flat[0].set_title("Solución teórica")
# axes.flat[1].set_title("Solución generada")

# fig.subplots_adjust(right=0.9)
# fig.colorbar(im, ax=axes.flat)

# axes.flat[0].grid(True)
# axes.flat[1].grid(True)

# plt.show()

# fig, axes = plt.subplots(nrows=1, ncols=1)

# plt.setp(axes, xticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90], xticklabels=[0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
#         yticks=[0, 50, 100], yticklabels=[0, 1.5708, 3.1416])
# fig.autofmt_xdate(rotation=45)

# error = np.abs(data[0][1] - data[1])

# im = axes.contourf(error, np.arange(0, np.max(error)+0.1, 0.01))

# fig.subplots_adjust(right=0.9)
# fig.colorbar(im, ax=axes)

# plt.show()



import numpy as np
import matplotlib.pyplot as plt
import sys

# Globals
L = np.pi # Length
p = [0,0,0,0] # Two intervals
nIter = 100 # For numeric Fourier
nPoints = 100 # Points of the grid
nTime = 100 # Points of the grid in time
nFunctions = 1000 # Number of functions to try
timeInterval = 0.001 # Time between points in the time grid
alpha = 0.5 # Thermic constant

tim = [t * timeInterval for t in range(nTime)]
slices = np.arange(0, L, L/(nPoints-1)).tolist()
slices.append(L)

file = "resultados/" + sys.argv[1] + ".res"

data = np.load(file, allow_pickle=True)

fig, axes = plt.subplots(nrows=2, ncols=1)

a = [0,10,20,30,40,50,60,70,80,90]

plt.setp(axes, xticks=a, xticklabels=[format(tim[b], '.3f') for b in a],
        yticks=[0, 50, 100], yticklabels=[0, 1.5708, 3.1416], xlabel='t', ylabel='x')
fig.autofmt_xdate(rotation=45)

im = axes.flat[0].contourf(data[0][1], np.arange(np.min(np.minimum(data[0][1], data[1])), np.max(np.maximum(data[0][1], data[1]))+0.1, 0.1))
im = axes.flat[1].contourf(data[1], np.arange(np.min(np.minimum(data[0][1], data[1])), np.max(np.maximum(data[0][1], data[1]))+0.1, 0.1))
axes.flat[0].set_title("Exact solution")
axes.flat[1].set_title("Neural Network solution")

axes.flat[0].grid(True)
axes.flat[1].grid(True)

fig.colorbar(im, ax=axes.flat)

plt.show()

fig, axes = plt.subplots(nrows=1, ncols=1)

plt.setp(axes, xticks=a, xticklabels=[format(tim[b], '.3f') for b in a],
        yticks=[0, 50, 100], yticklabels=[0, 1.5708, 3.1416], xlabel='t', ylabel='x')
fig.autofmt_xdate(rotation=45)

error = np.log(np.abs(data[0][1] - data[1]))

im = axes.contourf(error, np.arange(np.min(error)+0.1, np.max(error)+0.1, 0.1))
axes.set_title("Log of absolute error")

axes.grid(True)

fig.subplots_adjust(right=0.9)
fig.colorbar(im, ax=axes)

plt.show()