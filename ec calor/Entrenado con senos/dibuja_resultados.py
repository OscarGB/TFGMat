import numpy as np
import matplotlib.pyplot as plt
import sys

file = "resultados/" + sys.argv[1] + ".res"

data = np.load(file, allow_pickle=True)

fig, axes = plt.subplots(nrows=2, ncols=1)

plt.setp(axes, xticks=[0, 10, 20, 30, 40], xticklabels=[0, 0.01, 0.02, 0.03, 0.04],
        yticks=[0, 50, 100], yticklabels=[0, 1.5708, 3.1416])

im = axes.flat[0].contourf(data[0][1], np.arange(np.min(np.minimum(data[0][1], data[1])), np.max(np.maximum(data[0][1], data[1])), 0.2))
im = axes.flat[1].contourf(data[1], np.arange(np.min(np.minimum(data[0][1], data[1])), np.max(np.maximum(data[0][1], data[1])), 0.2))
axes.flat[0].set_title("Solución teórica")
axes.flat[1].set_title("Solución generada")

fig.subplots_adjust(right=0.9)
fig.colorbar(im, ax=axes.flat)

plt.show()

fig, axes = plt.subplots(nrows=1, ncols=1)

plt.setp(axes, xticks=[0, 10, 20, 30, 40], xticklabels=[0, 0.01, 0.02, 0.03, 0.04],
        yticks=[0, 50, 100], yticklabels=[0, 1.5708, 3.1416])

error = np.abs(data[0][1] - data[1])

im = axes.contourf(error, np.arange(0, np.max(error)+0.1, 0.01))

fig.subplots_adjust(right=0.9)
fig.colorbar(im, ax=axes)

plt.show()