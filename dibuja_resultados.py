import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
import sys

file = "resultados/" + sys.argv[1] + ".res"

data = np.load(file, allow_pickle=True)

fig, axes = plt.subplots(nrows=2, ncols=1)
im = axes.flat[0].contourf(data[0][1])
im = axes.flat[1].contourf(data[1])
fig.subplots_adjust(right=0.8)
fig.colorbar(im, ax=axes.flat)
plt.show()