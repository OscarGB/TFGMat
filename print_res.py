import numpy as np
import matplotlib.pyplot as plt

from keras.models import model_from_json

# Globals
L = np.pi # Length
p = [0,0,0,0] # Two intervals
nIter = 100 # For numeric Fourier
nPoints = 100 # Points of the grid
nTime = 50 # Points of the grid in time
nFunctions = 1000 # Number of functions to try
timeInterval = 0.001 # Time between points in the time grid
alpha = 0.5 # Thermic constant

slices = np.arange(0, L, L/(nPoints-1)).tolist()
slices.append(L)

def f(x, new=False):
	global p
	if(new):
		p = [np.random.uniform(0, 1)*(L-0.01) for _ in range(4)]
		p.sort()
	else:
		if(x > p[0] and x < p[1]) or (x > p[2] and x < p[3]):
			return 1
		else:
			return 0

f(0, True)

X = [[[f(x) for x in slices]]]

 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

y = loaded_model.predict(X)

A = np.reshape(y, (100,50))

plt.imshow(A, cmap='hot', interpolation='nearest', aspect='auto')

plt.xticks([0,24,49], [0, 0.125, 0.25])
plt.yticks([0, 49, 99], [0,format(np.pi/2, '.4f'),format(np.pi, '.4f')])

plt.show()