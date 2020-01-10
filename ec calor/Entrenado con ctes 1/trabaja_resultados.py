from scipy.integrate import quad    
import numpy as np
from keras.models import model_from_json
import sys


# Globals
L = np.pi # Length
p = [0,0,0,0] # Two intervals
nIter = 100 # For numeric Fourier
nPoints = 100 # Points of the grid
nTime = 50 # Points of the grid in time
timeInterval = 0.001 # Time between points in the time grid
alpha = 0.5 # Thermic constant

slices = np.arange(0, L, L/(nPoints-1)).tolist()
slices.append(L)

outFile = "resultados/" + sys.argv[1] + ".res"

def integrand(x, n, L):
	return f(x) * np.sin((n*x*np.pi)/L)

def f(x, new=False):
	# x = x/np.pi
	# global p
	# if(new):
	# 	p = [np.random.uniform(0, 1)*(L-0.01) for _ in range(4)]
	# 	p.sort()
	# else:
	# 	if(x > p[0] and x < p[1]) or (x > p[2] and x < p[3]):
	# 		return 1
	# 	else:
	# 		return 0
	# return x*x
	if x < np.pi/2:
		return 0.3
	else:
		return 0.8

def getSolution(x, alpha, L):
	res = [0]*(nTime)
	res[0] = f(x)
	for n in range(nIter):
		I = quad(integrand, 0, L, args=(n,L))
		F = I[0]*np.sin(n*np.pi*x/L)
		for t in range(1, nTime):
			F = F*np.e**(-n**2*np.pi**2*alpha*t*timeInterval/L**2)
			res[t] += F*2/L
	return res

def main():

	f(0, True)
	mat = []
	tup = [[],[]]
	tup[0] = [f(x) for x in slices]
	for x in slices:
		tup[1] += (getSolution(x,alpha,L))
	tup[1] = np.reshape(tup[1], (100,50))
	mat.append(tup)
	
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

	mat.append(A)

	file = open(outFile, "wb")
	np.save(file, mat)
	file.close()

if __name__ == '__main__':
	main()
