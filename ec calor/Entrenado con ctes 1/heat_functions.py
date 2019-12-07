from scipy.integrate import quad    
import numpy as np

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

outFile = "heat_eqs.eqs"

def integrand(x, n, L):
	return f(x) * np.sin((n*x*np.pi)/L)

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
	mat = []
	for _ in range(nFunctions):
		tup = [[],[]]
		f(0, True)
		tup[0] = [f(x) for x in slices]
		for x in slices:
			tup[1] += (getSolution(x,alpha,L))
		mat.append(tup)
		print(len(mat))
	file = open(outFile, "wb")
	np.save(file, mat)
	file.close()

if __name__ == '__main__':
	main()
