import numpy as np
import matplotlib.pyplot as plt
import sys

f = open("wavef.dat")

n = []
t = []
aux = []
fig = []

for line in f.readlines():
	line = line.split()
	if len(line) == 1:
		if len(aux) != 0:
			n.append(aux)
			aux = []
			t.append(float(line[0]))
	else:
		aux.append([float(a) for a in line])

n.append(aux)
aux = []

print(len(t))

for i in range(len(t)):
	fig.append(plt.plot([a[0] for a in n[i]], [a[1] for a in n[i]], label='t = %.2f' % t[i]))

plt.plot([a[0] for a in n[i]], ([(a[0]**2.)/50. for a in n[i]]), ':')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()