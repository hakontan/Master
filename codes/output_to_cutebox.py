import numpy as np
import sys

inputfile = str(sys.argv[1])
outputfile = str(sys.argv[2])


data = np.loadtxt(inputfile, skiprows=int(sys.argv[3]))
x = data[:, 0]
y = data[:, 1]
z = data[:, 2]
outputdata = np.zeros(shape=(len(x), 3))
outputdata[:, 0] = x
outputdata[:, 1] = y
outputdata[:, 2] = z
np.savetxt(outputfile, outputdata)
