import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('data.csv', delimiter=',')

data = data [1:,1:]

plt.scatter(data[:,0], data[:,1], c=data[:,-1], cmap='rainbow')
plt.show()