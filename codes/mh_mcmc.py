import numpy as np
import matplotlib.pyplot as plt 
import random
mu = np.array([0, 0])
sigma = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
N = int(1e5)
def q(x, y):
    input_arr = np.array([x, y]) 
    Normal_dist =  1.0 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))
    Normal_dist *= np.exp(-0.5 * (input_arr - mu).T @ sigma @ (input_arr - mu))
    return Normal_dist
"""
def p(x):
    return 3 * np.exp(-0.5*(x-1)**2) + 5 * np.exp(-0.5*(x + 2)**2)
"""
def p(x, y):
    # Multivariate function
    V = np.array([[2.0, 1.2],
                  [1.2, 2.0]])
    input_arr = np.array([x, y]) 
    Normal_dist =  1.0 / (2 * np.pi * np.sqrt(np.linalg.det(V)))
    Normal_dist *= np.exp(-0.5 * (input_arr - mu).T @ V @ (input_arr - mu))
    return Normal_dist
"""
def p(x, y):
    # square function
    if 3 < x < 7 and 1 < y < 9:
        return 1
    else:
        return 0
"""
u = np.random.rand(N)
y = []
y.append(np.array([4, 4]))
#y.append(np.random.multivariate_normal(mu, sigma))
for i in range(N-1):
    y_new = np.random.multivariate_normal(mu, sigma)
    #print(y_new)
    y_prev = y[i]
    alpha = min(1, p(y_new[0], y_new[1]) * q(y_prev[0], y_prev[1]) / (p(y_prev[0], y_prev[1]) * q(y_new[0], y_new[1])))
    if u[i] < alpha:
        y.append(y_new)
    else:
        y.append(y[i])  
y = np.array(y)
#print(y[:,0])
plt.hist2d(y[:,0],y[:,1], bins = 50, density=True)
plt.show()