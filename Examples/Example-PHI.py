import os, sys
# Add parent directory to path for importing pyMOBOS
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pyMOBOS.utilities import phi
from pyMOBOS.surrogates import Gaussian

import matplotlib.pyplot as plt
import numpy as np

current_directory = os.path.dirname(os.path.abspath(__file__))
style = os.path.join(current_directory, 'publication.mplstyle')
plt.style.use(style)

def f(x):
    y = x ** 4 - x ** 3 - 2*x**2 - 3*x
    return y

x_real = np.linspace(-2,3,1000)
y_real = f(x_real)

#X = np.arange(-2,3).reshape(-1,1)
n_initial_samples = 6
X = np.random.uniform(-2, 3, size=n_initial_samples).reshape(-1,1)
Y = f(X)
X_Bounds = np.array([[-2],[3]])

print(X)

surrogate = Gaussian(X, Y)

xj = np.array([[-2],[-1],[0],[1],[2]])
h = phi(surrogate, X, Y, X_Bounds, xj)


plt.plot(x_real, y_real, '-')
plt.plot(X, Y, '.')
plt.show()

for i in range(5):
    h[i].plot(-2,3)


