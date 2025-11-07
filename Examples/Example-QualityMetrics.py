import os, sys
# Add parent directory to path for importing pyMOBOS
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pyMOBOS.acquisition import QualityMetric
from pyMOBOS.acquisition.QualityMetrics import MOS, IHD
from pyMOBOS.surrogates import Gaussian
from pyMOBOS.testfunctions.numerical import ZDT3

from sklearn.gaussian_process.kernels import Matern

import matplotlib.pyplot as plt
import numpy as np

number_of_initial_samples = 10
number_of_parameters = 3
quality_metrics = [MOS, IHD]
number_of_iterations = 10

current_directory = os.path.dirname(os.path.abspath(__file__))
style = os.path.join(current_directory, 'publication.mplstyle')
plt.style.use(style)


f = ZDT3(number_of_parameters)

[X, Y] = f.initial_samples(number_of_initial_samples)
X_Bounds = f.bounds()

real_solution = f.plot_data()

x_real = real_solution[:,0]
y_real = real_solution[:,1]



for i in range(number_of_iterations):
    surrogate = Gaussian(X, Y)

    ac = QualityMetric(X, Y, X_Bounds, surrogate, quality_metrics)
    new_x = ac()
 
    #ac = Penalty(X, Y, X_Bounds, surrogate)
    #new_x = ac(2)
    new_y = f(new_x)

    print('Expected Improvment Method, Euclidean: \n', new_x)

    
    plt.plot(x_real, y_real, '.', markersize=1)
    plt.plot(Y[:,0],Y[:,1], '*')
    plt.plot(new_y[:,0],new_y[:,1], '*')
    plt.xlabel('f1')
    plt.ylabel('f2')

    
    
    plt.legend(['True Solution', 'Original Samples', 'Recommended Point'])

    
    
    plt.show()


    X = np.append(X, new_x, axis=0)
    Y = np.append(Y, new_y, axis=0)
    
    del ac
    del surrogate