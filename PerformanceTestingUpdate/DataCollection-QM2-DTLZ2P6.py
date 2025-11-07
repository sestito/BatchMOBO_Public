import sys
import os

# Add parent directory to path for importing pyMOBOS
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pyMOBOS.acquisition import Ensemble, Sequential, Penalty, PenaltyQualityMetric, QualityMetric
from pyMOBOS.acquisition.QualityMetrics import IHD, MOS
from pyMOBOS.surrogates import Gaussian

from pyMOBOS.testfunctions.numerical import ZDT1, ZDT2, ZDT3, FON, DTLZ2

from pyMOBOS.qualitytests.stoppingconditions import NumberOfSamples

from pyMOBOS.qualitytests.helper import CheckStoppingConditions

from pyMOBOS.qualitytests import Test

import numpy as np

#TODO: Implement reloading when mistakes

## Parameters
# Parameters
initial_number_of_samples = 30
max_number_of_samples = 300

number_of_runs_per_test = 30

# Save Folder
test_name = "Trial"
save_folder = "DataFiles"

save_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), save_folder)


test_functions = [
    DTLZ2(number_of_parameters = 6, number_of_objectives = 5)
    ]

test_methods = [
    Test(Gaussian, PenaltyQualityMetric, "PenaltyQualityMetricIHDMOS", quality_metrics=[IHD, MOS], batch_size=2)
    ]


stopping_conditions = [NumberOfSamples(max_number_of_samples, initial_number_of_samples)]



## Main algorithm

# Initial Samples
for test in test_methods:
    for f in test_functions:
        run = 1
        while run <= number_of_runs_per_test:
            X_Bounds = f.bounds()
            [X, Y] = f.initial_samples(number_of_samples=initial_number_of_samples, random_style = f.Random.LHS)

            while not CheckStoppingConditions(X, Y, stopping_conditions):
                new_samples = test(X, Y, X_Bounds)
                new_solutions = f(new_samples)

                X = np.append(X, new_samples, axis=0)
                Y = np.append(Y, new_solutions, axis=0)

                # File Format
                # TestPrefix-MethodName-TestName-Parameters-Objectives-BatchSize-Samples/Solutions-Itteration

                to_join = [
                    test_name,
                    test.name,
                    "nB" + str(test.batch_size),
                    f.name,
                    "nP" + str(f.number_of_parameters),
                    "nO" + str(f.number_of_objectives)
                ]

                file_name_prefix = '-'.join(to_join)
                file_samples = os.path.join(save_folder,file_name_prefix + '-samples-' + str(run) + '.npy')
                file_solutions = os.path.join(save_folder, file_name_prefix + '-solutions-' + str(run) + '.npy')
                np.save(file_samples, X)
                np.save(file_solutions, Y)
            
            run += 1