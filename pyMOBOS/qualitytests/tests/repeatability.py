import numpy as np
from pyMOBOS.acquisition import Penalty
from pyMOBOS.qualitytests.quantification.hypervolume import hypervolume
from pyMOBOS.surrogates import Gaussian

class repeatability:


    def __init__(self, X, sampling_style, stopping_conditions, number_of_runs, initial_number_of_samples: int, batch_size: int = 1):
       

       #stoppingcondition_dict = {
       #     'numberofsamples': stoppingconditions.numberofsamples    
       #     }
       # Stopping conditions would be a function that has a specified inputs (maybe just current samples)
       # If stopping conditions are met, returns true
       # Example, If number of samples equals initial_samples + 70, stop
        self.X = X
        self.inital_number_of_samples = initial_number_of_samples
        self.sampling_style = sampling_style
        self.batch_size = batch_size
        self.stopping_conditions = stopping_conditions
        self.number_of_runs = number_of_runs

        pass

    def __call__(self, X, Y, stopping_conditions, test_functions, methods, intitial_number_of_samples):
        # methods = [method1, method2, method3]
        # currently hard-coded for penalty w/gaussian surrogate
        methods = [Penalty(X, Y, Gaussian(X, Y))]
        # test_functions = [TF1, TF2]
        # currently hard-coded for hypervolume
        test_functions = [hypervolume]

        # Loop through the methods
        for method in methods:
            # Loop through the test_functions
            for test_function in test_functions:
                # Call the multiiteration
                [xmulti_iter, ymulti_iter] = self.multi_iteration(X, stopping_conditions, method, test_function, initial_number_of_samples)
                
        
        return xmulti_iter, ymulti_iter

    def multi_iteration(self, X, stopping_conditions, method, test_function, initial_number_of_samples):
        # Call single_iteration multiple times
        
        # Use a list
        # results = [0] * 10
        # Output of results will be [[xres1, yres1], [xres2, yres2], [xres3, yres3]...]


        for iteration in range(self.number_of_runs):
            [xsingle_iter, ysingle_iter] = self.single_iteration(X, stopping_conditions, method, test_function, initial_number_of_samples)
            #X_data = np.append(X, xsingle_iter, axis=0)
            #Y_data = np.append(Y, ysingle_iter, axis=0)
            # results[i] = [xsingle_iter, ysingle_iter]
            iteration = iteration + 1
            

        # Concatinate the data
            # w/in the for loop take the outputs of the single iterations and tack them onto the end of an array
            # <- there may be a better way to do this 
        # Return information = the final arrays w/all the data
        return xsingle_iter, ysingle_iter

    def single_iteration(self, X, stopping_conditions, method, test_function, initial_number_of_samples):
        # PLAN:
        # Call initialization
        # For Loop
        #    Call single_sample
        #    Check stopping conditions
        # Return information
        
        # Call initialization
        [X, Y] = self.initialization(X, method)

        # for testing
        max_num_samples = initial_number_of_samples*5000
        # loop <- is there a way to do it this way or do we have to call a stopping condition function within
        while (stopping_conditions(X, Y)):
            [newX, newY] = self.single_sample(X,Y,method,test_function)
            X = np.append(X, newX, axis=0)
            Y = np.append(Y, newY, axis=0)
            # check stopping conditions
            numberofsamples = np.ndarray.size(Y)
            
            #if numberofsamples == max_num_samples:
            #    stopping_conditions == True
            #else:
            #    stopping_conditions == False
        return X, Y

    def single_sample(self, X, Y, method, test_function):
        # Take current X, Y data, run method to get next sample
        # new_x = method(X,Y)
        #surrogate = Gaussian(X, Y)
        new_x = method(X,Y)
       
        # Calculate next solution
        # new_y = test_function(new_x)
        new_y = test_function(new_x)

        # Combine new sample and solution with X, Y data
        # Return X, Y
        X_data = np.append(X, new_x, axis=0)
        Y_data = np.append(Y, new_y, axis=0)
        return X_data, Y_data

    def initialization(self, X, method):
        # Base this on TestFunction.py (Random and LHC)
        # Get initial set of X, Y data
        # Get initial samples (X)
        # Calculate cooresponding solutions (Y)   Y = method(X)

        X = self.X
        Y = method(X)
        return X, Y
