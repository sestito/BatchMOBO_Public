import numpy as np
from scipy.stats import norm

from pyMOBOS.utilities import ParetoEfficient

import time

from sympy import true

from . import __acquisition_common as acquisition_common


# TODO implement Ensemble based on Lyu 2018 and DEMO algorithm


class Ensemble:


    time: bool = False # If set to true, will output the time it takes for the recommendation calculation

    def __init__(self, X: np.ndarray, Y: np.ndarray, X_Bounds: np.ndarray, surrogate: object, n_restarts: int = 25, epsilon: float = 0.1, beta: float = 1, mo_method: str = 'Mean') -> None:
        '''
        Initialization of the EIMe class


        Sphinx Markup
        ------------
        :param X np.ndarray: Design Variables. This is a 2d array in the form of [number of samples, number of parameters]
        :param Y np.ndarray: Objectives. This is a 2d array in the form of [number of samples, number of objectives]
        :param X_Bounds np.ndarray: These are the bounds of your design variabls in the form of [2, number of design variables]. X_Bounds[0,:] corresponds to your minimum and X_Bounds[1,:] corresponds to your maximum.
        :param surrogate object: Surrogate Object. This is the initialized surrogate to be used.
        :param n_restarts int: Number of restarts during the optimization process. More restarts provide a higher chance to find the true optimal at increased computation time. Default is 25.
        '''

        styles_lookup = {
            'Euclidean': 'Euclidean',
            'Mean': 'Mean'
        }
        try:
            self.mo_method = styles_lookup[mo_method]
        except KeyError:
            raise Exception('Multi-objective method ' + str(mo_method) + ' does not exist!')


        self.X = X
        self.Y = Y
        self.X_Bounds = X_Bounds
        self.surrogate = surrogate
        self.n_restarts = n_restarts
        self.epsilon = epsilon
        self.beta = beta

        #self.surrogate.init(self.X, self.Y)

    def __call__(self, batch: int = 1) -> np.ndarray:
        return self.propose_location(batch)


    def propose_location(self, batch: int = 1) -> np.ndarray:
        """
        This method proposes the next recommended set of design variables that will maximize the expected improvement.

        Sphinx Markup
        ------------
        :return np.ndarray: This is a 1d array in the form of [design variables]
        """

        pop_size = batch ** 2
        if pop_size < 100:
            pop_size = 100
        scaling_factor = 0.5 # F
        crossover_rate = 0.7
        number_of_iterations = 100

        if self.time:
            start_time = time.time()


        number_of_parameters = self.X.shape[1]

        # Use DEMO algorithm to find pareto front
        # Select points randomly from pareto front
        samples, solutions = self.__DEMO(pop_size, scaling_factor, crossover_rate, number_of_iterations)

        # Get pareto front from samples
        num_pareto_solutions = 0
 
        proposed_points = np.zeros((batch, self.X.shape[1]))

        current_batch = batch
        top = 0

        pareto_samples, _, mask = ParetoEfficient(samples, solutions, True)
        #pareto_samples, _, mask = acquisition_common.non_dominated(samples, solutions, True)
        num_pareto_solutions = pareto_samples.shape[0]
        while num_pareto_solutions < current_batch:
            bottom = top + num_pareto_solutions
            proposed_points[top:bottom, :] = pareto_samples
            
            # Set updated variables
            top = bottom
            current_batch -= num_pareto_solutions

            # Delete mask values
            samples = samples[~mask,:]
            solutions = solutions[~mask,:]

            pareto_samples, _, mask = ParetoEfficient(samples, solutions, True)
            num_pareto_solutions = pareto_samples.shape[0]

        '''
        for i in range(100):
            pareto_samples, _ = acquisition_common.non_dominated(samples, solutions)
            num_pareto_solutions = pareto_samples.shape[0]
            if num_pareto_solutions >= batch:
                success = True
                break
        
        
        if success == False:
            raise Exception('Never found enough pareto solutions')
        '''

        #select randomly from Pareto front
        new_proposed_points = pareto_samples[np.random.choice(np.arange(num_pareto_solutions),current_batch), :]
        proposed_points[top:, :] = new_proposed_points


        if self.time:
            elapsed_time = int(time.time() - start_time)
            print('Calculation took ' + str(elapsed_time) + ' seconds.' )


        return proposed_points




    
    def __DEMO(self, pop_size: int = 40, scaling_factor: float = 0.5, crossover_rate: float = 0.6, number_of_iterations: int = 100):
        number_of_candidates = 3
        number_of_parameters = self.X.shape[1]


        # Initialize population of candidate solution
        pop = np.random.uniform(self.X_Bounds[0, :], self.X_Bounds[1, :], size=(pop_size, number_of_parameters))
      


        # for each individual paretn in the population do...
        for k in range(number_of_iterations):

            # Evaluate Population
            obj = np.zeros((pop_size, 3))
            obj[:,0] = acquisition_common.ExpectedImprovementArray(self,pop)[:,0]
            obj[:,1] = acquisition_common.ProbabilityImprovementArray(self,pop)[:,0]
            obj[:,2] = acquisition_common.LowerConfidenceBoundArray(self,pop)[:,0]
            # This is in terms of (pop size, number of objectives)

            for i in range(pop_size):
                parent = pop[i,:]
                solution = obj[i,:]

                # Create candidate C from parent
                potential_choices = np.delete(np.arange(pop_size), i)
                choices = np.random.choice(potential_choices, number_of_candidates)
                
                # Create candidate as C = Pi1 + F (Pi2 - Pi3)
                candidate = pop[choices[0],:] + scaling_factor*(pop[choices[1],:] - pop[choices[2],:])

                # Check bounds of candidate
                candidate = np.clip(candidate, self.X_Bounds[0,:], self.X_Bounds[1,:])

                # Binary crossover with parent
                p = np.random.rand(number_of_parameters)
                trial = [candidate[j] if p[j] < crossover_rate else parent[j] for j in range(number_of_parameters)]
                candidate = np.array(trial)            

                # Evaluate the candidate
                candidate_solution = [0] * 3
                
                candidate_solution[0] = acquisition_common.ExpectedImprovementArray(self, np.array([candidate]))
                candidate_solution[1] = acquisition_common.ProbabilityImprovementArray(self, np.array([candidate]))
                candidate_solution[2] = acquisition_common.LowerConfidenceBoundArray(self, np.array([candidate]))

                # Check if dominate
                dominate = candidate_solution > solution # We want to maximize, so if all True then choose candidate
                num_dominated = np.sum(dominate)
                if num_dominated == 0:
                    # parent dominates candidate
                    pass

                elif num_dominated == 3:
                    # candidate dominates parent
                    pop[i,:] = candidate

                else:
                    # neither dominates
                    pop = np.append(pop, candidate.reshape(1,-1), axis=0)

            # Truncate to pop_size
            # Randomly enumerate the individuals in the population
            current_pop_number = pop.shape[0]
            if current_pop_number > pop_size:
                pop = pop[np.random.choice(np.arange(current_pop_number),pop_size),:]


        # Evaluate Population
        obj = np.zeros((pop_size, 3))
        obj[:,0] = acquisition_common.ExpectedImprovementArray(self, pop)[:,0]
        obj[:,1] = acquisition_common.ProbabilityImprovementArray(self, pop)[:,0]
        obj[:,2] = acquisition_common.LowerConfidenceBoundArray(self, pop)[:,0]
        # This is in terms of (pop size, number of objectives)    
        return pop, obj

