from ..TestFunction import TestFunction

import numpy as np
import warnings

class WeldedBeam(TestFunction):
    """
    Welded Beam Design Multi-Objective Optimization Problem
    
    A classic engineering benchmark problem for multi-objective optimization.
    The problem involves designing a welded steel beam to minimize fabrication 
    cost and end deflection subject to constraints on shear stress, bending 
    stress, buckling load, and geometric requirements.
    
    References:
    -----------
    [1] Deb, K., Sundar, J., Rao, U.B.N., and Chaudhuri, S. (2006). 
        "Reference Point Based Multi-Objective Optimization Using Evolutionary 
        Algorithms." International Journal of Computational Intelligence Research, 
        Vol. 2, No. 3, pp. 273-286.
    
    [2] Ray, T. and Liew, K.M. (2002). "A Swarm Metaphor for Multiobjective 
        Design Optimization." Engineering Optimization, Vol. 34, pp. 141-153.
    
    [3] Reklaitis, G.V., Ravindran, A., and Ragsdell, K.M. (1983). 
        Engineering Optimization Methods and Applications. New York: Wiley.
    
    Parameters:
    -----------
    x1 (h): Weld thickness [0.125, 5.0] inches
    x2 (l): Weld length [0.1, 10.0] inches  
    x3 (t): Beam height [0.1, 10.0] inches
    x4 (b): Beam width [0.125, 5.0] inches
    
    Objectives:
    -----------
    f1: Fabrication cost (minimize)
    f2: End deflection of the beam (minimize)
    
    Constraints (handled via penalty or constraint handling):
    ---------------------------------------------------------
    g1: Shear stress constraint (tau <= 13,600 psi)
    g2: Bending stress constraint (sigma <= 30,000 psi)
    g3: Geometric constraint (h <= b)
    g4: Buckling load constraint (Pc >= 6,000 lbs)
    
    :param x np.ndarray: 2D array of shape [n_samples, 4] with parameter values
    :return: 2D array of shape [n_samples, 2] containing [cost, deflection]
    """
    
    name = "WeldedBeam"
    number_of_parameters = 4
    number_of_objectives = 2
    
    # Problem constants
    P = 6000.0      # Applied load (lbs)
    L = 14.0        # Beam length (inches)
    E = 30e6        # Young's modulus (psi)
    G = 12e6        # Shear modulus (psi)
    tau_max = 13600.0   # Maximum allowable shear stress (psi)
    sigma_max = 30000.0 # Maximum allowable bending stress (psi)
    delta_max = 0.25    # Maximum allowable deflection (inches)
    
    def __init__(self, number_of_parameters: int = 4):
        if number_of_parameters != 4:
            warnings.warn("WeldedBeam uses exactly 4 parameters. Setting to 4.")
        self.set_bounds()
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the welded beam objectives.
        
        :param x: 2D array [n_samples, 4] with columns [h, l, t, b]
        :return: 2D array [n_samples, 2] with columns [cost, deflection]
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        if x.shape[1] != 4:
            raise ValueError('WeldedBeam requires exactly 4 parameters: [h, l, t, b]')
        
        return self.__compute_objectives(x)
    
    def set_number_of_parameters(self, number_of_parameters: int) -> None:
        warnings.warn("WeldedBeam uses exactly 4 parameters. Cannot change.")
    
    def set_bounds(self) -> None:
        """
        Set parameter bounds.
        
        x1 (h): [0.125, 5.0]   - Weld thickness
        x2 (l): [0.1, 10.0]   - Weld length
        x3 (t): [0.1, 10.0]   - Beam height
        x4 (b): [0.125, 5.0]   - Beam width
        """
        bounds = np.zeros((2, self.number_of_parameters))
        bounds[0, :] = [0.125, 0.1, 0.1, 0.125]  # Lower bounds
        bounds[1, :] = [5.0, 10.0, 10.0, 5.0]    # Upper bounds
        self.parameter_bounds = bounds
    
    def __compute_objectives(self, x: np.ndarray) -> np.ndarray:
        """Compute both objective functions."""
        n_samples = x.shape[0]
        output = np.zeros((n_samples, 2))
        
        output[:, 0] = self.__cost(x)
        output[:, 1] = self.__deflection(x)
        
        output = output + self.penalty(x).reshape(-1, 1)

        return output
    
    def __cost(self, x: np.ndarray) -> np.ndarray:
        """
        Objective 1: Fabrication cost
        f1(x) = 1.10471 * h^2 * l + 0.04811 * t * b * (14.0 + l)
        """
        h, l, t, b = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        return 1.10471 * h**2 * l + 0.04811 * t * b * (14.0 + l)
    
    def __deflection(self, x: np.ndarray) -> np.ndarray:
        """
        Objective 2: End deflection
        
        delta(x) = 2.1952 / (t^3 * b)
        
        Derived from: delta = 4*P*L^3 / (E*t^3*b)
        With P=6000 lbs, L=14 in, E=30e6 psi:
        4 * 6000 * 14^3 / 30e6 = 2.1952
        """
        h, l, t, b = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        return 2.1952 / (t**3 * b)
    
    def penalty(self, x: np.ndarray) -> np.ndarray:
        """
        Compute penalty for constraint violations.
        
        :param x: 2D array [n_samples, 4] with parameter values
        :return: 1D array [n_samples] with total penalty per sample
        """
        g = self.evaluate_constraints(x)
        penalties = np.maximum(0, g).sum(axis=1)
        return penalties

    def evaluate_constraints(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate constraint violations.
        Returns array of shape [n_samples, 4] where values <= 0 indicate feasibility.
        
        g1: tau(x) - tau_max <= 0 (shear stress)
        g2: sigma(x) - sigma_max <= 0 (bending stress)
        g3: h - b <= 0 (geometric)
        g4: P - Pc(x) <= 0 (buckling)
        """
        if x.ndim == 1:
            x = x.reshape(1, -1)
            
        n_samples = x.shape[0]
        g = np.zeros((n_samples, 4))
        
        h, l, t, b = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        
        # Compute shear stress tau
        tau = self.__shear_stress(x)
        g[:, 0] = tau - self.tau_max
        
        # Compute bending stress sigma
        sigma = self.__bending_stress(x)
        g[:, 1] = sigma - self.sigma_max
        
        # Geometric constraint: h <= b
        g[:, 2] = h - b
        
        # Buckling constraint: P <= Pc
        Pc = self.__buckling_load(x)
        g[:, 3] = self.P - Pc
        
        return g
    
    def __shear_stress(self, x: np.ndarray) -> np.ndarray:
        """Calculate shear stress in the weld."""
        h, l, t, b = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        
        # Primary shear stress
        tau_prime = self.P / (np.sqrt(2) * h * l)
        
        # Secondary shear stress due to moment
        M = self.P * (self.L + l / 2.0)
        R = np.sqrt(l**2 / 4.0 + ((h + t) / 2.0)**2)
        J = 2.0 * np.sqrt(2) * h * l * (l**2 / 12.0 + ((h + t) / 2.0)**2)
        tau_double_prime = M * R / J
        
        # Combined shear stress
        tau = np.sqrt(tau_prime**2 + 2*tau_prime*tau_double_prime*(l/(2*R)) + tau_double_prime**2)
        
        return tau
    
    def __bending_stress(self, x: np.ndarray) -> np.ndarray:
        """Calculate bending stress in the beam."""
        h, l, t, b = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        return 6.0 * self.P * self.L / (b * t**2)
    
    def __buckling_load(self, x: np.ndarray) -> np.ndarray:
        """Calculate critical buckling load."""
        h, l, t, b = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        return (4.013 * self.E / (6.0 * self.L**2)) * t * b**3 * (1.0 - (t / (2.0 * self.L)) * np.sqrt(self.E / self.G))