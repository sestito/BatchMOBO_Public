import torch
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import NondominatedPartitioning
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

import numpy as np

class qEHVI:
    def __init__(self, X: np.ndarray, Y: np.ndarray, X_Bounds: np.ndarray):
        self.X = X
        self.Y = Y
        self.X_Bounds = X_Bounds
    
    def __call__(self, batch_size: int = 1) -> np.ndarray:
        return self.propose_location(batch_size)
    

    def propose_location(self, batch_size):
        # Use dtype=torch.double to avoid numerical gradient issues
        train_X = torch.from_numpy(self.X).to(dtype=torch.double)
        train_Y = torch.from_numpy(self.Y).to(dtype=torch.double)
        bounds = torch.tensor(self.X_Bounds, dtype=torch.double)

        # Get dimensionality
        d = train_X.shape[-1]
        num_objectives = train_Y.shape[-1]

        # Use ModelListGP with separate GP per objective for better numerical stability
        # This is more robust than a single multi-output GP
        models = []
        for i in range(num_objectives):
            model_i = SingleTaskGP(
                train_X,
                train_Y[:, i:i+1],  # Single objective
                input_transform=Normalize(d=d, bounds=bounds),
                outcome_transform=Standardize(m=1)
            )
            models.append(model_i)
        
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Define reference point (worse than all objectives)
        # Scale offset proportionally to objective range
        y_range = train_Y.max(dim=0).values - train_Y.min(dim=0).values
        y_range = torch.clamp(y_range, min=1e-6)  # Avoid zero range
        ref_point = train_Y.min(dim=0).values - 0.1 * y_range

        # Compute current Pareto frontier for partitioning
        partitioning = NondominatedPartitioning(
            ref_point=ref_point,
            Y=train_Y
        )

        # Define q-EHVI acquisition function
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))
        acq_func = qLogExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),  # Convert to list for ModelListGP
            partitioning=partitioning,
            sampler=sampler
        )

        # Optimize with more restarts and samples for robustness
        candidates, acq_value = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=batch_size,
            num_restarts=20,       # Increased from 10
            raw_samples=1024,      # Increased from 512
            options={
                "batch_limit": 5,
                "maxiter": 200,
            }
        )

        return candidates.detach().numpy()