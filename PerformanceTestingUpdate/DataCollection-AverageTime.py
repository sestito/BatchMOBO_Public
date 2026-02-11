import sys
import os
import time
import csv
import pandas as pd
import numpy as np

# Add parent directory to path for importing pyMOBOS
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pyMOBOS.acquisition import qEHVI, PenaltyQualityMetric, QualityMetric, Ensemble, Sequential, Penalty
from pyMOBOS.acquisition.QualityMetrics import IHD, MOS
from pyMOBOS.surrogates import Gaussian

from pyMOBOS.testfunctions.engineering import WeldedBeam
from pyMOBOS.testfunctions.numerical import ZDT1, ZDT2, ZDT3, FON, DTLZ2

from pyMOBOS.qualitytests.stoppingconditions import NumberOfSamples

from pyMOBOS.qualitytests.helper import CheckStoppingConditions

from pyMOBOS.qualitytests import Test_BoTorch, Test

## Parameters
initial_number_of_samples = 30
max_number_of_samples = 300

number_of_runs_per_test = 30

# CSV file for timing results
csv_filename = "timing_results.csv"
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_filename)

# Test function
test_function = ZDT1(number_of_parameters=3)

# Test methods with batch sizes 1-5
test_methods = [
    # QualityMetric (batch_size=1 is QualityMetric, 2-5 is PenaltyQualityMetric)
    Test(Gaussian, QualityMetric, "QualityMetricIHDMOS", quality_metrics=[IHD, MOS]),
    Test(Gaussian, PenaltyQualityMetric, "PenaltyQualityMetricIHDMOS", quality_metrics=[IHD, MOS], batch_size=2),
    Test(Gaussian, PenaltyQualityMetric, "PenaltyQualityMetricIHDMOS", quality_metrics=[IHD, MOS], batch_size=3),
    Test(Gaussian, PenaltyQualityMetric, "PenaltyQualityMetricIHDMOS", quality_metrics=[IHD, MOS], batch_size=4),
    Test(Gaussian, PenaltyQualityMetric, "PenaltyQualityMetricIHDMOS", quality_metrics=[IHD, MOS], batch_size=5),
    
    # Ensemble
    Test(Gaussian, Ensemble, "Ensemble"),
    Test(Gaussian, Ensemble, "Ensemble", batch_size=2),
    Test(Gaussian, Ensemble, "Ensemble", batch_size=3),
    Test(Gaussian, Ensemble, "Ensemble", batch_size=4),
    Test(Gaussian, Ensemble, "Ensemble", batch_size=5),
    
    # EIMe (batch_size=1 is Sequential, 2-5 is Penalty)
    Test(Gaussian, Sequential, "EIMe"),
    Test(Gaussian, Penalty, "PenaltyEIMe", batch_size=2),
    Test(Gaussian, Penalty, "PenaltyEIMe", batch_size=3),
    Test(Gaussian, Penalty, "PenaltyEIMe", batch_size=4),
    Test(Gaussian, Penalty, "PenaltyEIMe", batch_size=5),
    
    # qEHVI
    Test_BoTorch(qEHVI, name="qEHVI", batch_size=1),
    Test_BoTorch(qEHVI, name="qEHVI", batch_size=2),
    Test_BoTorch(qEHVI, name="qEHVI", batch_size=3),
    Test_BoTorch(qEHVI, name="qEHVI", batch_size=4),
    Test_BoTorch(qEHVI, name="qEHVI", batch_size=5)
]

stopping_conditions = [NumberOfSamples(max_number_of_samples, initial_number_of_samples)]


def initialize_csv():
    """Create CSV file with headers if it doesn't exist."""
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['run', 'method_name', 'batch_size', 'current_samples', 'time_seconds'])


def get_completed_runs():
    """Read CSV and determine which runs are complete for each method/batch_size combination."""
    if not os.path.exists(csv_path):
        return {}
    
    df = pd.read_csv(csv_path)
    
    completed = {}
    for _, group in df.groupby(['method_name', 'batch_size']):
        method_name = group['method_name'].iloc[0]
        batch_size = group['batch_size'].iloc[0]
        runs_completed = group['run'].unique()
        
        # Check if each run is complete (should have samples from 30 to max_number_of_samples)
        complete_runs = []
        for run in runs_completed:
            run_data = group[group['run'] == run]
            max_samples_in_run = run_data['current_samples'].max()
            # A run is complete if it reached max_number_of_samples
            if max_samples_in_run >= max_number_of_samples:
                complete_runs.append(run)
        
        key = (method_name, batch_size)
        completed[key] = set(complete_runs)
    
    return completed


def append_timing_to_csv(run, method_name, batch_size, current_samples, time_seconds):
    """Append a single timing measurement to the CSV file."""
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([run, method_name, batch_size, current_samples, time_seconds])


def get_method_display_name(test):
    """Get a clean display name for the method."""
    # Simplify the names for display
    if "QualityMetric" in test.name:
        return "QualityMetric"
    elif test.name == "Ensemble":
        return "Ensemble"
    elif "EIMe" in test.name:
        return "EIMe"
    elif test.name == "qEHVI":
        return "qEHVI"
    return test.name


## Main algorithm
def main():
    print("Initializing timing measurements...")
    initialize_csv()
    
    print("Checking for completed runs...")
    completed_runs = get_completed_runs()
    
    total_tests = len(test_methods) * number_of_runs_per_test
    current_test = 0
    
    for test in test_methods:
        method_name = get_method_display_name(test)
        batch_size = test.batch_size
        key = (method_name, batch_size)
        
        completed_for_this_method = completed_runs.get(key, set())
        
        print(f"\n{'='*60}")
        print(f"Method: {method_name}, Batch Size: {batch_size}")
        print(f"Completed runs: {sorted(completed_for_this_method)}")
        print(f"{'='*60}")
        
        for run in range(1, number_of_runs_per_test + 1):
            current_test += 1
            
            if run in completed_for_this_method:
                print(f"  Run {run}/{number_of_runs_per_test}: SKIPPING (already completed)")
                continue
            
            print(f"  Run {run}/{number_of_runs_per_test}: Starting...")
            
            # Initialize samples for this run
            X_Bounds = test_function.bounds()
            [X, Y] = test_function.initial_samples(
                number_of_samples=initial_number_of_samples, 
                random_style=test_function.Random.LHS
            )
            
            iteration = 0
            while not CheckStoppingConditions(X, Y, stopping_conditions):
                current_samples = len(X)
                
                # Time the sample generation
                start_time = time.time()
                new_samples = test(X, Y, X_Bounds)
                end_time = time.time()
                
                elapsed_time = end_time - start_time
                
                # Save timing to CSV immediately
                append_timing_to_csv(run, method_name, batch_size, current_samples, elapsed_time)
                
                # Evaluate new samples and update
                new_solutions = test_function(new_samples)
                X = np.append(X, new_samples, axis=0)
                Y = np.append(Y, new_solutions, axis=0)
                
                iteration += 1
                if iteration % 10 == 0:
                    print(f"    Iteration {iteration}: {current_samples} samples, last timing: {elapsed_time:.4f}s")
            
            print(f"  Run {run}/{number_of_runs_per_test}: COMPLETED ({len(X)} total samples)")
            print(f"  Progress: {current_test}/{total_tests} total tests")
    
    print(f"\n{'='*60}")
    print("All timing measurements complete!")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()