import os
import sys

# MUST INSTALL openpyxl

# Add parent directory to path for importing pyMOBOS
#pyMOBOS_dir = ".."
pyMOBOS_dir = "C:\\Users\\aholder\\Documents\\GitHub\\BatchMOBO-Python"
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), pyMOBOS_dir))

import numpy as np
import pandas as pd

from pyMOBOS.acquisition.QualityMetrics import MOS, IHD
from pyMOBOS.utilities import ParetoEfficient


# CONSTANTS
data_directory = "BMOBO Data"


data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), data_directory)
files = [x for x in os.listdir(data_directory) if x.endswith('.npy')]

col_names = ["Test", "Iteration", "Method", "Number of Batch Samples", \
             "Test Name", "Number of Parameters", "Number of Objectives", \
            "Pareto Points", "DOS", "IHD"]

output_data = []

for file in files:
    file_info = file.split('-')
    # File info layout
    # [TestPrefix, Method, Number of Batch Samples (nB#), ...
    #  Test Name (FON, etc.), Number of Parameters (nP#), ...
    #  Number of Objectives (nO#, if # is None, default to 2), ...
    #  Samples/Solutions, Iteration]

    # We only need to parse solution data
    if file_info[6] == "samples":
        continue

    # Initialize output for row in dataframe    
    write_data = [0] * len(col_names)
    
    # Parse information data
    write_data[0] = file_info[0]
    write_data[1] = int(file_info[7].split('.')[0]) # This is #.npy so parse off the .npy
    
    # methods under same name for b=1, b=2, and b=3
    if file_info[1] == 'PenaltyEIMe':
        write_data[2] = 'EIMe'
    elif file_info[1] == 'PenaltyQualityMetricIHDMOS':
        write_data[2] = 'QualityMetricIHDMOS'
    else:
        write_data[2] = file_info[1]

    write_data[3] = int(file_info[2][2:])
    write_data[4] = file_info[3]
    write_data[5] = int(file_info[4][2:])
    
    num_objectives_text = file_info[5][2:]
    if num_objectives_text == "None":
        num_objectives = 2
    else:
        num_objectives = int(num_objectives_text)
    write_data[6] = num_objectives
    
    solutions = np.load(os.path.join(data_directory,file))
    #init_samples = solutions[0:30]

    # get batch size
    batch_text = file_info[2][2:]
    batch_size = int(batch_text)

    # only take first 100 runs for b=1, first 200 runs for b=2, etc.
    chomp_length = batch_size*100 + 30
    final_arr = solutions[0:chomp_length]

    #print(chomp_length)
    
    pareto = ParetoEfficient(final_arr)

    # Number of Pareto Front Objectives
    n_pareto = pareto.shape[0]
    write_data[7] = n_pareto

    # Quality of the Pareto Front
    write_data[8] = MOS(pareto)
    write_data[9] = IHD(pareto)
    
    # Add data to dataframe
    output_data.append(write_data)

# Create dataframe and export to Excel
df = pd.DataFrame(data=output_data, columns=col_names)

export_file = "outputChomp.xlsx"
export_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), export_file)
df.to_excel(export_file, index=False)
