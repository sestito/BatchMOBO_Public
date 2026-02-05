import os
import sys

import numpy as np
import pandas as pd

# CONSTANTS
export_file = "outputChomp.xlsx"
export_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), export_file)

data = pd.read_excel(export_file)


def identify_row(row):
    titles = ['Method', 'Number of Batch Samples', 'Test Name', 'Number of Parameters',\
              'Number of Objectives']
    s = ""
    for title in titles:
        s += str(row[title]) + "-"
    return s[:-1] # Remove last character

def return_data(row):
    titles = ['Pareto Points', 'MOS', 'IHD']
    data = []
    for title in titles:
        data.append(row[title])
    return data

def analyze_data(data):
    pass

identifier = ""
all_identifiers = []
all_data = []
test_data = []

for index, row in data.iterrows():
    identifier_new = identify_row(row)

    if identifier != identifier_new:
        all_identifiers.append(identifier_new)
        identifier = identifier_new
        if test_data != []:
            all_data.append(test_data)
        test_data = []
    
    test_data.append(return_data(row))
       
all_data.append(test_data)


data = np.array(all_data)

# data is in the shape of (test, sample (30), property of interest (ex. Pareto Points))
# Test is stored in the all_identifiers

# For All problesm, let's calculate min, max, average, std
#fname = "AnalyzedData.csv"
#fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), fname)
#f = open(fname, "w")

titles = ['Pareto Points', 'MOS', 'IHD']

st = "Method,BatchSize,Test,nParameters,nObjectives,Property,Minimum,Maximum,Average,StandardDeviation"
all_analyzed_data = []

#f.write(st)
#f.write("\n")
for i in range(len(all_identifiers)):
    identifier = all_identifiers[i]
    x = data[i, :, :]
    for j in range(x.shape[1]):
        # NORMAL DATA
        analyzed_data = identifier.split('-')

        y = x[:, j]

        minimum = np.min(y)
        maximum = np.max(y)
        average = np.average(y)
        standard_deviation = np.std(y)

        analyzed_data += [titles[j], str(minimum), str(maximum), \
                         str(average), str(standard_deviation)]
        all_analyzed_data.append(analyzed_data)
        #s = ","
        #s = s.join(analyzed_data)
        #s += "\n"
        #f.write(s)


#f.close()

df = pd.DataFrame(data=all_analyzed_data, columns=st.split(','))

df["Test"] = pd.Categorical(df["Test"], ["ZDT1", "ZDT2", "ZDT3", "FON", "DTLZ2"])
df = df.sort_values(by=["Method", "BatchSize", "Test","nParameters", "nObjectives"])


export_file = "AnalyzedData_Sorted.xlsx"
export_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), export_file)
df.to_excel(export_file, index=False)

pass

# For ZDT1 problem, let's go box plots