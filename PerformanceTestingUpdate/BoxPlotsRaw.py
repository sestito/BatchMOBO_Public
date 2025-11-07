import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def save_all_figures_pdf(filename):
    f = PdfPages(filename)
    figures_list = [plt.figure(i) for i in plt.get_fignums()]

    for single_figure in figures_list:
        single_figure.savefig(f, format='pdf')

    f.close()

# changes the figure size [width, height]
plt.rcParams.update({'figure.figsize': [10,7],
              'ytick.direction': 'in',
              'ytick.right': True,
              'ytick.minor.left': True,
              'ytick.minor.right': True,
              'ytick.minor.visible': True,
              'xtick.direction': 'inout'})

export_file = "output.xlsx"
export_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), export_file)

data = pd.read_excel(export_file)

df = pd.DataFrame(data, columns= ['Method', 'Number of Batch Samples', 'Test Name', 'Number of Parameters',\
              'Number of Objectives', 'Pareto Points', 'DOS', 'IHD'])

# renaming
df.loc[df['Method'] == 'PenaltyEIMe', 'Method'] = 'EIMe'
df.loc[df['Method'] == 'QualityMetricIHDMOS', 'Method'] = 'QM'
df.loc[df['Method'] == 'PenaltyQualityMetricIHDMOS', 'Method'] = 'QM'

metrics = ['Pareto Points', 'DOS', 'IHD']
Tests = ['ZDT1', 'ZDT2', 'ZDT3', 'FON']
MethodsList = pd.Series(df['Method']).drop_duplicates().tolist()

# make a row of subplots for each test
for test in Tests:
    for method in MethodsList:
        df_method = df.loc[df['Method'] == method]
        df_test = df_method.loc[df_method['Test Name'] == test]
        fig, axes = plt.subplots(1, len(metrics), sharey = False)
        plt.title(test)

        # can change colors/properties with boxprops, meanprops, medianprops
        #boxplot = df_test.boxplot(by = "Number of Batch Samples", column = metrics, grid = False, ax = axes,
                                #color = dict(boxes ='k', whiskers = 'k', medians = 'k', caps = 'k'))
        boxplot = df_test.plot.box(by = "Number of Batch Samples", column = metrics, grid = False, ax = axes,
                                color = dict(boxes ='k', whiskers = 'k', medians = 'k', caps = 'k'),
                                figsize = (3,8), showfliers=False)
        
        title_txt = test + ' & ' + method
        fig.suptitle(title_txt)
        title_text = fig.text(0.50, 0.02, 'Batch Size', horizontalalignment='center')


# hard coding the DTLZ2 plots
df_dtlz2 = df.loc[df['Test Name'] == 'DTLZ2']
# NOTE: showfliers=False will make outliers not show up on the plots

for method in MethodsList:
    for num_obj in pd.Series(df_dtlz2['Number of Objectives']).drop_duplicates().tolist():
        df_test = df_dtlz2.loc[df_dtlz2['Number of Objectives'] == num_obj]
        fig, axes = plt.subplots(1, len(metrics), sharey = False)
        plt.title('DTLZ2 ' + str(num_obj) + ' Objectives'  + ' & ' + method)
        
        # can change colors/properties with boxprops, meanprops, medianprops
        boxplot = df_test.boxplot(by = "Number of Batch Samples", column = metrics, grid = False, ax = axes,
                                color = dict(boxes ='k', whiskers = 'k', medians = 'k', caps = 'k'), 
                                showfliers=False)

        title_txt = 'DTLZ2 ' + str(num_obj) + ' Objectives'  + ' & ' + method
        fig.suptitle(title_txt)
        title_text = fig.text(0.50, 0.02, 'Batch Size', horizontalalignment='center')      


#plt.show()

# save all boxplots to a pdf

filename = 'boxplotRaw.pdf'
save_all_figures_pdf(filename)