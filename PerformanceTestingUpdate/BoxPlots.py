import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# mplstyle does not work with pandas boxplots
#style = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'boxplots.mplstyle')
#plt.style.use(style)
plt.rcParams.update({'figure.figsize': [7,10],
              'xtick.direction': 'in',
              'xtick.top': True,
              'xtick.minor.bottom': True,
              'xtick.minor.top': True,
              'xtick.minor.visible': True,
              'ytick.direction': 'inout'})


def save_all_figures_pdf(filename):
    f = PdfPages(filename)
    figures_list = [plt.figure(i) for i in plt.get_fignums()]

    for single_figure in figures_list:
        single_figure.savefig(f, format='pdf')

    f.close()


export_file = "output.xlsx"
export_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), export_file)

data = pd.read_excel(export_file)

df = pd.DataFrame(data, columns= ['Method', 'Number of Batch Samples', 'Test Name', 'Number of Parameters',\
              'Number of Objectives', 'Pareto Points', 'DOS', 'IHD'])

# rename PenaltyEIMe to EIMe & PenaltyQualityMetricIHDMOS and QualityMetricIHDMOS to QM
df.loc[df['Method'] == 'PenaltyEIMe', 'Method'] = 'EIMe'
df.loc[df['Method'] == 'QualityMetricIHDMOS', 'Method'] = 'QM'
df.loc[df['Method'] == 'PenaltyQualityMetricIHDMOS', 'Method'] = 'QM'


metrics = ['Pareto Points', 'DOS', 'IHD']
Tests = ['ZDT1', 'ZDT2', 'ZDT3', 'FON']
Methods = pd.Series(df['Method']).drop_duplicates().tolist()
BatchSizes = [2, 3]

# make a row of subplots for each test (uses all data)
for test in Tests:
    for batchsize in BatchSizes:
        df_test = df.loc[df['Test Name'] == test]
        df_batch = df_test.loc[df_test['Number of Batch Samples'] == batchsize]
        
        #fig, axes = plt.subplots(1, len(metrics), sharey = False) #1 row 3 col
        fig, axes = plt.subplots(len(metrics), 1, sharey = False) #3 row 1 col
        
        # can change colors/properties with boxprops, meanprops, medianprops
        boxplot = df_batch.plot.box(by = "Method", column = metrics, grid = False, ax = axes, vert=False,
                                color = dict(boxes ='k', whiskers = 'k', medians = 'k', caps = 'k'),
                                showfliers=False)
        title_txt = test + ' (b = ' + str(batchsize) + ')'
        fig.suptitle(title_txt)
        plt.tight_layout()

# DTLZ2 plots (uses all data)
df_dtlz2 = df.loc[df['Test Name'] == 'DTLZ2']

for num_obj in pd.Series(df_dtlz2['Number of Objectives']).drop_duplicates().tolist():
    for batchsize in BatchSizes:
        df_test = df_dtlz2.loc[df_dtlz2['Number of Objectives'] == num_obj]
        df_batch = df_test.loc[df_test['Number of Batch Samples'] == batchsize]

        fig, axes = plt.subplots(len(metrics), 1, sharey = False)
        
        # can change colors/properties with boxprops, meanprops, medianprops
        boxplot = df_batch.plot.box(by = "Method", column = metrics, grid = False, ax = axes, vert=False,
                                color = dict(boxes ='k', whiskers = 'k', medians = 'k', caps = 'k'),
                                showfliers=False)

        fig.suptitle('DTLZ2 ' + str(num_obj) + ' Objectives' + ' (b = ' + str(batchsize) + ')')
        plt.tight_layout()

#plt.show()

filename = 'boxplots_v2.pdf'
save_all_figures_pdf(filename)
'''
# makes individual plots for each test/metric combo
for metric in metrics:
    for test in Tests:
        df_test = df.loc[df['Test Name'] == test]

        boxplot = df_test.boxplot(by = "Method", column = metric, grid = False)
        boxplot.plot()
        plt.show()
'''