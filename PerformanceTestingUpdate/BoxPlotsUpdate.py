import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from pylab import *

# mplstyle does not work with pandas boxplots
#style = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'boxplots.mplstyle')
#plt.style.use(style)

#import_file = "outputChomp.xlsx"
import_file = "output.xlsx"
import_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), import_file)

#export_prefix = "Chomp"
export_prefix = "Normal"
export_folder = 'BoxPlotFigures'


def plot_bp(ax, data, label):
    return ax.boxplot(data, labels = label, notch=False, patch_artist=False, showfliers = False)



def format_plot(bp, ax):

    # colors = ['#ffff33','#a65628']
    # for patch in bp['boxes']:
    #     patch.set_facecolor('k')
    #     patch.set_alpha(0.7)

    for whisker in bp['whiskers']:
        whisker.set(color = 'k',linewidth=1)#,linestyle=':')

    for cap in bp['caps']:
        cap.set(color = 'k',linewidth = 1)

    for median in bp['medians']:
        median.set(color='k',linewidth = 1)

    for flier in bp['fliers']:
        flier.set(marker='.',color='#e7298a',alpha=0.5)

    #ax.set_ylim(-60.001,60.001)
    ax.minorticks_on()
    ax.tick_params(axis='y', which='minor', direction='in')
    ax.tick_params(axis='y', direction='in')
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(axis='x', direction='in')
    #ax.tick_params(axis='x',which='both',bottom=False)
    ax.yaxis.set_ticks_position('both')




data = pd.read_excel(import_file)


# changes the figure size [width, height]
# plt.rcParams.update({'figure.figsize': [10,7],
#               'ytick.direction': 'in',
#               'ytick.right': True,
#               'ytick.minor.left': True,
#               'ytick.minor.right': True,
#               'ytick.minor.visible': True,
#               'xtick.direction': 'inout'})

params = {
   'axes.labelsize': 7,
   'font.size': 7,
   'legend.fontsize': 7,
   'xtick.labelsize': 7,
   'ytick.labelsize': 7,
   'text.usetex': False,
   'figure.figsize': [3.543, (3.7*5/7)*.8]
   }
rcParams.update(params)


df = pd.DataFrame(data, columns= ['Method', 'Number of Batch Samples', 'Test Name', 'Number of Parameters',\
              'Number of Objectives', 'Pareto Points', 'DOS', 'IHD'])

# rename
df.loc[df['Method'] == 'QualityMetricIHDMOS', 'Method'] = 'QM'
df.loc[df['Method'] == 'PenaltyQualityMetricIHDMOS', 'Method'] = 'QM'
df.loc[df['Method'] == 'PenaltyEIMe', 'Method'] = 'EIMe'
df.loc[df['Test Name'] == 'WeldedBeam', 'Test Name'] = 'Beam'

metrics = ['Pareto Points', 'DOS', 'IHD']
Tests = ['ZDT1', 'ZDT2', 'ZDT3', 'FON']
MethodsList = pd.Series(df['Method']).drop_duplicates().tolist()


#test = 'ZDT1'

for test in ['ZDT1', 'ZDT2', 'ZDT3', 'FON']:
    for metric in ['Pareto Points', 'DOS', 'IHD']:
        #metric = 'Pareto Points'

        # Get everything by a specific test name
        df_test = df.loc[df['Test Name'] == test]

        data = []

        # Go through each of the methods
        for method in ['EIMe', 'QM', 'Ensemble', 'qEHVI']:
            df_method = df_test.loc[df_test['Method'] == method]

            # Go through each batch
            for batch in [1, 2, 3]:
                df_batch = df_method.loc[df_method['Number of Batch Samples'] == batch]
                data.append(np.array(df_batch[metric]))


        fig = figure()
        fig.subplots_adjust(left=0.12, right=.97, top=0.90, bottom= 0.15)
        ax = plt.subplot(111)

        bp = plot_bp(ax, data, ['1', '2', '3', '1', '2', '3', '1', '2', '3', '1', '2', '3'])

        format_plot(bp, ax)

        ax.set_ylabel(metric)
        plt.title('      EIMe             QM            Ensemble       qEHVI     ')
        plt.xlabel('Batch Size')

        ax.axvline(3.5, color = 'k', linestyle='--', linewidth = 1)
        ax.axvline(6.5, color = 'k', linestyle='--', linewidth = 1)
        ax.axvline(9.5, color = 'k', linestyle='--', linewidth = 1)

        export_file = export_prefix + '_' + test + '_' + metric


        export_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), export_folder, export_file)
        savefig(export_file + '.png', dpi=1000)

        plt.close()

#test = 'Beam'

for test in ['Beam']:
    for metric in ['Pareto Points', 'DOS', 'IHD']:
        #metric = 'Pareto Points'

        # Get everything by a specific test name
        df_test = df.loc[df['Test Name'] == test]

        data = []

        # Go through each of the methods
        for method in ['EIMe', 'QM', 'Ensemble', 'qEHVI']:
            df_method = df_test.loc[df_test['Method'] == method]

            # Go through each batch
            if method == 'qEHVI':
                batches = [1]
            else:
                batches = [1, 2, 3]

            for batch in batches:
                df_batch = df_method.loc[df_method['Number of Batch Samples'] == batch]
                data.append(np.array(df_batch[metric]))


        fig = figure()
        fig.subplots_adjust(left=0.12, right=.97, top=0.90, bottom= 0.15)
        ax = plt.subplot(111)

        bp = plot_bp(ax, data, ['1', '2', '3', '1', '2', '3', '1', '2', '3', '1'])

        format_plot(bp, ax)

        ax.set_ylabel(metric)
        plt.title(
                    '         EIMe                 QM               Ensemble   qEHVI',
                    loc='left'
                )
        
        plt.xlabel('Batch Size')

        ax.axvline(3.5, color = 'k', linestyle='--', linewidth = 1)
        ax.axvline(6.5, color = 'k', linestyle='--', linewidth = 1)
        ax.axvline(9.5, color = 'k', linestyle='--', linewidth = 1)

        export_file = export_prefix + '_' + test + '_' + metric


        export_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), export_folder, export_file)
        savefig(export_file + '.png', dpi=1000)

        plt.close()


test = 'DTLZ2'
# Get everything by a specific test name
df_test = df.loc[df['Test Name'] == test]
o = 3

df_objective = df_test.loc[df_test['Number of Objectives'] == o]
for metric in ['Pareto Points', 'DOS', 'IHD']:
    #metric = 'Pareto Points'

    data = []

    # Go through each of the methods
    for method in ['EIMe', 'QM', 'Ensemble']:
        df_method = df_objective.loc[df_objective['Method'] == method]

        # Go through each batch
        for batch in [1, 2, 3]:
            df_batch = df_method.loc[df_method['Number of Batch Samples'] == batch]
            data.append(np.array(df_batch[metric]))


    fig = figure()
    fig.subplots_adjust(left=0.12, right=.97, top=0.90, bottom= 0.15)
    ax = plt.subplot(111)

    bp = plot_bp(ax, data, ['1', '2', '3', '1', '2', '3', '1', '2', '3'])

    format_plot(bp, ax)
    
    ax.set_ylabel(metric)
    plt.title('   EIMe                    QM                  Ensemble')
    plt.xlabel('Batch Size')

    ax.axvline(3.5, color = 'k', linestyle='--', linewidth = 1)
    ax.axvline(6.5, color = 'k', linestyle='--', linewidth = 1)

    export_file = export_prefix + '_' + test + '_' + str(o) + '_' + metric

    export_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), export_folder, export_file)
    savefig(export_file + '.png', dpi=1000)

    plt.close()


for o in [4, 5]:

    df_objective = df_test.loc[df_test['Number of Objectives'] == o]
    for metric in ['Pareto Points', 'DOS', 'IHD']:
        #metric = 'Pareto Points'

        data = []

        # Go through each of the methods
        for method in ['EIMe', 'Ensemble']:
            df_method = df_objective.loc[df_objective['Method'] == method]

            # Go through each batch
            for batch in [1, 2, 3]:
                df_batch = df_method.loc[df_method['Number of Batch Samples'] == batch]
                data.append(np.array(df_batch[metric]))


        fig = figure()
        fig.subplots_adjust(left=0.12, right=.97, top=0.90, bottom= 0.15)
        ax = plt.subplot(111)

        bp = plot_bp(ax, data, ['1', '2', '3', '1', '2', '3'])

        format_plot(bp, ax)
        ax.set_ylabel(metric)

        plt.title('      EIMe                              Ensemble')
        plt.xlabel('Batch Size')
        ax.axvline(3.5, color = 'k', linestyle='--', linewidth = 1)
        export_file = export_prefix + '_' + test + '_' + str(o) + '_' + metric
        

        export_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), export_folder, export_file)
        savefig(export_file + '.png', dpi=1000)

        plt.close()


""" # make a row of subplots for each test
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

filename = 'boxplotupdate.pdf'
save_all_figures_pdf(filename) """