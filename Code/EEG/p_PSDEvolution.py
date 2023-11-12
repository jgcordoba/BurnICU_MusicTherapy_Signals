import matplotlib

matplotlib.use('TkAgg')  # Set the backend for matplotlib
import matplotlib.pyplot as plt
import os
import scipy
import scipy.io
import numpy as np
from Functions.f_AnalysisGraphs import *  # Import custom analysis graph functions
from Functions.f_AnalysisFunctions import *  # Import custom analysis functions
from Functions.f_EEGPowerAnalysis import *  # Import custom EEG power analysis functions
import matplotlib.pylab as pl
import pandas as pd

b_indvPSDEvolution = True  # Flag for individual PSD evolution analysis
s_allPathsPSDEvolution = True  # Flag for all paths PSD evolution analysis

Data_dir = '../Data/EEGFeatures/'  # Directory containing EEG features data
Out_dir = 'Results/PSDEvolution/'  # Output directory for results
os.makedirs(Out_dir, exist_ok=True)  # Create the output directory if it doesn't exist

df_AllInfo = pd.read_csv('./../RawData/data_info.csv', sep=';')  # Read the data info from a CSV file
v_strPaths = df_AllInfo['name_path'].tolist()  # Path Names

# v_strPaths = ['FSFB_0101']  # Path Names (uncomment to process specific paths)
v_FreqBands_Names = ['Delta', 'Theta', 'Alpha', 'Slow Beta', 'Fast Beta']  # Names of the frequency bands

s_windSize = 3  # Size of the window
s_windStep = 1.5  # Step size of the window

m_AbsPSD_AllPaths = []  # List to store the absolute PSD data for all paths
m_RelPSD_AllPaths = []  # List to store the relative PSD data for all paths
v_Allpat_SessionTimes = [200, 800]  # Session times for all patients

# Loop through each path
for i_path in range(len(v_strPaths)):
    str_name = v_strPaths[i_path] + '_PSD.mat'  # Construct the file name
    if str_name in os.listdir(Data_dir):  # Check if the file exists
        print('___________________________________________')
        print(f'Processing pat: {v_strPaths[i_path]}')

        d_AllData = scipy.io.loadmat(Data_dir + str_name)  # Load the data from the mat file
        m_AbsPSD = d_AllData['AbsPSD']  # Absolute PSD data
        v_TimeArray = np.arange(0, np.size(m_AbsPSD[0])) / s_windStep  # Time values
        v_ChanNames = d_AllData['v_ChanNames']  # Channel names by order
        v_SessionTimes = d_AllData['v_SessionTimes'][0]  # Times of the MT session [Start, End]
        v_FreqBands = d_AllData['v_FreqBands']  # Frequency bands

        if b_indvPSDEvolution:
            s_AbsTitle = f'Absolute PSD evolution - Pat: {v_strPaths[i_path]} '

            # Perform individual PSD evolution analysis
            psdEvolution(m_AbsPSD, s_windStep, v_FreqBands_Names, v_FreqBands, v_ChanNames,
                         v_SessionTimes * s_windStep, s_AbsTitle, relative=False)

            str_out = v_strPaths[i_path] + '_AbsPsd_BandPower.png'
            Out_dirAbs = Out_dir + 'AbsPsd/'
            os.makedirs(Out_dirAbs, exist_ok=True)  # Create the output directory for individual PSD if it doesn't exist
            plt.savefig(Out_dirAbs + str_out)  # Save the figure
            plt.close()

        if s_allPathsPSDEvolution:
            # Fill the data for all paths to have the same session times
            m_AbsPSDFull = FullfillData(m_AbsPSD, v_SessionTimes, s_windStep)
            m_AbsPSD_AllPaths.append(m_AbsPSDFull)

    else:
        print(f'##################################################')
        print(f'WARNING - File {str_name} doesnt exist')
        print(f'##################################################')
        break

if s_allPathsPSDEvolution:
    # Perform mean PSD evolution analysis for all paths
    m_dataAllPathsMean_AbsPsd, m_dataAllPathsStd_AbsPsd = MeanPsdEvolution(m_AbsPSD_AllPaths)
    s_AbsTitleAllPaths = f'Absolute PSD evolution - All Paths'

    # Plot the band evolution for all paths
    PlotBandEvolutionAllPaths(m_dataAllPathsMean_AbsPsd, m_dataAllPathsStd_AbsPsd / np.sqrt(16), s_windStep,
                              v_FreqBands_Names, v_FreqBands, v_ChanNames, np.array([200, 800]) * s_windStep,
                              s_AbsTitleAllPaths, relative=False)

    Out_dirAbs = Out_dir + 'AbsPsd/'
    os.makedirs(Out_dirAbs, exist_ok=True)  # Create the output directory for all paths PSD if it doesn't exist
    str_outAbs = 'AllPaths_AbsPsd_BandPower.svg'
    plt.savefig(Out_dirAbs + str_outAbs, transparent=True)  # Save the figure

"""
Importing libraries:

The code starts by importing the necessary libraries, including matplotlib, os, scipy, numpy, and pandas. These libraries are used for data analysis, plotting, and file manipulation.
Setting flags and directories:

Two boolean flags, b_indvPSDEvolution and s_allPathsPSDEvolution, are set to control the analysis options.
The Data_dir variable is set to the directory path where the EEG features data is stored.
The Out_dir variable is set to the output directory path where the results will be saved. If the directory doesn't exist, it is created using os.makedirs().
Loading data info:

The code reads a CSV file called data_info.csv using pd.read_csv(). This file contains information about the data paths.
The paths are stored in the v_strPaths variable as a list of path names.
Looping over paths:

The code iterates over each path in the v_strPaths list using a for loop.
Inside the loop, it constructs the file name by appending "_PSD.mat" to the path name.
It checks if the file exists in the Data_dir directory using os.listdir().
Processing existing files:

If the file exists, the code proceeds to load the data from the mat file using scipy.io.loadmat().
The loaded data includes absolute PSD (m_AbsPSD), time array (v_TimeArray), channel names (v_ChanNames), session times (v_SessionTimes), and frequency bands (v_FreqBands).
Individual PSD evolution analysis:

If the b_indvPSDEvolution flag is set to True, the code performs individual PSD evolution analysis.
It calls the psdEvolution() function from a custom module, passing the necessary data and parameters for analysis.
The resulting figure is saved in the Out_dir directory under the AbsPsd subdirectory.
All paths PSD evolution analysis:

If the s_allPathsPSDEvolution flag is set to True, the code performs all paths PSD evolution analysis.
It calls the FullfillData() function to fill the data for all paths with the same session times.
The filled data is appended to the m_AbsPSD_AllPaths list.
Mean PSD evolution analysis for all paths:

After processing all paths, the code calls the MeanPsdEvolution() function to calculate the mean and standard deviation of the PSD evolution for all paths.
The resulting mean and standard deviation data are stored in m_dataAllPathsMean_AbsPsd and m_dataAllPathsStd_AbsPsd variables.
Plotting all paths PSD evolution:

The code calls the PlotBandEvolutionAllPaths() function to plot the band evolution for all paths.
The function takes the mean and standard deviation data, frequency band names, channel names, and session times as inputs.
The resulting figure is saved in the Out_dir directory under the AbsPsd subdirectory."""
