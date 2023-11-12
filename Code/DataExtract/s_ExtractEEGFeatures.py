## -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                     Imports
# -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import copy
import os
import scipy.io
import numpy as np
import pandas as pd
from Functions.f_EEGPowerAnalysis import *
from Functions.f_AnalysisFunctions import *
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import scipy as sci

# -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Paths
# -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data_dir = './../Data/MatData/'  # Define the directory path for filtered and artifact-removed data
Out_dir = './../Data/EEGFeatures/'  # Define the output directory path to save matrices and information
df_AllInfo = pd.read_csv('./../RawData/data_info.csv', sep=';')  # Read a CSV file using pandas
v_strPaths = df_AllInfo['name_path'].tolist()  # Convert a column of the DataFrame to a list

if not os.path.isdir(Out_dir):  # Check if the output directory exists; if not, create it
    os.mkdir(Out_dir)

s_ReplaceData = True  # Flag indicating whether to replace existing data

# -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Variables
# -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

v_strPaths = ['FSFB_0101']  # List of path names


# v_FreqBands = [[1, 4], [4, 8], [8, 12], [12, 18], [18, 30]]  # Frequency bands to analyze
v_FreqBands = [[1, 4], [4, 8], [8, 12], [12, 18], [18, 30]]  # Frequency bands to analyze

v_FreqBands_Names = ['Delta', 'Theta', 'Alpha', 'Low Beta', 'Fast Beta']  # Names of the frequency bands
d_windSize = 3  # Window size for power analysis
d_windStep = 1.5  # Window step for power analysis
d_maxAplitud = 100  # Maximum amplitude threshold

# -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Main Loop
# -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i_path in range(len(v_strPaths)):  # Iterate over each path in the list
    Path_dir = Data_dir + 'DataMat_' + v_strPaths[i_path] + '.mat'  # Construct the path to the data file
    d_AllData = scipy.io.loadmat(Path_dir)  # Load the data from the MATLAB file

    # -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Relevant Data
    # -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    m_data = d_AllData['m_Data']  # Filtered Channels EEG
    d_sampleRate = d_AllData['d_sampleRate'][0][0]  # Sampling rate (Hz)
    v_TimeArray = np.arange(0, np.size(m_data[0])) / d_sampleRate  # Time values
    v_ChanNames = d_AllData['v_ChanNames']  # Channel names by order
    v_SessionTimes = d_AllData['v_sessionTimes'][0]  # Times of the MT session [Start, End]
    str_name = v_strPaths[i_path] + '_rawPSD.mat'

    if str_name in os.listdir(Out_dir) and not s_ReplaceData:  # Check if the file already exists and not replacing data
        print(f'############################################')
        print(f'WARNING! --- {str_name} Already exists in the path.')
        print(f'############################################')
        d_AllData = scipy.io.loadmat(Out_dir + str_name)  # Load the existing file
        m_PatAbsPsd = d_AllData['AbsPSD']  # Extract the absolute PSD matrix
        continue

    # - ########################################################
    # Extract Band Power per Channel and per Band
    # - ########################################################

    v_ChanNamesEEG = []  # Initialize list for EEG channel names
    m_PatAbsPsd = []  # Initialize list for absolute PSD matrix

    for i_chan in range(len(v_ChanNames)):  # Iterate over each channel
        if v_ChanNames[i_chan] in ['ECG', 'EMG', 'EOG']:  # Skip non-EEG channels
            continue
        v_ChanNamesEEG.append(v_ChanNames[i_chan])  # Add EEG channel name to the list
        print('___________________________________________')
        print(f'Processing patient: {v_strPaths[i_path]} -- Channel: {v_ChanNames[i_chan]}')

        v_dataChann = copy.copy(m_data[i_chan])  # Create a copy of the EEG channel data
        v_dataChann[v_dataChann > d_maxAplitud] = d_maxAplitud  # Clip values above the maximum amplitude threshold
        v_dataChann[
            v_dataChann < -d_maxAplitud] = -d_maxAplitud  # Clip values below the negative maximum amplitude threshold

        m_AbsPsd = []  # Initialize list for absolute PSD per band
        for i_band in range(len(v_FreqBands)):  # Iterate over each frequency band
            print(f'  --- Processing frequency band: {v_FreqBands[i_band]}')
            v_AbsPSD = f_powerEvolution(v_dataChann, v_FreqBands[i_band], d_sampleRate, d_windSize, d_windStep)
            # Calculate power evolution for the current channel and frequency band
            m_AbsPsd.append(v_AbsPSD)  # Add the calculated power evolution to the list
        m_PatAbsPsd.append(m_AbsPsd)  # Add the absolute PSD matrix to the list
    m_PatAbsPsd = np.array(m_PatAbsPsd)  # Convert the list to a NumPy array

    ## -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Clean Data and Extract Features
    # -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    d_BslWind = int((2.5 * 60) / d_windStep)  # Baseline window size
    d_MtlWind = int((7 * 60) / d_windStep)  # Music therapy window size
    d_AflWind = int((2.5 * 60) / d_windStep)  # Post intervention window size
    m_PatFinalAbsPsd = cleanDataExract(m_PatAbsPsd, v_SessionTimes, v_ChanNamesEEG, v_FreqBands, d_windStep, d_BslWind,
                                       d_MtlWind, d_AflWind, 3, True)
    # Clean the data and extract features using custom function "cleanDataExract()"

    m_interData = []  # Initialize list for interpolated data
    for i_band in range(len(m_PatFinalAbsPsd)):  # Iterate over each frequency band
        m_interChan = []  # Initialize list for interpolated channel data
        for i_chan in range(len(m_PatFinalAbsPsd[0])):  # Iterate over each channel
            v_dataWind = m_PatFinalAbsPsd[i_band][i_chan]  # Extract data for the current frequency band and channel
            v_timeWind = np.arange(len(v_dataWind)) * d_windStep  # Generate time values for the window

            d_maxPower = np.mean(v_dataWind) + (np.std(v_dataWind) * 10)  # Calculate the maximum power threshold
            v_timeClean = v_timeWind[v_dataWind < d_maxPower]  # Extract time values below the threshold
            v_dataClean = v_dataWind[v_dataWind < d_maxPower]  # Extract data values below the threshold

            f_splineInter = sci.interpolate.CubicSpline(v_timeClean, v_dataClean, bc_type='natural')
            # Create a cubic spline interpolation function using cleaned data
            v_dataInterWind = f_splineInter(v_timeWind)  # Interpolate data for the entire window
            # v_dataInterWind[v_dataInterWind > 10] = 10

            m_interChan.append(v_dataInterWind)  # Add the interpolated data to the channel list
        m_interData.append(m_interChan)  # Add the channel list to the interpolated data list
    m_interData = np.array(m_interData)  # Convert the interpolated data list to a NumPy array

    # for i_band in range(len(v_FreqBands)):  # Iterate over each frequency band
    #     plt.figure()
    #     plt.plot(np.transpose(m_interData[:, i_band]))  # Plot the interpolated data
    #     plt.figure()
    #     plt.plot(np.transpose(m_PatAbsPsd[:, i_band]))  # Plot the original data

    d_Data = {'AbsPSD': m_interData,  # Create a dictionary with the interpolated data and other information
              'd_sampleRate': d_sampleRate,
              'v_ChanNames': v_ChanNamesEEG,
              'v_FreqBands': v_FreqBands,
              'v_SessionTimes': [int(d_BslWind), int(d_BslWind) + int(d_MtlWind)]}

    str_out = Out_dir + str_name  # Define the output file path
    scipy.io.savemat(str_out, d_Data)  # Save the data dictionary to a MATLAB file
    print(f'---------------------------------------------')
    print(f'{str_name} successfully saved in the path.')
