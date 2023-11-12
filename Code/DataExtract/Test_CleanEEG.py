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
from scipy.ndimage import gaussian_filter
from matplotlib import colors
import matplotlib.pylab as pl

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

# v_strPaths = ['FSFB_0801']  # List of path names
v_FreqBands = [[1, 4], [4, 8], [8, 12], [12, 18], [18, 30]]  # Frequency bands to analyze
v_FreqBands_Names = ['Delta', 'Theta', 'Alpha', 'Low Beta', 'Fast Beta']  # Names of the frequency bands
d_windSize = 3  # Window size for power analysis
d_windStep = 1.5  # Window step for power analysis
d_maxAplitud = 100  # Maximum amplitude threshold

# -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Main Loop
# -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lista = []
for i_path in range(len(v_strPaths)):  # Iterate over each path in the list

    Path_dir = Data_dir + 'DataMat_' + v_strPaths[i_path] + '.mat'  # Construct the path to the data file
    d_AllData = scipy.io.loadmat(Path_dir)  # Load the data from the MATLAB file
    print(f'############################################################')
    print(f'Processing path: {v_strPaths[i_path]} - {i_path + 1}/{len(v_strPaths)}')
    print(f'############################################################')
    # -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Relevant Data
    # -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    m_data = d_AllData['m_Data']  # Filtered Channels EEG
    d_sampleRate = d_AllData['d_sampleRate'][0][0]  # Sampling rate (Hz)
    v_TimeArray = np.arange(0, np.size(m_data[0])) / d_sampleRate  # Time values
    v_ChanNames = d_AllData['v_ChanNames']  # Channel names by order
    v_SessionTimes = d_AllData['v_sessionTimes'][0]  # Times of the MT session [Start, End]
    str_name = v_strPaths[i_path] + '_PSD.mat'
    print(f'Time PRE: {v_SessionTimes[0]/60}')
    print(f'Time MTI: {(v_SessionTimes[1] - v_SessionTimes[0])/60}')
    lista.append(v_TimeArray[-1] - v_SessionTimes[1])

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

##
lista = np.array(lista)
# ##
# fig, ax = plt.subplots(3, 1, sharex=True)
#
# d_timeSec = 5
# d_windSize = int(d_timeSec * d_sampleRate)
# d_posIni = int(d_timeSec * d_sampleRate)
# v_plotTime = np.arange(d_windSize) / d_sampleRate
#
# kw = dict(color='k', linewidth=0.75)
# ax[0].plot(v_plotTime, m_data[7][d_posIni:d_posIni + d_windSize], **kw)
# ax[1].plot(v_plotTime, m_data[8][d_posIni:d_posIni + d_windSize], **kw)
# ax[2].plot(v_plotTime, m_data[9][d_posIni:d_posIni + d_windSize], **kw)
#
# for i in ax: i.axis(False)
# # ax[2].spines['bottom'].set_visible(True)
#
# fig.subplots_adjust(hspace=0, wspace=0.1)
#
# ##
#
# factor = 16
# f, t, Sxx = signal.spectrogram(m_data[9], d_sampleRate, nfft=d_sampleRate * factor,
#                                nperseg=d_sampleRate * int(factor / 2),
#                                noverlap=d_sampleRate * (factor / 2 - 0.5))
#
# # Sxx = f_Matrix2RelAmplitud(Sxx)
# # Sxx = gaussian_filter(Sxx, sigma=3)
# # Sxx = f_Matrix2RemLogTrend(Sxx, f)
#
# v_EMG = copy.copy(m_data[9])
# d_threshold = 800
# v_EMG[v_EMG > d_threshold] = d_threshold
# v_EMG[v_EMG < -1 * d_threshold] = -1 * d_threshold
#
# cmap = colors.ListedColormap(['#35558B', '#52AF6B', '#EFE64C', '#EFE64C', '#EFE64C', '#EFE64C'])
# cmap = colors.ListedColormap(['#35558B', '#52AF6B', '#EFE64C', '#EFE64C', '#EFE64C', '#EFE64C'])
#
# cmap = pl.cm.get_cmap('Reds', 11)
# # bounds = [0, 0.02, 0.07, 0.20, 0.3, 0.4]
# # norm = colors.BoundaryNorm(bounds, cmap.N)
# ##
# fig, ax = plt.subplots(2, 1, sharex=True, constrained_layout=True)
#
# ax[0].plot(v_TimeArray, m_data[9], **kw)
# ax[0].set_ylim(-1000, 1000)
# immat = ax[1].imshow(Sxx, cmap='viridis', interpolation='none',
#                      origin='lower', aspect='auto',
#                      extent=[t[0], t[-1],
#                              f[0], f[-1]],
#                      # norm=norm
#                      vmin=0, vmax=0.5
#                      )
# fig.colorbar(immat, ax=ax[1])
# # plt.pcolormesh(t, f, Sxx, shading='gouraud')
# ax[1].set_ylabel('Frequency (Hz)')
# ax[1].set_xlabel('Time (s)')
# ax[1].set_ylim(1, 200)
# plt.show()
#
# # fig.subplots_adjust(hspace=1)
#
# ## PLot EEG sample
# from scipy.signal import welch
# fig, ax = plt.subplots(1, 1, figsize=(5, 6), sharex=True)
#
# d_timeSec = 3
# d_windSize = int(d_timeSec * d_sampleRate)
# d_posIni = int(500 * d_sampleRate)
# v_plotTime = np.arange(d_windSize) / d_sampleRate
#
# kw = dict(color='k', linewidth=0.5)
# for i_plot in range(6):
#     v_idata = m_data[i_plot][d_posIni:d_posIni + d_windSize]
#     d_length = len(v_idata)  # Get the length of the input signal
#     v_freqs, v_psd = welch(v_idata, d_sampleRate, nfft=d_length)  # Estimate the (PSD) of the signal using Welch's method
#     v_freq_res = v_freqs[1] - v_freqs[0]
#     idx_band = np.logical_and(v_freqs >= 1, v_freqs <= 30)
#     v_freqs, v_psd = v_freqs[idx_band], v_psd[idx_band]
#     ax.semilogy(v_freqs, v_psd, **kw)
#     # ax[i_plot].fill_between((1, 4), max(v_idata), min(v_idata), alpha=0.5, color= 'grey')
#     # ax[i_plot]
#     # ax.axis(False)
# # ax.hline(7)
# # ax.set_xlim(1,30)
# matplotlib.pyplot.subplots_adjust(hspace=-0.25)
#
