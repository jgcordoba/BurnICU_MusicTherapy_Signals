import matplotlib.pyplot as plt
from Functions.f_AnalysisFunctions import *
from Functions.f_AnalysisStatistics import *
from Functions.f_AnalysisGraphs import *
import os
import pandas as pd
import scipy.io
import numpy as np
import scipy as sci
import matplotlib

matplotlib.use('TkAgg')

Data_dir = './../Data/EMGFeatures/'  # Filtered and removed artifact data
Out_dir = './../EMG/Results/'  # Out path to save the matrix and important information
if not os.path.isdir(Out_dir):  # Create the path if this doesn't exist
    os.mkdir(Out_dir)

v_strPaths = os.listdir(Data_dir)
# v_strPaths = [v_strPaths[11]]
m_rms_slow = []
m_rms_fast = []
m_allPathMean = []
for i_path in range(len(v_strPaths)):
    # if v_strPaths[i_path] not in ['FSFB_0701_PSD.mat']:
    #     continue
    Path_dir = Data_dir + v_strPaths[i_path]
    dict_AllData = scipy.io.loadmat(Path_dir)
    d_sampleRate = dict_AllData['s_Freq'][0][0]
    v_sessionTimes = dict_AllData['SessionTimes'][0]

    v_dataEMG = np.array(dict_AllData['v_Data'][0])
    v_time = np.arange(0, np.size(v_dataEMG)) / d_sampleRate

    # print(v_sessionTimes)
    # plt.plot(v_time,v_dataEMG)

    d_windSec = dict_AllData['d_windSec'][0][0]
    d_stepSec = dict_AllData['d_stepSec'][0][0]
    d_windSize = d_windSec * d_sampleRate
    d_stepSize = d_stepSec * d_sampleRate

    v_SessionIndx = v_sessionTimes / d_stepSec
    v_SessionIndx = v_SessionIndx.astype(int)

    # st_Filt = f_GetIIRFilter(d_sampleRate, [19, 51], [20, 50])
    # st_Filt = f_GetIIRFilter(d_sampleRate, [49, 201], [50, 200])
    st_Filt = f_GetIIRFilter(d_sampleRate, [19, 201], [20, 200])
    v_dataEMG_fast = f_IIRBiFilter(st_Filt, v_dataEMG)

    # st_Filt = f_GetIIRFilter(d_sampleRate, [19, 51], [20, 50])
    # v_dataEMG_slow = f_IIRBiFilter(st_Filt, v_dataEMG)
    #
    v_rms_fast = []
    v_rms_slow = []
    for i_wind in range(int(len(v_dataEMG_fast) / d_stepSize)):
        v_wind = v_dataEMG_fast[int(i_wind * d_stepSize):int(i_wind * d_stepSize) + d_windSize]
        d_rms = np.sqrt(np.mean(v_wind ** 2))
        v_rms_fast.append(d_rms)

    # v_timeArange = np.arange(len(v_rms_fast)) * d_stepSec + 1
    # plt.plot(v_time, v_dataEMG_fast)
    # plt.plot(v_timeArange, v_rms_fast)

    # v_wind = v_dataEMG_slow[int(i_wind * d_stepSize):int(i_wind * d_stepSize) + d_windSize]
    # d_rms = np.sqrt(np.mean(v_wind ** 2))
    # v_rms_slow.append(d_rms)

    # v_rms_fast = (v_rms_fast - np.mean(v_rms_fast[:int(v_sessionTimes[0] / d_stepSec)])) / (np.std(
    #     v_rms_fast[:int(v_sessionTimes[0] / d_stepSec)]))

    v_rms_fast = v_rms_fast/max(v_rms_fast)


    v_rms_Bs = v_rms_fast[:v_SessionIndx[0]]
    v_rms_Mt = v_rms_fast[v_SessionIndx[0]:v_SessionIndx[1]]
    v_rms_Af = v_rms_fast[v_SessionIndx[1]:]

    v_rmsMean = [np.mean(v_rms_Bs),
                 np.mean(v_rms_Mt),
                 np.mean(v_rms_Af)]

    m_allPathMean.append(v_rmsMean)
    #
    # v_rms_slow = (v_rms_slow - np.mean(v_rms_slow[:int(v_sessionTimes[0] / d_stepSec)])) / np.std(
    #     v_rms_slow[:int(v_sessionTimes[0] / d_stepSec)])
    #
    v_rms_fast = OneDimFullFill(v_rms_fast, v_sessionTimes, d_stepSec)
    v_rmsTime = np.arange(0, len(v_rms_fast)) / d_stepSec
    m_rms_fast.append(f_averageMean(v_rms_fast, 20))
#
#     v_rms_slow = OneDimFullFill(v_rms_slow, v_sessionTimes, d_stepSec)
#     v_rmsTime = np.arange(0, len(v_rms_slow)) / d_stepSec
#     m_rms_slow.append(f_averageMean(v_rms_slow, 20))
#
#     # plt.plot(v_rmsTime, AverageMean(v_rms, 15), label=v_strPaths[i_path])
#
# # plt.legend()
# m_rms_slow = np.array(m_rms_slow)
# m_rms_slow[m_rms_slow == 0] = np.nan
# v_meanRms_slow = np.nanmean(m_rms_slow, 0)
# v_stdRms_slow = np.nanstd(m_rms_slow, 0) / np.sqrt(len(m_rms_slow[:, 0]))
# v_rmsTime_slow = np.arange(0, len(v_meanRms_slow)) / d_stepSec

##
m_allPathRMS = np.array(m_allPathMean)
#Statistics

v_permTestRMS_BsvsMt = f_PermTest(m_allPathRMS[:, 0], m_allPathRMS[:, 1])
v_permTestRMS_BsvsAf = f_PermTest(m_allPathRMS[:, 0], m_allPathRMS[:, 2])
v_permTestRMS_MtvsAf = f_PermTest(m_allPathRMS[:, 1], m_allPathRMS[:, 2])

MyBoxPlot(m_allPathMean, False)


##
# a = f_PermTest(m_allPathMean[:,0],m_allPathMean[:,1])

m_rms_fast = np.array(m_rms_fast)
m_rms_fast[m_rms_fast == 0] = np.nan
v_meanRms_fast = np.nanmean(m_rms_fast, 0)
v_stdRms_fast = np.nanstd(m_rms_fast, 0) / np.sqrt(len(m_rms_fast[:, 0]))
v_rmsTime_fast = np.arange(0, len(v_meanRms_fast)) / d_stepSec

##
d_windSoft = 1
colors = pl.cm.Reds(np.linspace(0, 1, 5))
fig, ax = plt.subplots(2,1 ,figsize=(10, 6))
# for i_axis in ax: i_axis.axis(False)


ax[0].plot(v_time, v_dataEMG)
ax[1].plot(v_rmsTime, f_averageMean(v_meanRms_fast, d_windSoft), color=colors[4], linewidth=2)
# ax.plot(v_rmsTime, f_averageMean(v_meanRms_slow, d_windSoft), color=colors[3], linewidth=2)
#
ax[1].plot(v_rmsTime, f_averageMean(v_meanRms_fast + v_stdRms_fast, d_windSoft), color=colors[4], linewidth=0.5)
ax[1].plot(v_rmsTime, f_averageMean(v_meanRms_fast - v_stdRms_fast, d_windSoft), color=colors[4], linewidth=0.5)
ax[1].fill_between(v_rmsTime, f_averageMean(v_meanRms_fast + v_stdRms_fast, d_windSoft), f_averageMean(v_meanRms_fast - v_stdRms_fast, d_windSoft),
                alpha=0.3, color=colors[4])

##
v_freqs, v_psd = signal.welch(v_wind, d_sampleRate, nperseg=len(v_wind)/3)
v_psd =f_averageMean(v_psd, 4)

plt.plot(v_freqs,v_psd)
# ax.fill_between(v_rmsTime, f_averageMean(v_meanRms_slow + v_stdRms_slow, d_windSoft), f_averageMean(v_meanRms_slow - v_stdRms_slow, d_windSoft),
#                 alpha=0.3, color=colors[3])
#
# ax.axhline(0, v_rmsTime[0], v_rmsTime[-1], color='k')
# plt.vlines([300, 1200], -1, 1, linestyles='--', color='k', linewidth=1.5)
# ax.set_ylim(-1.2, 1.2)
# ax.tick_params(axis='both', which='major', labelsize=20)
# ax.set_xlim(v_rmsTime[0] - 100, v_rmsTime[-1] + 100)
# [ax.spines[i].set_visible(False) for i in ['top', 'right', 'bottom', 'left']]
#
# Out_dirBoxPSD = Out_dir + 'RMS/'
# if not os.path.isdir(Out_dirBoxPSD):  # Create the path if this doesn't exist
#     os.mkdir(Out_dirBoxPSD)
# str_outPSD = f'AllPaths_RMSPlot.svg'
# plt.savefig(Out_dirBoxPSD + str_outPSD,transparent=True)
# plt.close()
