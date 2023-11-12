import matplotlib.pyplot as plt

from Functions.f_AnalysisStatistics import *
from Functions.f_AnalysisFunctions import *
from Functions.f_SignalProcFuncLibs import *
from Functions.f_AnalysisGraphs import *
import os
import pandas as pd
import scipy.io
import numpy as np
import scipy as sci
import matplotlib
matplotlib.use('TkAgg')


b_plotPSD = True
b_stats = True
b_boxMNF = True
Data_dir = './../Data/EMGFeatures/'  # Filtered and removed artifact data
Out_dir = './../EMG/Results/'  # Out path to save the matrix and important information
if not os.path.isdir(Out_dir):  # Create the path if this doesn't exist
    os.mkdir(Out_dir)

v_strPaths = os.listdir(Data_dir)

m_allPathsPSD_Bs = []
m_allPathsPSD_Mt = []
m_allPathsPSD_Af = []

m_allPathsMNF = []
for i_path in range(len(v_strPaths)):
    Path_dir = Data_dir + v_strPaths[i_path]
    dict_AllData = scipy.io.loadmat(Path_dir)

    v_sessionTimes = dict_AllData['SessionTimes'][0]
    v_dataEMG = np.array(dict_AllData['v_Data'][0])
    v_freq = dict_AllData['v_freqs'][0]
    d_max = np.where(v_freq > 220)[0][0]
    d_min = np.where(v_freq > 0)[0][0]
    d_sampleRate = dict_AllData['s_Freq']
    d_windSec = dict_AllData['d_windSec'][0]
    d_stepSec = dict_AllData['d_stepSec'][0]
    v_freq = v_freq[d_min:d_max]
    v_time = np.arange(0, np.size(v_dataEMG)) / d_sampleRate
    v_time = v_time[0]

    # plt.figure()
    # plt.plot(v_time,v_dataEMG,'k')

    m_psd = dict_AllData['m_Psd']

    m_psd_Bs = m_psd[:int(v_sessionTimes[0] / d_stepSec), :]
    m_psd_Mt = m_psd[int(v_sessionTimes[0] / d_stepSec):int(v_sessionTimes[1] / d_stepSec), :]
    m_psd_Af = m_psd[int(v_sessionTimes[1] / d_stepSec):, :]

    v_psd_Bs = np.mean(m_psd_Bs, 0)[d_min:d_max]
    v_psd_Mt = np.mean(m_psd_Mt, 0)[d_min:d_max]
    v_psd_Af = np.mean(m_psd_Af, 0)[d_min:d_max]

    d_meanFreq_Bs = np.sum(v_psd_Bs * v_freq) / np.sum(v_psd_Bs)
    d_meanFreq_Mt = np.sum(v_psd_Mt * v_freq) / np.sum(v_psd_Mt)
    d_meanFreq_Af = np.sum(v_psd_Af * v_freq) / np.sum(v_psd_Af)
    m_allPathsMNF.append([d_meanFreq_Bs, d_meanFreq_Mt, d_meanFreq_Af])

    m_allPathsPSD_Bs.append(v_psd_Bs / np.sum(v_psd_Bs))
    m_allPathsPSD_Mt.append(v_psd_Mt / np.sum(v_psd_Mt))
    m_allPathsPSD_Af.append(v_psd_Af / np.sum(v_psd_Af))

##
# Stats
m_allPathsMNF = np.array(m_allPathsMNF)

m_allPathsPSD_Bs = np.array(m_allPathsPSD_Bs)
m_allPathsPSD_Mt = np.array(m_allPathsPSD_Mt)
m_allPathsPSD_Af = np.array(m_allPathsPSD_Af)

m_allPathsMeanPSD_Bs = np.mean(m_allPathsPSD_Bs, 0)
m_allPathsMeanPSD_Mt = np.mean(m_allPathsPSD_Mt, 0)
m_allPathsMeanPSD_Af = np.mean(m_allPathsPSD_Af, 0)
#
# v_freqFill = [60, 120]
#
# m_allPathsMeanPSD_Bs = fillPSDFilter(m_allPathsMeanPSD_Bs, v_freq, v_freqFill)
# m_allPathsMeanPSD_Mt = fillPSDFilter(m_allPathsMeanPSD_Mt, v_freq, v_freqFill)
# m_allPathsMeanPSD_Af = fillPSDFilter(m_allPathsMeanPSD_Af, v_freq, v_freqFill)

if b_stats:
    v_permTestMNF_BsvsMt = f_PermTest(m_allPathsMNF[:, 0], m_allPathsMNF[:, 1])
    v_permTestMNF_BsvsAf = f_PermTest(m_allPathsMNF[:, 0], m_allPathsMNF[:, 2])
    v_permTestMNF_MtvsAf = f_PermTest(m_allPathsMNF[:, 1], m_allPathsMNF[:, 2])

if b_plotPSD:
    d_softWind = 5
    v_dataPlot = [m_allPathsMeanPSD_Bs, m_allPathsMeanPSD_Mt, m_allPathsMeanPSD_Af]
    EMGPSDplot(v_dataPlot, v_freq, m_allPathsMNF, d_softWind)
    Out_dirBoxPSD = Out_dir + 'PSD/'
    if not os.path.isdir(Out_dirBoxPSD):  # Create the path if this doesn't exist
        os.mkdir(Out_dirBoxPSD)
    str_outPSD = f'AllPaths_EMGPSD.svg'
    plt.savefig(Out_dirBoxPSD + str_outPSD,transparent=True)
    plt.close()

if b_boxMNF:
    MyBoxPlot(m_allPathsMNF, True)
    Out_dirBoxPSD = Out_dir + 'BoxPlot/'
    if not os.path.isdir(Out_dirBoxPSD):  # Create the path if this doesn't exist
        os.mkdir(Out_dirBoxPSD)
    str_outPSD = f'AllPaths_MNFBoxPlot.svg'
    plt.savefig(Out_dirBoxPSD + str_outPSD, transparent=True)

