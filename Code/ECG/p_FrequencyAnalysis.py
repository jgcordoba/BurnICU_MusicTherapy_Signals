import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from Functions.f_AnalysisFunctions import *
from Functions.f_SignalProcFuncLibs import *
from Functions.f_ECGAnalysis import *
from Functions.f_AnalysisGraphs import *
from Functions.f_AnalysisStatistics import *
import os
import pandas as pd
import scipy.io
import numpy as np
import scipy as sci

import matplotlib

matplotlib.use('TkAgg')

Data_dir = './../Data/ECGFeatures/'  # Filtered and removed artifact data
Out_dir = './../ECG/Results/'  # Out path to save the matrix and important information
if not os.path.isdir(Out_dir):  # Create the path if this doesn't exist
    os.mkdir(Out_dir)

b_plotBox = True
b_plotPSD = True
b_stast = True
v_strPaths = os.listdir(Data_dir)
# v_strPaths = [v_strPaths[8]]
# v_strPaths = [v_strPaths[4], v_strPaths[9], v_strPaths[11], v_strPaths[14]]

m_density_Bs = []
m_density_Mt = []
m_density_Af = []
m_power_Bs = []
m_power_Mt = []
m_power_Af = []

d_FsHzNew = 10
for i_path in range(len(v_strPaths)):
    # if v_strPaths[i_path] in ['FSFB_0501_HRV.mat', 'FSFB_0502_HRV.mat']:
    #     continue
    Path_dir = Data_dir + v_strPaths[i_path]
    dict_AllData = scipy.io.loadmat(Path_dir)
    v_taco = np.array(dict_AllData['v_RRTaco'][0])
    v_BPM = np.array(dict_AllData['v_BPM'][0])
    v_time = np.array(dict_AllData['v_TacoTime'][0])
    v_SessionTimes = np.array(dict_AllData['v_SessionTimes'][0])
    v_SessionIndx = [np.where(v_time > v_SessionTimes[0])[0][0], np.where(v_time > v_SessionTimes[1])[0][0]]

    # v_taco = (v_taco - np.mean(v_taco[:v_SessionIndx[0]])) / np.std(v_taco[:v_SessionIndx[0]])

    v_timeInterpol = np.arange(0, v_time[-1], 1 / d_FsHzNew)
    f = sci.interpolate.CubicSpline(v_time, v_taco, bc_type='natural')
    v_tacoInterpol = f(v_timeInterpol)

    v_tacoInterpol_Bs = v_tacoInterpol[:int(v_SessionTimes[0] * d_FsHzNew), ]
    v_tacoInterpol_Mt = v_tacoInterpol[int(v_SessionTimes[0] * d_FsHzNew):int(v_SessionTimes[1] * d_FsHzNew)]
    v_tacoInterpol_Af = v_tacoInterpol[int(v_SessionTimes[1] * d_FsHzNew):]

    d_meanSize = 1
    d_freqMin = 0.04
    d_freqMid = 0.15
    d_freqMax = 0.4

    d_windSize = int(30 * d_FsHzNew)
    d_windStep = int(10 * d_FsHzNew)

    m_powers_Bs = []
    m_powers_Mt = []
    m_powers_Af = []

    v_freqRes, v_density_Bs, v_powers_Bs = extractECGfreq(v_tacoInterpol_Bs, d_FsHzNew, )
    m_density_Bs.append(f_averageMean(v_density_Bs, d_meanSize))
    m_power_Bs.append(v_powers_Bs)
    v_freqRes, v_density_Mt, v_powers_Mt = extractECGfreq(v_tacoInterpol_Mt, d_FsHzNew)
    m_density_Mt.append(f_averageMean(v_density_Mt, d_meanSize))
    m_power_Mt.append(v_powers_Mt)
    v_freqRes, v_density_Af, v_powers_Af = extractECGfreq(v_tacoInterpol_Af, d_FsHzNew)
    m_density_Af.append(f_averageMean(v_density_Af, d_meanSize))
    m_power_Af.append(v_powers_Af)

m_density_Bs = np.array(m_density_Bs)
m_density_Mt = np.array(m_density_Mt)
m_density_Af = np.array(m_density_Af)

v_meanDensity_Bs = np.mean(m_density_Bs, 0)
v_meanDensity_Mt = np.mean(m_density_Mt, 0)
v_meanDensity_Af = np.mean(m_density_Af, 0)


#
m_power_Bs = np.array(m_power_Bs)
m_power_Mt = np.array(m_power_Mt)
m_power_Af = np.array(m_power_Af)

v_Data_Rat = np.array([m_power_Bs[:, 0] / m_power_Bs[:, 1], m_power_Mt[:, 0] / m_power_Mt[:, 1],
                       m_power_Af[:, 0] / m_power_Af[:, 1]])

v_Data_Lf = np.array([m_power_Bs[:, 0], m_power_Mt[:, 0], m_power_Af[:, 0]])
v_Data_Hf = np.array([m_power_Bs[:, 1], m_power_Mt[:, 1], m_power_Af[:, 1]])

# v_Data_LfRel = v_Data_Lf / (v_Data_Hf + v_Data_Lf)
# v_Data_HfRel = v_Data_Hf / (v_Data_Hf + v_Data_Lf)
# v_Data_RatRel = v_Data_LfRel/v_Data_HfRel
# ##
# plt.plot(v_Data_HfRel[0], v_Data_LfRel[0], '.', color='k')
# plt.plot(v_Data_HfRel[1], v_Data_LfRel[1], '.', color='r')
# plt.plot(v_Data_HfRel[2], v_Data_LfRel[2], '.', color='b')

#
if b_plotBox:
    v_FreqBands_Names = ['Ratio', 'LF', 'HF']
    m_allFrequencies = [v_Data_Rat, v_Data_Lf, v_Data_Hf]
    Out_dirBoxPlot = Out_dir + 'BoxPlot/'
    if not os.path.isdir(Out_dirBoxPlot):  # Create the path if this doesn't exist
        os.mkdir(Out_dirBoxPlot)
    for i_band in range(len(m_allFrequencies)):
        MyBoxPlot(list(m_allFrequencies[i_band]), 0)
        str_outBox = f'AllPaths_BoxPlot_{v_FreqBands_Names[i_band]}.svg'
        plt.savefig(Out_dirBoxPlot + str_outBox, transparent=True)
        plt.close()

m_allDensity = [v_meanDensity_Bs, v_meanDensity_Mt, v_meanDensity_Af]

ECGPSDPlot(m_allDensity, v_freqRes, d_windSize=1)
if b_plotPSD:
    m_allDensity = [v_meanDensity_Bs, v_meanDensity_Mt, v_meanDensity_Af]
    ECGPSDPlot(m_allDensity, v_freqRes, d_windSize=1)

    Out_dirBoxPSD = Out_dir + 'PSD/'
    if not os.path.isdir(Out_dirBoxPSD):  # Create the path if this doesn't exist
        os.mkdir(Out_dirBoxPSD)
    str_outPSD = f'AllPaths_ECGPSD.svg'
    plt.savefig(Out_dirBoxPSD + str_outPSD, transparent=True)
    plt.close()

##
if b_stast:
    v_permTestLF_BsvsMt = f_PermTest(v_Data_Lf[1], v_Data_Lf[0])
    v_permTestLF_BsvsAf = f_PermTest(v_Data_Lf[2], v_Data_Lf[0])
    v_permTestLF_MtvsAf = f_PermTest(v_Data_Lf[1], v_Data_Lf[2])

    v_permTestHF_BsvsMt = f_PermTest(v_Data_Hf[1], v_Data_Hf[0])
    v_permTestHF_BsvsAf = f_PermTest(v_Data_Hf[2], v_Data_Hf[0])
    v_permTestHF_MtvsAf = f_PermTest(v_Data_Hf[1], v_Data_Hf[2])

    v_permTestRat_BsvsMt = f_PermTest(v_Data_Rat[1], v_Data_Rat[0])
    v_permTestRat_BsvsAf = f_PermTest(v_Data_Rat[2], v_Data_Rat[0])
    v_permTestRat_MtvsAf = f_PermTest(v_Data_Rat[1], v_Data_Rat[2])

##
indx1 = 1
indx2 = 0

stats = f_PermTest(v_Data_Lf[indx1], v_Data_Lf[indx2])
print(f'Mean = {np.mean(v_Data_Lf[indx1] - v_Data_Lf[indx2])}')
print(f'std = {np.std(v_Data_Lf[indx1] - v_Data_Lf[indx2])}')
print(f'pvalue = {stats[1][0]}')
##
