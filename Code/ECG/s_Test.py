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
    if v_strPaths[i_path] not in ['FSFB_0301_HRV.mat']:
        continue
    Path_dir = Data_dir + v_strPaths[i_path]
    dict_AllData = scipy.io.loadmat(Path_dir)
    d_sampleRate = np.array(dict_AllData['d_sRate'])[0][0]
    v_ECG = np.array(dict_AllData['v_ECG'][0])
    v_ECGTime = np.arange(len(v_ECG)) / d_sampleRate

    d_windSec = 40
    d_windIni = int(50*d_sampleRate)
    d_windSize = int(d_windSec*d_sampleRate)
    kw = dict(color='k', linewidth=1)


    # plt.plot(v_ECGTime, v_ECG, **kw)

    v_ECGWind = v_ECG[0+d_windIni:d_windSize+d_windIni]
    v_ECGWindTime = np.arange(len(v_ECGWind)) / d_sampleRate

    v_ECGWind = v_ECG[0+d_windIni:d_windSize+d_windIni]
    v_ECGWindTime = np.arange(len(v_ECGWind)) / d_sampleRate

    fig, ax = plt.subplots(1, 1, figsize=(10, 2), sharex=True)
    # plt.plot(v_ECGWindTime,v_ECGWind, **kw)


    # v_ECGDataClean, _, v_peaksClean, m_timeRemoveLimits = cleanECGbyTachogram(v_ECG, d_sampleRate, 0)
    # v_ECGTimeClean = np.arange(0, len(v_ECGDataClean)) / d_sampleRate
##


    v_taco = np.array(dict_AllData['v_RRTaco'][0])
    v_BPM = np.array(dict_AllData['v_BPM'][0])
    v_time = np.array(dict_AllData['v_TacoTime'][0])
    v_SessionTimes = np.array(dict_AllData['v_SessionTimes'][0])
    v_SessionIndx = [np.where(v_time > v_SessionTimes[0])[0][0], np.where(v_time > v_SessionTimes[1])[0][0]]

    plt.plot(v_time, v_taco,**kw)