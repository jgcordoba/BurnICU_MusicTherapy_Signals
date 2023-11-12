from Functions.f_AnalysisGraphs import *
from Functions.f_ECGAnalysis import *
from Functions.f_AnalysisGraphs import *
from Functions.f_AnalysisStatistics import *
import os
import scipy.io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
##
Data_dir = './../Data/ECGFeatures/'  # Filtered and removed artifact data
Out_dir = './../ECG/Results/'  # Out path to save the matrix and important information
if not os.path.isdir(Out_dir):  # Create the path if this doesn't exist
    os.mkdir(Out_dir)

b_plotBox = True
b_plotPoincare = True
b_statsPoincare = True
v_strPaths = os.listdir(Data_dir)

m_allPathStd = []
m_allPathsNN50 = []
m_allPathMean = []
v_allPathTaco_Bs = np.array([])
v_allPathTaco_Mt = np.array([])
v_allPathTaco_Af = np.array([])

m_SD1 = []
m_SD2 = []
m_SD12 = []
# v_strPaths = [v_strPaths[14]]
for i_path in range(len(v_strPaths)):
    # if  v_strPaths[i_path] in ['FSFB_0801_HRV.mat']:
    #     continue

    Path_dir = Data_dir + v_strPaths[i_path]
    dict_AllData = scipy.io.loadmat(Path_dir)
    v_ECG = np.array(dict_AllData['v_ECG'][0])
    v_taco = np.array(dict_AllData['v_RRTaco'][0]) * 1000
    v_BPM = np.array(dict_AllData['v_BPM'][0])
    v_time = np.array(dict_AllData['v_TacoTime'][0])
    v_SessionTimes = np.array(dict_AllData['v_SessionTimes'][0])
    v_SessionIndx = [np.where(v_time > v_SessionTimes[0])[0][0], np.where(v_time > v_SessionTimes[1])[0][0]]

    # plt.figure()
    # plt.title(v_strPaths[i_path])
    # plt.plot(v_taco)

    v_taco_Bs = v_taco[:v_SessionIndx[0]]
    v_taco_Mt = v_taco[v_SessionIndx[0]:v_SessionIndx[1]]
    v_taco_Af = v_taco[v_SessionIndx[1]:]

    v_TacoStd = [np.std(v_taco_Bs),
                 np.std(v_taco_Mt),
                 np.std(v_taco_Af)]

    v_TacoMean = [np.mean(v_taco_Bs),
                  np.mean(v_taco_Mt),
                  np.mean(v_taco_Af)]

    v_TacoNN5O = [(np.sum(np.abs(np.diff(v_taco_Bs)) > 50) / (len(v_taco_Bs) - 1)) * 100,
                  (np.sum(np.abs(np.diff(v_taco_Mt)) > 50) / (len(v_taco_Mt) - 1)) * 100,
                  (np.sum(np.abs(np.diff(v_taco_Af)) > 50) / (len(v_taco_Af) - 1)) * 100]

    m_allPathsNN50.append(v_TacoNN5O)
    m_allPathStd.append(v_TacoStd)
    m_allPathMean.append(v_TacoMean)

    v_taco = (v_taco - np.mean(v_taco[:v_SessionIndx[0]])) / np.std(v_taco[:v_SessionIndx[0]])
    v_taco_Bs = v_taco[:v_SessionIndx[0]]
    v_taco_Mt = v_taco[v_SessionIndx[0]:v_SessionIndx[1]]
    v_taco_Af = v_taco[v_SessionIndx[1]:]

    v_tacoAll = [v_taco_Bs, v_taco_Mt, v_taco_Af]
    # PlotPoincareComparations(v_tacoAll)
    v_SD1 = []
    v_SD2 = []
    v_SD12 = []

    for i_taco in v_tacoAll:
        d_SD1, d_SD2 = poincare_parameters(i_taco)
        v_SD1.append(d_SD1)
        v_SD2.append(d_SD2)
        v_SD12.append(d_SD1 / d_SD2)

    m_SD1.append(v_SD1)
    m_SD2.append(v_SD2)
    m_SD12.append(v_SD12)

    v_allPathTaco_Bs = np.concatenate((v_allPathTaco_Bs, v_taco_Bs))
    v_allPathTaco_Mt = np.concatenate((v_allPathTaco_Mt, v_taco_Mt))
    v_allPathTaco_Af = np.concatenate((v_allPathTaco_Af, v_taco_Af))

m_allPathsNN50 = np.array(m_allPathsNN50)
m_allPathStd = np.array(m_allPathStd)
m_allPathMean = np.array(m_allPathMean)

##
v_tacoAll = [v_allPathTaco_Bs, v_allPathTaco_Mt, v_allPathTaco_Af]

for i_taco in v_tacoAll:
    sd1, sd2 = poincare_parameters(i_taco)
    print(f' Ratio SD1 = {sd1}')
    print(f' Ratio SD2 = {sd2}')
    print(f' Ratio SD1/SD2 = {sd1 / sd2}')

if b_plotPoincare:
    PlotPoincareComparisons(v_tacoAll)
    Out_dirBoxPoin = Out_dir + 'Poincare/'
    if not os.path.isdir(Out_dirBoxPoin):  # Create the path if this doesn't exist
        os.mkdir(Out_dirBoxPoin)
    str_outPoin = f'AllPaths_Poincare.svg'
    plt.savefig(Out_dirBoxPoin + str_outPoin, transparent=True)
    plt.close()

if b_statsPoincare:
    m_SD1 = np.array(m_SD1)
    m_SD2 = np.array(m_SD2)
    m_SD12 = np.array(m_SD12)
    #
    a = f_PermTest(m_SD1[:, 1], m_SD1[:, 2])
    b = f_PermTest(m_SD2[:, 1], m_SD2[:, 2])
    c = f_PermTest(m_SD12[:, 1], m_SD12[:, 2])
#
# if b_plotBox:
#     v_FreqBands_Names = ['NN50', 'Std', 'Mean']
#     m_allFrequencies = [m_allPathsNN50, m_allPathStd, m_allPathMean]
#     Out_dirBoxPlot = Out_dir + 'BoxPlot/'
#     if not os.path.isdir(Out_dirBoxPlot):  # Create the path if this doesn't exist
#         os.mkdir(Out_dirBoxPlot)
#     for i_band in range(len(m_allFrequencies)):
#         MyBoxPlot(np.transpose(m_allFrequencies[i_band]),False)
#         str_outBox = f'AllPaths_BoxPlot_{v_FreqBands_Names[i_band]}.svg'
#         plt.savefig(Out_dirBoxPlot + str_outBox, transparent=True)
#         plt.close()
#

#
statsNN50 = f_PermTest(m_allPathsNN50[:, 0], m_allPathsNN50[:, 1])
statsStd = f_PermTest(m_allPathStd[:, 0], m_allPathStd[:, 1])
statsMean = f_PermTest(m_allPathMean[:, 0], m_allPathMean[:, 1])

statsNN50 = f_PermTest(m_allPathsNN50[:, 2], m_allPathsNN50[:, 1])
statsStd = f_PermTest(m_allPathStd[:, 2], m_allPathStd[:, 1])
statsMean = f_PermTest(m_allPathMean[:, 2], m_allPathMean[:, 1])
##
indx1 = 1
indx2 = 2

stats = f_PermTest(m_allPathsNN50[:, indx1], m_allPathsNN50[:, indx2])

print(f'Mean = {np.mean(m_allPathsNN50[:, indx1] - m_allPathsNN50[:, indx2])}')
print(f'std = {np.std(m_allPathsNN50[:, indx1] - m_allPathsNN50[:, indx2])}')
print(f'pvalue = {stats[1][0]}')
