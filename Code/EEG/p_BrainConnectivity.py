import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import os
import scipy.io as sciio
import numpy as np
from Functions.f_AnalysisFunctions import *
from Functions.f_AnalysisGraphs import *
from Functions.f_ComplexNetworks import *
import matplotlib.pylab as pl
import matplotlib
matplotlib.use('TkAgg')  # Set the backend for matplotlib

Data_dir = './../Data/ComplexNetworks/'
Out_dir = './../EEG/Results/BrainConnectivity/'

b_plotChangeMatrix = True
b_plotComplexNetworks = False
b_plotConnectivityMeasures = False
b_permTest = False

if not os.path.isdir(Out_dir):  # Create the path if this doesn't exist
    os.mkdir(Out_dir)

df_AllInfo = pd.read_csv('./../RawData/data_info.csv', sep=';')
v_strPaths = df_AllInfo['name_path'].tolist()  # Path Names
# v_strPaths = ['FSFB_0101']  # Path Names

v_FreqBands_Names = ['Delta', 'Theta', 'Alpha', 'Slow Beta', 'Fast Beta']  # Names of the frequency bands

s_windStep = 1.5
d_windSize = 30
d_windStep = 15

m_bandConnMatrix_MtvsBs = []
m_bandConnMatrix_AfvsBs = []
m_bandpValues_Af = []
m_bandScores_Af = []
m_bandpValues_Mt = []
m_bandScores_Mt = []

v_pathLenPvalue_Mt = []
v_clusterPvalue_Mt = []
v_pathLenPvalue_Af = []
v_clusterPvalue_Af = []

for i_band in range(len(v_FreqBands_Names)):
    m_allConnMatrix_Bs = []
    m_allConnMatrix_Mt = []
    m_allConnMatrix_Af = []

    m_allStrengthFull = []
    m_allPathLenFull = []
    m_allClusterCoefficientFull = []
    for i_path in range(len(v_strPaths)):
        str_name = v_strPaths[i_path] + '_ComplexNetwork.mat'
        if str_name in os.listdir(Data_dir):
            dict_AllData = sciio.loadmat(Data_dir + str_name)
            v_ChanNames = dict_AllData['v_ChanNames']  # Cahnnel names by order
            v_SessionTimes = dict_AllData['v_SessionTimes'][0]  # Times of the MT session [Start, End]
            v_FreqBands = dict_AllData['v_FreqBands']
            d_stepSize = dict_AllData['StepSize_sec']

            m_connMatrix = dict_AllData[v_FreqBands_Names[i_band]]
            m_strengthEvolution = []
            m_PathLenEvolution = []
            m_clusterCoefficient = []
            for i_wind in range(len(m_connMatrix)):
                m_windConnMatrix = m_connMatrix[i_wind]
                # m_windConnMatrix[m_windConnMatrix <= 0.5] = 0
                # m_windConnMatrix[m_windConnMatrix >= 0.5] = 1
                m_strengthEvolution.append(np.sum(m_windConnMatrix, 0) / 7)
                m_PathLenEvolution.append((np.sum(f_ShortestPathLength(m_windConnMatrix), 0)) / 7)
                m_clusterCoefficient.append(nx.average_clustering(nx.from_numpy_array(m_windConnMatrix),
                                                                  weight='weight'))

            m_strengthEvolution = np.transpose(m_strengthEvolution)
            m_PathLenEvolution = np.transpose(m_PathLenEvolution)
            m_clusterCoefficient = np.array(m_clusterCoefficient)

            d_stepSize = 15

            m_strengthFull = fullFillConnectivity(m_strengthEvolution, v_SessionTimes, d_stepSize)
            m_pathLenFull = fullFillConnectivity(m_PathLenEvolution, v_SessionTimes, d_stepSize)
            m_clusterCoefficientFull = OneDimFullFillConnectivity(m_clusterCoefficient, v_SessionTimes, d_stepSize)

            m_allStrengthFull.append(np.mean(m_strengthFull, 0))
            m_allPathLenFull.append(np.mean(m_pathLenFull, 0))
            m_allClusterCoefficientFull.append(m_clusterCoefficientFull)

            m_connMatrix_bs = np.mean(m_connMatrix[0:int(v_SessionTimes[0] / d_windStep) - 1], 0)
            m_connMatrix_Mt = np.mean(
                m_connMatrix[int(v_SessionTimes[0] / d_windStep):int(v_SessionTimes[1] / d_windStep) - 1], 0)
            m_connMatrix_Af = np.mean(m_connMatrix[int(v_SessionTimes[1] / d_windStep):], 0)

            m_allConnMatrix_Bs.append(m_connMatrix_bs)
            m_allConnMatrix_Mt.append(m_connMatrix_Mt)
            m_allConnMatrix_Af.append(m_connMatrix_Af)

        else:
            print(f'##################################################')
            print(f'WARNING - File {str_name} doesnt exist')
            print(f'##################################################')
            break

    # TODO Eval stadistics for:
    # Strength in matrix for each electrode combination
    # Cluster and lenpath for each band
    m_meanAllConnMatrix_Bs = np.mean(m_allConnMatrix_Bs, 0)
    m_meanAllConnMatrix_Mt = np.mean(m_allConnMatrix_Mt, 0)
    m_meanAllConnMatrix_Af = np.mean(m_allConnMatrix_Af, 0)

    if b_plotConnectivityMeasures:

        # m_allStrengthFull = np.array(m_allStrengthFull)
        # m_allStrengthFull[m_allStrengthFull == 0] = np.nan
        # m_allStrengthFullMean = np.nanmean(m_allStrengthFull, 0)
        # m_allStrengthFullStd = np.nanstd(m_allStrengthFull, 0) / np.sqrt((len(v_strPaths)))

        m_allPathLenFull = np.array(m_allPathLenFull)
        m_allPathLenFull[m_allPathLenFull == 0] = np.nan
        m_allPathLenFullMean = np.nanmean(m_allPathLenFull, 0)
        m_allPathLenFullStd = np.nanstd(m_allPathLenFull, 0) / np.sqrt((len(v_strPaths)))

        m_allClusterCoefficientFull = np.array(m_allClusterCoefficientFull)
        m_allClusterCoefficientFull[m_allClusterCoefficientFull == 0] = np.nan
        m_allClusterCoefficientFullMean = np.nanmean(m_allClusterCoefficientFull, 0)
        m_allClusterCoefficientFullStd = np.nanstd(m_allClusterCoefficientFull, 0) / np.sqrt((len(v_strPaths)))

        d_SoftWind = 3
        # m_allStrengthFullMean = AverageMean(m_allStrengthFullMean, d_SoftWind)
        m_allPathLenFullMean = f_averageMean(m_allPathLenFullMean, d_SoftWind)
        m_allClusterCoefficientFullMean = f_averageMean(m_allClusterCoefficientFullMean, d_SoftWind)

        # m_allStrengthFullStd = AverageMean(m_allStrengthFullStd, d_SoftWind)
        m_allPathLenFullStd = f_averageMean(m_allPathLenFullStd, d_SoftWind)
        m_allClusterCoefficientFullStd = f_averageMean(m_allClusterCoefficientFullStd, d_SoftWind)

        fig, ax = plt.subplots(figsize=(13, 5))
        colors = pl.cm.Reds(np.linspace(0, 1, 8))

        v_time = np.arange(len(m_allPathLenFullStd)) * d_stepSize
        # plt.plot(v_time, m_allStrengthFullMean, color=colors[5])
        # plt.fill_between(v_time, m_allStrengthFullMean - m_allStrengthFullStd,
        #                  m_allStrengthFullMean + m_allStrengthFullStd, color=colors[5], alpha=0.25)
        [ax.spines[i].set_visible(False) for i in ['top', 'right', 'bottom', 'left']]
        ax.plot(v_time, m_allPathLenFullMean, color=colors[5])
        ax.fill_between(v_time, m_allPathLenFullMean - m_allPathLenFullStd, m_allPathLenFullMean + m_allPathLenFullStd,
                         color=colors[4], alpha=0.45)
        ax.plot(v_time, m_allClusterCoefficientFullMean, color=colors[7])
        ax.fill_between(v_time, m_allClusterCoefficientFullMean - m_allClusterCoefficientFullStd,
                         m_allClusterCoefficientFullMean + m_allClusterCoefficientFullStd, color=colors[7], alpha=0.15)

        ax.yaxis.set_tick_params(labelsize=18)
        ax.xaxis.set_tick_params(labelsize=0)

        ax.axvline(v_SessionTimes[0], linewidth=2, linestyle='--', color='k')
        ax.axvline(v_SessionTimes[1], linewidth=2, linestyle='--', color='k')
        # plt.grid(linewidth=0.75, linestyle='--')

        Out_dirME = Out_dir + 'MeasuresEvolution/'
        if not os.path.isdir(Out_dirME):  # Create the path if this doesn't exist
            os.mkdir(Out_dirME)
        str_out = f'AllPaths_MeasuresEvolution_{v_FreqBands_Names[i_band]}.svg'
        plt.savefig(Out_dirME + str_out,transparent=True)
        plt.close()

    if b_plotComplexNetworks:
        sc = TopoPlotCorrAllpaths(m_meanAllConnMatrix_Bs, m_meanAllConnMatrix_Mt, m_meanAllConnMatrix_Af, 0)
        Out_dirNet = Out_dir + 'ComplexNetworks/'
        if not os.path.isdir(Out_dirNet):  # Create the path if this doesn't exist
            os.mkdir(Out_dirNet)
        str_out = Out_dirNet + f'AllPaths_ComplexNetworks_{v_FreqBands_Names[i_band]}.png'
        sc.screenshot(str_out, print_size=(5, 27), autocrop=False)

    if b_plotChangeMatrix:
        m_meanAllConnMatrix_MtvsBs = m_meanAllConnMatrix_Mt - m_meanAllConnMatrix_Bs
        m_bandConnMatrix_MtvsBs.append(m_meanAllConnMatrix_MtvsBs)
        m_meanAllConnMatrix_AfvsBs = m_meanAllConnMatrix_Af - m_meanAllConnMatrix_Bs
        m_bandConnMatrix_AfvsBs.append(m_meanAllConnMatrix_AfvsBs)

        PlotConnectivityMatrix(m_meanAllConnMatrix_MtvsBs, m_meanAllConnMatrix_AfvsBs, v_ChanNames)
        Out_dirMat = Out_dir + 'MatrixChange/'
        if not os.path.isdir(Out_dirMat):  # Create the path if this doesn't exist
            os.mkdir(Out_dirMat)
        str_outAbs = f'AllPaths_CorrChange_{v_FreqBands_Names[i_band]}.svg'
        plt.savefig(Out_dirMat + str_outAbs,transparent=True)
        plt.close()
    ##

    if b_permTest:
        ps_PermNum = 1600

        m_pScore, m_pValues = permTestCorrelationMatrix(m_allConnMatrix_Bs, m_allConnMatrix_Af, 1600)
        m_bandpValues_Af.append(m_pValues)
        m_bandScores_Af.append(m_pScore)

        m_pScore, m_pValues = permTestCorrelationMatrix(m_allConnMatrix_Bs, m_allConnMatrix_Mt, 1600)
        m_bandpValues_Mt.append(m_pValues)

        v_pValues = m_pValues.flatten()
        v_pValues = np.delete(v_pValues, v_pValues == 0)
        # v_pvalueValuesFDR = fdrcorrection(np.sort(v_pValues), 0.05 * (len(v_pValues) / 2), method='indep')
        m_bandScores_Mt.append(m_pScore)

        #
        m_allPathLenFull = np.array(m_allPathLenFull)
        m_allPathLenFull[m_allPathLenFull == 0] = np.nan
        m_allPathLenMean_Bs = np.nanmean(m_allPathLenFull[:, :int(v_SessionTimes[0] / d_windStep)], 1)
        m_allPathLenMean_Mt = np.nanmean(
            m_allPathLenFull[:, int(v_SessionTimes[0] / d_windStep):int(v_SessionTimes[1] / d_windStep)], 1)
        m_allPathLenMean_Af = np.nanmean(m_allPathLenFull[:, int(v_SessionTimes[1] / d_windStep):], 1)

        m_allClusterCoefficientFull = np.array(m_allClusterCoefficientFull)
        m_allClusterCoefficientFull[m_allClusterCoefficientFull == 0] = np.nan
        m_allClusterCoefficientMean_Bs = np.nanmean(m_allClusterCoefficientFull[:, :int(v_SessionTimes[0] / d_windStep)], 1)
        m_allClusterCoefficientMean_Mt = np.nanmean(
            m_allClusterCoefficientFull[:, int(v_SessionTimes[0] / d_windStep):int(v_SessionTimes[1] / d_windStep)], 1)
        m_allClusterCoefficientMean_Af = np.nanmean(m_allClusterCoefficientFull[:, int(v_SessionTimes[1] / d_windStep):], 1)

        d_hyp, d_pvalue= f_PermTest(m_allPathLenMean_Bs, m_allPathLenMean_Mt, ps_PermNum=ps_PermNum)
        v_pathLenPvalue_Mt.append((d_hyp[0],d_pvalue[0]))
        d_hyp, d_pvalue= f_PermTest(m_allPathLenMean_Bs, m_allPathLenMean_Af, ps_PermNum=ps_PermNum)
        v_pathLenPvalue_Af.append((d_hyp[0],d_pvalue[0]))
        d_hyp, d_pvalue= f_PermTest(m_allClusterCoefficientMean_Bs, m_allClusterCoefficientMean_Mt, ps_PermNum= ps_PermNum)
        v_clusterPvalue_Mt.append((d_hyp[0],d_pvalue[0]))
        d_hyp, d_pvalue= f_PermTest(m_allClusterCoefficientMean_Bs, m_allClusterCoefficientMean_Af, ps_PermNum= ps_PermNum)
        v_clusterPvalue_Af.append((d_hyp[0],d_pvalue[0]))

