import numpy as np
from scipy.stats import pearsonr
from Functions.f_SignalProcFuncLibs import *
import networkx as nx


def fullFillConnectivity(m_measureEvolution, v_SessionTimes, d_stepSize ):

    d_maxBs = int(5 * 60 / 15)
    d_maxMt = int(15 * 60 / 15)
    d_maxAf = int(5 * 60 / 15) - 1

    m_measure_Bs = m_measureEvolution[:, 0:int(v_SessionTimes[0] / d_stepSize)]
    m_measure_Mt = m_measureEvolution[:, int(v_SessionTimes[0] / d_stepSize):int(v_SessionTimes[1] / d_stepSize)]
    m_measure_Af = m_measureEvolution[:, int(v_SessionTimes[1] / d_stepSize)::]

    m_measureFull_Bs = np.zeros((len(m_measureEvolution), d_maxBs))
    m_measureFull_Mt = np.zeros((len(m_measureEvolution), d_maxMt))
    m_measureFull_Af = np.zeros((len(m_measureEvolution), d_maxAf))

    d_med = int((d_maxBs / 2))
    d_pos = int(len(m_measure_Bs[0]) / 2)
    m_measureFull_Bs[:, d_med - d_pos:d_med + d_pos] = m_measure_Bs

    d_med = int((d_maxMt / 2))
    d_pos = int(len(m_measure_Mt[0]) / 2)
    m_measureFull_Mt[:, d_med - d_pos:d_med + d_pos] = m_measure_Mt

    d_med = int((d_maxAf / 2))
    d_pos = (len(m_measure_Af[0]) / 2) + 0.1

    m_measureFull_Af[:, d_med - int(d_pos):d_med + round(d_pos)] = m_measure_Af

    m_measureFull = np.concatenate((m_measureFull_Bs, m_measureFull_Mt, m_measureFull_Af), 1)
    return(m_measureFull)


def BrainMatrixCorrelation(m_PSDEvolution, d_sRate, d_windSize, d_windStep):
    d_windSam = int(d_windSize * d_sRate)
    d_stepSam = int(d_windStep * d_sRate)
    d_channNumber = len(m_PSDEvolution)
    m_AllchannCorrelation = []

    d_indexStart = 0
    d_indexEnd = d_windSam
    complexMatrixNetworks = []
    while d_indexEnd <= len(m_PSDEvolution[0]):
        m_WindPSDEvolution = m_PSDEvolution[:, d_indexStart:d_indexEnd]
        m_WindChannCorrelation = np.zeros([d_channNumber, d_channNumber])
        for i_chan1 in range(d_channNumber):
            v_FreqEvolution1 = m_WindPSDEvolution[i_chan1]
            for i_chan2 in range(d_channNumber):
                v_FreqEvolution2 = m_WindPSDEvolution[i_chan2]
                if i_chan1 != i_chan2:
                    m_WindChannCorrelation[i_chan1][i_chan2] = np.abs(pearsonr(v_FreqEvolution1, v_FreqEvolution2)[0])

        d_indexStart = int(d_indexStart + d_stepSam)
        d_indexEnd = int(d_indexStart + d_windSam)
        complexMatrixNetworks.append(m_WindChannCorrelation)
    return (complexMatrixNetworks)


def f_ShortestPathLength(brain_matrix):
    G = nx.from_numpy_matrix(brain_matrix)

    # Define the edge weights as the values in the brain connectivity matrix
    edge_weights = {(i, j): 1 - brain_matrix[i][j] for i, j in G.edges()}

    # Set the edge weights in the graph
    nx.set_edge_attributes(G, values=edge_weights, name='weight')

    # Calculate the shortest path length between all pairs of nodes in the graph
    shortest_paths = dict(nx.shortest_path_length(G, weight='weight'))

    # Print the shortest path length between node 0 and all other nodes

    m_lengthPathMap = np.zeros((len(brain_matrix), len(brain_matrix)))
    for node1 in range(len(brain_matrix)):
        for node2, path_length in shortest_paths[node1].items():
            m_lengthPathMap[node1][node2] = path_length
    return (m_lengthPathMap)

def MatrixCorrelation(m_AllFreqEvolution, d_sRate, d_windSize, d_windStep):
    d_windSam = int(d_windSize * d_sRate)
    d_stepSam = int(d_windStep * d_sRate)

    m_AllchannCorrelation = []

    for i_band in range(len(m_AllFreqEvolution[0])):

        m_channCorrelation = np.zeros([len(m_AllFreqEvolution), len(m_AllFreqEvolution)])
        for i_chan1 in range(len(m_AllFreqEvolution)):
            v_FreqEvolution1 = m_AllFreqEvolution[i_chan1][i_band]

            for i_chan2 in range(len(m_AllFreqEvolution)):
                v_FreqEvolution2 = m_AllFreqEvolution[i_chan2][i_band]

                if i_chan1 != i_chan2:

                    d_indexStart = 0
                    d_indexEnd = int(d_indexStart + d_windSam)
                    v_features = []
                    d_count = 0
                    while d_indexEnd <= len(v_FreqEvolution1):
                        i_dataWind1 = v_FreqEvolution1[d_indexStart:d_indexEnd - 1]
                        i_dataWind2 = v_FreqEvolution2[d_indexStart:d_indexEnd - 1]

                        v_features.append(np.abs(pearsonr(i_dataWind1, i_dataWind2)[0]))

                        d_count += 1
                        d_indexStart = int(d_indexStart + d_stepSam)
                        d_indexEnd = int(d_indexStart + d_windSam)

                    d_featuresMean = np.mean(v_features)

                else:
                    d_featuresMean = 0
                m_channCorrelation[i_chan1, i_chan2] = d_featuresMean
        m_AllchannCorrelation.append(m_channCorrelation)

    return m_AllchannCorrelation
a = np.random.rand(50)
b = np.random.rand(50)
pearsonr(a, b)[1]