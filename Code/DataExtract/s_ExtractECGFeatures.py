import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.io
import numpy as np
from Functions.f_ECGAnalysis import *
import scipy as sci


Data_dir = './../Data/MatData/'  # Filtered and removed artifact data
Out_dir = './../Data/ECGFeatures/'  # Out path to save the matrix and important information
if not os.path.isdir(Out_dir):  # Create the path if this doesn't exist
    os.mkdir(Out_dir)
df_AllInfo = pd.read_csv('./../RawData/data_info.csv', sep=';')
v_strPaths = df_AllInfo['name_path'].tolist()  # Path Names
# # v_strPaths = [v_strPaths[4], v_strPaths[9], v_strPaths[11], v_strPaths[14]]
v_strPaths = [v_strPaths[14]]
for i_path in range(len(v_strPaths)):
    str_name = v_strPaths[i_path] + '_HRV'
    Path_dir = Data_dir + 'DataMat_' + v_strPaths[i_path] + '.mat'  # Format name of the pat
    d_AllData = scipy.io.loadmat(Path_dir)
    m_data = d_AllData['m_Data']
    d_sampleRate = d_AllData['d_sampleRate'][0][0]
    v_ChanNames = d_AllData['v_ChanNames']  # Cahnnel names by order
    df_PathInfo = df_AllInfo[df_AllInfo['name_path'] == v_strPaths[i_path]]
    v_SessionTimes = [int(df_PathInfo['start_time']), int(df_PathInfo['end_time'])]
    d_ECGIndex = np.where(v_ChanNames == 'ECG')[0]

    if len(d_ECGIndex) != 0:
        print(f'############################################################')
        print(f'Pat: {i_path + 1}/{len(v_strPaths)} - Processing ECG channel for {v_strPaths[i_path]} ')
        print(f'############################################################')
        d_ECGIndex = d_ECGIndex[0]
        v_ECGData = m_data[d_ECGIndex]  # [:-20 * d_sampleRate]
        v_Time = np.arange(0, len(v_ECGData)) / d_sampleRate

        v_ECGDataClean, _, v_peaksClean, m_timeRemoveLimits = cleanECGbyTachogram(v_ECGData, d_sampleRate, 1)
        v_ECGTimeClean = np.arange(0, len(v_ECGDataClean)) / d_sampleRate

        if len(m_timeRemoveLimits) > 0:
            if not all((m_timeRemoveLimits[0] < v_SessionTimes[0]) == (m_timeRemoveLimits[1] < v_SessionTimes[0])):
                v_SessionTimes[0] = v_SessionTimes[0] - np.sum((m_timeRemoveLimits[1] - m_timeRemoveLimits[0])
                                                               [m_timeRemoveLimits[1] < v_SessionTimes[0]]) \
                                    + (v_SessionTimes[0] - m_timeRemoveLimits[0][m_timeRemoveLimits[0] < v_SessionTimes[0]][0])

            # m_timeRemoveLimits[1][m_timeRemoveLimits[1] > v_SessionTimes[1]][0]
            else:
                v_SessionTimes[0] = v_SessionTimes[0] - np.sum(
                    (m_timeRemoveLimits[1] - m_timeRemoveLimits[0])[m_timeRemoveLimits[1] < v_SessionTimes[0]])

            if not all((m_timeRemoveLimits[0] < v_SessionTimes[1]) == (m_timeRemoveLimits[1] < v_SessionTimes[1])):
                print(v_SessionTimes[1])
                v_SessionTimes[1] = v_SessionTimes[1] - (np.sum((m_timeRemoveLimits[1] - m_timeRemoveLimits[0])[m_timeRemoveLimits[1] < v_SessionTimes[1]]) + (v_SessionTimes[1] - m_timeRemoveLimits[0][m_timeRemoveLimits[0] < v_SessionTimes[1]][0]))
                print(v_SessionTimes[1])
            # m_timeRemoveLimits[1][m_timeRemoveLimits[1] > v_SessionTimes[1]][0]
            else:
                v_SessionTimes[1] = v_SessionTimes[1] - np.sum(
                    (m_timeRemoveLimits[1] - m_timeRemoveLimits[0])[m_timeRemoveLimits[1] < v_SessionTimes[1]])

        v_RRTaco = np.diff(v_peaksClean) / d_sampleRate
        v_RRTime = v_ECGTimeClean[v_peaksClean[1::]]

        v_RRTacoClean, v_RRTimeClean = cleanTachogram(v_RRTaco, v_RRTime, d_adjuster=1.15)

        f = sci.interpolate.CubicSpline(v_RRTimeClean, v_RRTacoClean, bc_type='natural')
        v_RRTacoInter = f(v_RRTime)

        # fig, ax = plt.subplots(2, 1, sharex=True)
        # ax[0].plot(v_ECGTimeClean, v_ECGDataClean, 'k')
        # ax[0].plot(v_ECGTimeClean[v_peaksClean], v_ECGDataClean[v_peaksClean], '.r')
        # ax[1].plot(v_RRTime, v_RRTaco, 'gray')
        # ax[1].plot(v_RRTime, v_RRTacoInter, 'k')


        v_RRTacoInter[v_RRTacoInter > 1.5] = 1.5
        v_RRTacoInter[v_RRTacoInter < 0.4] = 0.4

        v_RRTaco_Bs = v_RRTacoInter[v_RRTime < v_SessionTimes[0]]
        v_RRTime_Bs = np.array([sum(v_RRTaco_Bs[:i + 1]) for i in range(len(v_RRTaco_Bs))])
        v_SessionTimes[0] = v_RRTime_Bs[-1]

        v_RRTaco_Mt = v_RRTacoInter[
            [np.array([v_RRTime > v_SessionTimes[0]]) * np.array([v_RRTime < v_SessionTimes[1]])][0][0]]
        v_RRTime_Mt = v_RRTime[
            [np.array([v_RRTime > v_SessionTimes[0]]) * np.array([v_RRTime < v_SessionTimes[1]])][0][0]]
        v_RRTime_Mt = np.array([sum(v_RRTaco_Mt[:i + 1]) for i in range(len(v_RRTaco_Mt))])
        v_SessionTimes[1] = v_RRTime_Mt[-1] + v_SessionTimes[0]

        v_RRTaco_Af = v_RRTacoInter[v_RRTime > v_SessionTimes[1]]
        v_RRTime_Af = v_RRTime[v_RRTime > v_SessionTimes[1]]
        v_RRTime_Af = np.array([sum(v_RRTaco_Af[:i + 1]) for i in range(len(v_RRTaco_Af))])

        d_BslWind = int((5 * 60))
        d_MtlWind = int((15 * 60))
        d_AflWind = int((5 * 60))

        if v_RRTime_Bs[-1] > d_BslWind:
            v_RRTacoWind_Bs = v_RRTaco_Bs[np.array(v_RRTime_Bs < (v_RRTime_Bs[-1] / 2) + d_BslWind / 2) * np.array(
                v_RRTime_Bs > (v_RRTime_Bs[-1] / 2) - d_BslWind / 2)]
            v_RRTimeWind_Bs = np.array([sum(v_RRTacoWind_Bs[:i + 1]) for i in range(len(v_RRTacoWind_Bs))])
            v_SessionTimes[0] = v_RRTimeWind_Bs[-1]
        else:
            v_RRTacoWind_Bs = v_RRTaco_Bs
            v_RRTimeWind_Bs = v_RRTime_Bs

        if v_RRTime_Mt[-1] > d_MtlWind:
            v_RRTacoWind_Mt = v_RRTaco_Mt[np.array(v_RRTime_Mt < (v_RRTime_Mt[-1] / 2) + d_MtlWind / 2) * np.array(
                v_RRTime_Mt > (v_RRTime_Mt[-1] / 2) - d_MtlWind / 2)]
            v_RRTimeWind_Mt = np.array(
                [sum(v_RRTacoWind_Mt[:i + 1]) + v_SessionTimes[0] for i in range(len(v_RRTacoWind_Mt))])
            v_SessionTimes[1] = v_RRTimeWind_Mt[-1]
        else:
            v_RRTacoWind_Mt = v_RRTaco_Mt
            v_RRTimeWind_Mt = np.array(
                [sum(v_RRTacoWind_Mt[:i + 1]) + v_SessionTimes[0] for i in range(len(v_RRTacoWind_Mt))])
            v_SessionTimes[1] = v_RRTimeWind_Mt[-1]

        if v_RRTime_Af[-1] > d_AflWind:
            v_RRTacoWind_Af = v_RRTaco_Af[np.array(v_RRTime_Af < (v_RRTime_Af[-1] / 2) + d_AflWind / 2) * np.array(
                v_RRTime_Af > (v_RRTime_Af[-1] / 2) - d_AflWind / 2)]
            v_RRTimeWind_Af = np.array(
                [sum(v_RRTacoWind_Af[:i + 1]) + v_RRTimeWind_Mt[-1] for i in range(len(v_RRTacoWind_Af))])
        else:
            v_RRTacoWind_Af = v_RRTaco_Af
            v_RRTimeWind_Af = np.array(
                [sum(v_RRTacoWind_Af[:i + 1]) + v_RRTimeWind_Mt[-1] for i in range(len(v_RRTacoWind_Af))])

        v_RRTacoClean = np.concatenate((v_RRTacoWind_Bs, v_RRTacoWind_Mt, v_RRTacoWind_Af))
        v_RRTimeClean = np.concatenate((v_RRTimeWind_Bs, v_RRTimeWind_Mt, v_RRTimeWind_Af))
        v_BPM = 60 / v_RRTacoClean

        # plt.figure()
        # plt.plot(v_RRTimeClean,v_RRTacoClean)
        scipy.io.savemat(Out_dir + str_name + '.mat', mdict={
            'v_ECG': v_ECGDataClean,
            'v_BPM': v_BPM,
            'v_TacoTime': v_RRTimeClean,
            'v_RRTaco': v_RRTacoClean,
            'v_SessionTimes': v_SessionTimes,
            'd_sRate': d_sampleRate})

    else:
        print(f'############################################################')
        print(f'Pat: {i_path + 1}/{len(v_strPaths)} - WARNING! --- ECG channel not found in {v_strPaths[i_path]} ')
        print(f'############################################################')
