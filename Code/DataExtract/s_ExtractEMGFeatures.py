import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import mne
from Functions.f_TRC_Reader import *
from Functions.f_SignalProcFuncLibs import *
from Functions.f_AnalysisFunctions import *
import pandas as pd

str_DataPath = './../RawData/'
str_OutPath = './../Data/EMGFeatures/'
if not os.path.isdir(str_OutPath):  # Create the path if this doesn't exist
    os.mkdir(str_OutPath)

df_AllInfo = pd.read_csv('./../RawData/data_info.csv', sep=';')
v_PathNames = df_AllInfo['name_path'].tolist()  # Path Names
# v_PathNames = [v_PathNames[4]]
for i_path in range(len(v_PathNames)):

    print(f'############################################################')
    print(f'Processing path: {v_PathNames[i_path]} - {i_path + 1}/{len(v_PathNames)}')
    print(f'------------------------------------------------------------')

    str_ReadName = str_DataPath + v_PathNames[i_path] + '.TRC'
    str_SaveName = v_PathNames[i_path] + '_PSD'

    df_PathInfo = df_AllInfo[df_AllInfo['name_path'] == v_PathNames[i_path]]
    v_sessionTimes = [int(df_PathInfo['start_time']), int(df_PathInfo['end_time'])]

    if v_PathNames[i_path] in ['FSFB_0101', 'FSFB_0102']:
        str_ChannStr = []
        b_multipleRecs = False

    elif v_PathNames[i_path] == 'FSFB_0202':
        str_ChannStr = ['EMG2+']  # 0202
        b_multipleRecs = False

    elif v_PathNames[i_path] in ['FSFB_0501']:
        str_ChannStr = ['EMG1+']  # 0202
        v_RecNames = ['FSFB_0501_A', 'FSFB_0501_B', 'FSFB_0501_C']
        b_multipleRecs = True

    elif v_PathNames[i_path] in ['FSFB_0502']:
        str_ChannStr = ['EMG1+']  # 0202
        v_RecNames = ['FSFB_0502_A', 'FSFB_0502_B']

        b_multipleRecs = True

    elif v_PathNames[i_path] in ['FSFB_0801']:
        str_ChannStr = ['EMG1+']  # 0202
        v_RecNames = ['FSFB_0801_A', 'FSFB_0801_B']
        b_multipleRecs = True

    else:
        str_ChannStr = ['EMG1+']  # 0202
        b_multipleRecs = False

    m_AllData = []

    if len(str_ChannStr) != 0:

        if b_multipleRecs:
            for i_str in v_RecNames:
                if len(m_AllData) == 0:
                    str_ReadName = str_DataPath + i_str + '.TRC'
                    st_TRCHead = f_GetTRCHeader(str_ReadName)  # Function that extracts the header of the TCR
                    d_SampleRate = st_TRCHead['RecFreq']  # Sample frequency
                    # print(f'Sampling rate: {d_SampleRate}')

                    m_AllData = f_GetSignalsTRC(str_ReadName,
                                                str_ChannStr)  # Function that extracts the data from the TRC file
                else:
                    str_ReadName = str_DataPath + i_str + '.TRC'
                    m_NewData = f_GetSignalsTRC(str_ReadName, str_ChannStr)
                    m_AllData = np.concatenate((m_AllData, m_NewData), 1)
        else:

            if len(m_AllData) == 0:
                str_ReadName = str_DataPath + v_PathNames[i_path] + '.TRC'
                st_TRCHead = f_GetTRCHeader(str_ReadName)  # Function that extracts the header of the TCR
                d_SampleRate = st_TRCHead['RecFreq']  # Sample frequency
                print(f'Sampling rate: {d_SampleRate}')

                m_AllData = f_GetSignalsTRC(str_ReadName,
                                            str_ChannStr)  # Function that extracts the data from the TRC file

        if d_SampleRate < 500:
            print(f'------------------------------------------------------------')
            print(
                f'sample frequency {d_SampleRate} under 512Hz in : {v_PathNames[i_path]} - {i_path + 1}/{len(v_PathNames)}')
            print(f'############################################################')
            continue
        v_time = np.arange(0, len(m_AllData[0])) / d_SampleRate
        v_data = m_AllData[0]

        st_Filt = f_GetIIRFilter(d_SampleRate, [20, 200], [19, 201])
        v_dataFilt = f_IIRBiFilter(st_Filt, v_data)

        for i_freq in [60,120]:
            sos = signal.butter(5, [i_freq - 1, i_freq + 1], 'bandstop', fs=d_SampleRate, output='sos')
            v_dataFilt = signal.sosfilt(sos, v_dataFilt)

        d_windSec = 3
        d_windSize = d_SampleRate * d_windSec
        v_meanEMG = []
        for i_wind in range(int(len(v_dataFilt) / d_windSize)):
            v_windEMG = v_dataFilt[i_wind * d_windSize:(i_wind + 1) * d_windSize]
            d_mean = np.mean(np.abs(v_windEMG))
            v_meanEMG.append(d_mean)

        v_meanEMG = np.array(v_meanEMG)
        v_TimeMeans = np.arange(len(v_meanEMG)) * d_windSec

        d_ampThreshold = np.mean(v_meanEMG) + np.std(v_meanEMG) * 3
        v_rejectIndx = v_TimeMeans[v_meanEMG > d_ampThreshold]
        d_artCount_Bs = len(v_rejectIndx[v_rejectIndx <= v_sessionTimes[0]])
        v_sessionTimes[0] = v_sessionTimes[0] - (d_artCount_Bs * d_windSec)
        d_artCount_Mt = len(v_rejectIndx[v_rejectIndx <= v_sessionTimes[1]])
        v_sessionTimes[1] = v_sessionTimes[1] - (d_artCount_Mt * d_windSec)

        v_dataClean = v_dataFilt
        for i_index in range(len(v_rejectIndx)):
            v_dataClean = np.delete(v_dataClean, np.arange(int(v_rejectIndx[i_index] * d_SampleRate),
                                                           int(v_rejectIndx[i_index] * d_SampleRate)
                                                           + d_windSize))
            v_rejectIndx = v_rejectIndx - d_windSec

        ##

        v_dataClean_Bs = v_dataClean[:int(v_sessionTimes[0] * d_SampleRate)]
        v_dataClean_Mt = v_dataClean[int(v_sessionTimes[0] * d_SampleRate):int(v_sessionTimes[1] * d_SampleRate)]
        v_dataClean_Af = v_dataClean[int(v_sessionTimes[1] * d_SampleRate):]

        d_BslWind = int((5 * 60) * d_SampleRate)
        d_MtlWind = int((15 * 60) * d_SampleRate)
        d_AflWind = int((5 * 60) * d_SampleRate)

        if d_BslWind < len(v_dataClean_Bs):
            d_lenMid = int(len(v_dataClean_Bs) / 2)
            v_dataClean_Bs = v_dataClean_Bs[int(d_lenMid - d_BslWind / 2):int(d_lenMid + d_BslWind / 2)]
        v_sessionTimes[0] = int(len(v_dataClean_Bs) / d_SampleRate)

        if d_MtlWind < len(v_dataClean_Mt):
            d_lenMid = int(len(v_dataClean_Mt) / 2)
            v_dataClean_Mt = v_dataClean_Mt[int(d_lenMid - d_MtlWind / 2):int(d_lenMid + d_MtlWind / 2)]
        v_sessionTimes[1] = int(len(v_dataClean_Mt) / d_SampleRate) + v_sessionTimes[0]

        if d_AflWind < len(v_dataClean_Af):
            d_lenMid = int(len(v_dataClean_Af) / 2)
            v_dataClean_Af = v_dataClean_Af[int(d_lenMid - d_AflWind / 2):int(d_lenMid + d_AflWind / 2)]

        v_dataClean = np.concatenate((v_dataClean_Bs, v_dataClean_Mt, v_dataClean_Af))

        v_dataNomr = v_dataClean / np.mean(np.abs(v_dataClean[:int(v_sessionTimes[0] * d_SampleRate)]))
        v_TimeClean = np.arange(len(v_dataClean)) / d_SampleRate

        d_windSec = 2
        d_stepSec = 1
        d_windSize = d_windSec * d_SampleRate
        d_stepSize = d_stepSec * d_SampleRate

        m_psd = []

        for i_wind in range(int(len(v_dataNomr) / d_stepSize)-1):
            v_wind = v_dataNomr[int(i_wind * d_stepSize):int(i_wind * d_stepSize) + d_windSize]
            v_freqs, v_psd = signal.welch(v_wind, d_SampleRate, nperseg=len(v_wind)/3)
            # v_psd = AverageMean(v_psd,3)

            m_psd.append(v_psd)

        m_psd = np.array(m_psd)

        sio.savemat(str_OutPath + str_SaveName + '.mat', mdict={'v_Data': v_dataNomr,
                                                                'v_freqs': v_freqs,
                                                                'm_Psd': m_psd,
                                                                's_Freq': d_SampleRate,
                                                                'd_windSec': d_windSec,
                                                                'd_stepSec': d_stepSec,
                                                                'SessionTimes': v_sessionTimes})
        print(f'---------------------------------------------')
        print(f'{str_SaveName} successfully saved in path.')
        print(f'---------------------------------------------')

    else:
        print(f'------------------------------------------------------------')
        print(f'Not EMG channel in : {v_PathNames[i_path]} - {i_path + 1}/{len(v_PathNames)}')
        print(f'############################################################')
