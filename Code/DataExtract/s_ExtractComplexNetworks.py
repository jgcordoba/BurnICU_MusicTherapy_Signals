
import os
import scipy.io
from Functions.f_ComplexNetworks import *
import pandas as pd

Data_dir = './../Data/EEGFeatures/'
Out_dir = './../Data/ComplexNetworks/'
if not os.path.isdir(Out_dir):  # Create the path if this doesn't exist
    os.mkdir(Out_dir)

df_AllInfo = pd.read_csv('./../RawData/data_info.csv', sep=';')
v_strPaths = df_AllInfo['name_path'].tolist()  # Path Names
v_FreqBands_Names = ['Delta', 'Theta', 'Alpha', 'Slow Beta', 'Fast Beta']  # Names of the frequency bands
s_windStep = 1.5

s_corrWindSize = 30
s_corrWindStep = 15
dict_AllFreqComplexNetworks = {}
for i_path in range(len(v_strPaths)):

    str_name = v_strPaths[i_path] + '_PSD.mat'

    if str_name in os.listdir(Data_dir):
        d_AllData = scipy.io.loadmat(Data_dir + str_name)
        m_AbsPSD = d_AllData['AbsPSD']

        v_ChanNames = d_AllData['v_ChanNames']  # Cahnnel names by order
        v_SessionTimes = d_AllData['v_SessionTimes'][0]  # Times of the MT session [Start, End]
        v_FreqBands = d_AllData['v_FreqBands']
        for i_band in range(len(v_FreqBands)):
            m_complexNetwork = np.array(BrainMatrixCorrelation(m_AbsPSD[:, i_band], 1 / s_windStep, s_corrWindSize, s_corrWindStep))
            dict_AllFreqComplexNetworks[v_FreqBands_Names[i_band]] = m_complexNetwork

        dict_AllFreqComplexNetworks['WindSize_sec'] = s_corrWindSize
        dict_AllFreqComplexNetworks['StepSize_sec'] = s_corrWindStep
        dict_AllFreqComplexNetworks['v_SessionTimes'] = v_SessionTimes*s_windStep
        dict_AllFreqComplexNetworks['v_ChanNames'] = v_ChanNames
        dict_AllFreqComplexNetworks['v_FreqBands'] = v_FreqBands

        str_name = v_strPaths[i_path] + '_ComplexNetwork.mat'
        str_out = Out_dir + str_name
        scipy.io.savemat(str_out, dict_AllFreqComplexNetworks)
        print(f'---------------------------------------------')
        print(f'{str_name} successfully saved in path.')
    else:
        print(f'##################################################')
        print(f'WARNING - File {str_name} doesnt exist')
        print(f'##################################################')
        break
