import os  # for interacting with the operating system
import scipy.io  # for loading and saving MATLAB files
import pandas as pd  # for working with data in tabular form
from Functions.f_AnalysisFunctions import *  # import custom analysis functions
from Functions.f_AnalysisStatistics import *

# Set flags and directory paths
s_SavePermTest = True  # Flag indicating whether to save the permutation test results
s_PermTest_AllPath = True  # Flag indicating whether to compute permutation tests for all paths
Data_dir = './../Data/EEGFeatures/'  # Directory path for input data
Out_dir = './Results/PermTestPsd/'  # Directory path for output files

# Create output directory if it does not exist
if not os.path.isdir(Out_dir):
    os.mkdir(Out_dir)

# Read the data info file into a pandas dataframe
df_AllInfo = pd.read_csv('./../RawData/data_info.csv', sep=';')

# Get a list of paths to the input data files
v_strPaths = df_AllInfo['name_path'].tolist()

# Define the names of the frequency bands
v_FreqBands_Names = ['Delta', 'Theta', 'Alpha', 'Slow Beta', 'Fast Beta']

# Initialize empty lists for storing computed erences in absolute and relative PSD
m_AbsPSDAllPaths_Bs = []
m_AbsPSDAllPaths_Af = []
m_AbsPSDAllPaths_Mt = []

# Iterate over each path in the list of input data files
for i_path in range(len(v_strPaths)):
    # if v_strPaths[i_path] in ['FSFB_0501','FSFB_0502']:
    #     continue
    str_name = v_strPaths[i_path] + '_PSD.mat'

    # Check if the data file exists in the specified directory
    if str_name in os.listdir(Data_dir):
        print(f'##################################################')
        print(f'Processing file {str_name}')
        print(f'##################################################')

        # Load the data from the MATLAB file
        d_AllData = scipy.io.loadmat(Data_dir + str_name)

        # Extract absolute and relative PSD data
        m_AbsPSD = d_AllData['AbsPSD']  # Absolute PSD data

        # Extract channel names, session times, and frequency bands
        v_ChanNames = d_AllData['v_ChanNames']  # Channel names by order
        v_SessionTimes = d_AllData['v_SessionTimes'][0]  # Times of the MT session [Start, End]
        v_FreqBands = d_AllData['v_FreqBands']  # Frequency bands

        # Compute average changes in absolute PSD and store the results in corresponding lists
        m_Abs_BS, m_Abs_Mt, m_Abs_Af = f_AveragePSD(m_AbsPSD, v_SessionTimes, False)
        m_AbsPSDAllPaths_Bs.append(m_Abs_BS)  # erences between Music Therapy (MT) and baseline (BS)
        m_AbsPSDAllPaths_Mt.append(m_Abs_Mt)  # erences between After Intervetion (AF) and baseline (BS)
        m_AbsPSDAllPaths_Af.append(m_Abs_Af)  # erences between AF and MT

m_AbsPSDAllPaths_Bs = np.array(m_AbsPSDAllPaths_Bs)
m_AbsPSDAllPaths_Mt = np.array(m_AbsPSDAllPaths_Mt)
m_AbsPSDAllPaths_Af = np.array(m_AbsPSDAllPaths_Af)

ps_PermNum = 1600
df_PermTestAbs_MtvsBs = extractPermTestData(m_AbsPSDAllPaths_Mt, m_AbsPSDAllPaths_Bs, v_FreqBands_Names, v_ChanNames,
                                            ps_PermNum)
df_PermTestAbs_AfvsBs = extractPermTestData(m_AbsPSDAllPaths_Af, m_AbsPSDAllPaths_Bs, v_FreqBands_Names, v_ChanNames, ps_PermNum)

df_PermTestAbs_MTvsAf = extractPermTestData(m_AbsPSDAllPaths_Mt, m_AbsPSDAllPaths_Af, v_FreqBands_Names, v_ChanNames, ps_PermNum)

##
ps_PermNum = 1600
chan = 3
band = 4
v_chan1 = m_AbsPSDAllPaths_Mt[:,chan, band]
v_chan2 = m_AbsPSDAllPaths_Bs[:, chan, band]

print(np.mean(v_chan1-v_chan2))
print(np.std(v_chan1-v_chan2))
##
s_hyp, s_pvalues, v_StatsRef = f_PermTest(v_chan1, v_chan2, ps_PermNum=ps_PermNum)

