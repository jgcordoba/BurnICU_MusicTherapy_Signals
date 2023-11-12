import matplotlib
import numpy as np
import seaborn as sns

matplotlib.use('TkAgg')  # Set the backend for matplotlib
import matplotlib.pyplot as plt
import os
import scipy.io
import pandas as pd
from Functions.f_AnalysisGraphs import *
from Functions.f_AnalysisFunctions import *
from Functions.f_AnalysisStatistics import *
import pandas as pd
from scipy.interpolate import make_interp_spline

Data_dir = '../Data/EEGFeatures/'
Out_dir = 'Results/FFA/'
if not os.path.isdir(Out_dir):
    os.mkdir(Out_dir)

df_AllInfo = pd.read_csv('./../RawData/data_info.csv', sep=';')
v_strPaths = df_AllInfo['name_path'].tolist()  # List of path names
v_Allpat_SessionTimes = [100, 380]

# Define the names of the frequency bands

v_leftFA_BS = []
v_leftFA_MT = []
v_leftFA_PI = []

v_rightFA_BS = []
v_rightFA_MT = []
v_rightFA_PI = []

for i_path in range(len(v_strPaths)):
    # if v_strPaths[i_path] not in ['FSFB_0101']:
    #      continue
    # Get the filename for the current path
    str_name = v_strPaths[i_path] + '_rawPSD.mat'
    # Check if the file exists in the data directory
    if str_name in os.listdir(Data_dir):
        # Load data from the mat file
        d_AllData = scipy.io.loadmat(Data_dir + str_name)
        m_AbsPSD = d_AllData['AbsPSD']  # Absolute PSD data
        v_ChanNames = d_AllData['v_ChanNames']  # Channel names
        v_SessionTimes = d_AllData['v_SessionTimes'][0]  # Session times [Start, End]
        v_FreqBands = d_AllData['v_FreqBands']  # Frequency bands
        d_leftLead = int(np.where(v_ChanNames == 'Fp1')[0])
        d_rightLead = int(np.where(v_ChanNames == 'Fp2')[0])
        d_alphaBand = 2

        v_leftFA = m_AbsPSD[d_leftLead, d_alphaBand, :]
        v_rightFA = m_AbsPSD[d_rightLead, d_alphaBand, :]

        d_leftFA_BS = np.log(np.mean(v_leftFA[0:v_SessionTimes[0]]))
        d_leftFA_MT = np.log(np.mean(v_leftFA[v_SessionTimes[0]:v_SessionTimes[1]]))
        d_leftFA_PI = np.log(np.mean(v_leftFA[v_SessionTimes[1]:]))

        v_leftFA_BS.append(d_leftFA_BS)
        v_leftFA_MT.append(d_leftFA_MT)
        v_leftFA_PI.append(d_leftFA_PI)

        d_rightFA_BS = np.log(np.mean(v_rightFA[0:v_SessionTimes[0]]))
        d_rightFA_MT = np.log(np.mean(v_rightFA[v_SessionTimes[0]:v_SessionTimes[1]]))
        d_rightFA_PI = np.log(np.mean(v_rightFA[v_SessionTimes[1]:]))

        v_rightFA_BS.append(d_rightFA_BS)
        v_rightFA_MT.append(d_rightFA_MT)
        v_rightFA_PI.append(d_rightFA_PI)
##
# plt.violinplot([v_leftFA_BS,v_rightFA_BS])

v_Data = np.array([v_leftFA_BS, v_rightFA_BS])
# v_Data = np.array([v_leftFA_MT, v_rightFA_MT])
v_Data = np.array([v_leftFA_PI, v_rightFA_PI])
##
# v_Data = v_Data / (v_Data[0]+v_Data[1])
df_Data = pd.DataFrame(np.transpose(v_Data), columns=['x', 'y'])
v_DataPos = np.random.uniform(low=0.9, high=1.1, size=(len(v_leftFA_BS)))

fig, ax = plt.subplots(1, 2, sharey=True)
fig.subplots_adjust(hspace=0, wspace=0)
fig.suptitle('Base line')
[ax[1].spines[i].set_visible(False) for i in ['top', 'right', 'bottom', 'left']]
[ax[0].spines[i].set_visible(False) for i in ['top', 'right', 'bottom', 'left']]

ax[0].plot(v_DataPos, v_Data[0], 'o')
ax[0].plot(v_DataPos + 1, v_Data[1], 'o')
ax[0].plot([v_DataPos, v_DataPos + 1], v_Data, color='grey', linewidth=1)
ax[0].set_ylim(-2, 4.5)
ax[0].set_xlim(0.75, 2.2)
ax[0].set_xticks([1, 2])
ax[0].set_xticklabels(['FP1', 'FP2'])

#
df_Data = df_Data.reset_index().melt('index', var_name='cols', value_name='vals')
sns.kdeplot(data=df_Data, ax=ax[1], y="vals", hue='cols', bw_adjust=0.7,  fill=True)
#
ax[1].axis("off")
ax[1].set_xlim(0, 0.6)
##

v_Data1 = np.array(v_leftFA_BS) - np.array(v_rightFA_BS)
v_Data2 = np.array(v_leftFA_MT) - np.array(v_rightFA_MT)
v_Data3 = np.array(v_leftFA_PI) - np.array(v_rightFA_PI)

a = f_PermTest(v_Data1,v_Data2, ps_PermNum=1600)
b = f_PermTest(v_Data2,v_Data3, ps_PermNum=1600)
c = f_PermTest(v_Data1,v_Data3, ps_PermNum=1600)


v_allData = [v_Data1,v_Data2,v_Data3]

plt.figure()
plt.boxplot(v_allData)