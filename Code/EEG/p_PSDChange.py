import matplotlib

matplotlib.use('TkAgg')  # Set the backend for matplotlib
import os
import scipy.io
import pandas as pd
from Functions.f_AnalysisGraphs import *
from Functions.f_AnalysisFunctions import *

# Set flags to control the behavior of the script
s_PlotSaveBandEvolution = False
s_PsdEvolutionAllpaths = True

# Set directories for input and output files
Data_dir = '../Data/EEGFeatures/'
Out_dir = 'Results/PSDChange/'

# Create the output directory if it doesn't exist
if not os.path.isdir(Out_dir):
    os.mkdir(Out_dir)
# Read data info from a CSV file
df_AllInfo = pd.read_csv('./../RawData/data_info.csv', sep=';')
v_strPaths = df_AllInfo['name_path'].tolist()  # List of path names
# Define the names of the frequency bands
v_FreqBands_Names = ['Delta', 'Theta', 'Alpha', 'Slow Beta', 'Fast Beta']
# Initialize lists to store data
m_DiffAbsPSDAllPaths_MtvsBs = []
m_DiffAbsPSDAllPaths_AfvsBs = []
m_DiffAbsPSDAllPaths_AfvsMt = []

# Loop over each path
for i_path in range(len(v_strPaths)):
    # if v_strPaths[i_path] in ['FSFB_0501','FSFB_0502']:
    #     continue
    # Get the filename for the current path
    str_name = v_strPaths[i_path] + '_PSD.mat'
    # Check if the file exists in the data directory
    if str_name in os.listdir(Data_dir):
        # Load data from the mat file
        d_AllData = scipy.io.loadmat(Data_dir + str_name)
        m_AbsPSD = d_AllData['AbsPSD']  # Absolute PSD data
        v_ChanNames = d_AllData['v_ChanNames']  # Channel names
        v_SessionTimes = d_AllData['v_SessionTimes'][0]  # Session times [Start, End]
        v_FreqBands = d_AllData['v_FreqBands']  # Frequency bands
        # Calculate the average changes in PSD for the current path
        m_DiffAbsMT_BS, m_DiffAbsAF_BS, m_DiffAbsMT_AF = f_AveragePSD(m_AbsPSD, v_SessionTimes, True)
        # Store the average changes for further analysis
        m_DiffAbsPSDAllPaths_MtvsBs.append(m_DiffAbsMT_BS)
        m_DiffAbsPSDAllPaths_AfvsBs.append(m_DiffAbsAF_BS)
        m_DiffAbsPSDAllPaths_AfvsMt.append(m_DiffAbsMT_AF)
    else:
        # Print a warning if the file doesn't exist
        print(f'##################################################')
        print(f'WARNING - File {str_name} doesn\'t exist')
        print(f'##################################################')
        break
# Calculate the mean differences in PSD across all paths
m_AbsPsdDiff_MTvsBS = np.mean(m_DiffAbsPSDAllPaths_MtvsBs, 0)
m_AbsPsdDiff_AfvsBs = np.mean(m_DiffAbsPSDAllPaths_AfvsBs, 0)
m_AbsPsdDiff_AfvsMt = np.mean(m_DiffAbsPSDAllPaths_AfvsMt, 0)
m_AbsPsdDiff = np.array([m_AbsPsdDiff_MTvsBS, m_AbsPsdDiff_AfvsBs])
# Iterate over each frequency band
for i_band in range(len(v_FreqBands)):
    # Get the mean differences in PSD for the current frequency band
    m_AbsPsdDiff_Band = m_AbsPsdDiff[:, :, i_band]
    # Generate a topography plot for the mean differences in PSD
    sc = TopoPlotMEANS(m_AbsPsdDiff_Band, False, False)
    # Save the topography plot as an image
    str_out = Out_dir + f'AllPaths_AbsPSDChange_{v_FreqBands_Names[i_band]}.png'
    sc.screenshot(str_out, print_size=(6, 20), autocrop=False)
# Convert the list of average changes in PSD to a numpy array
m_DiffAbsPSDAllPaths_MtvsBs = np.array(m_DiffAbsPSDAllPaths_MtvsBs)

"""Importing required modules and functions for analysis: This section includes importing additional modules and 
functions necessary for the analysis. It imports modules such as os, scipy, numpy, matplotlib.pylab, and pandas. It 
also imports specific functions from custom modules f_AnalysisGraphs and f_AnalysisFunctions. 

Setting flags to control the behavior of the script: Flags are boolean variables that control the behavior of the 
script. In this case, there are two flags: s_PlotSaveBandEvolution and s_PsdEvolutionAllpaths. These flags can be set 
to True or False to enable or disable specific functionalities in the script. 

Defining directories for input and output files: This section defines the paths to the input and output directories. 
The Data_dir variable represents the path to the directory where the input data files are stored. The Out_dir 
variable represents the path to the directory where the output files will be saved. 

Checking if the output directory exists, and creating it if necessary: This section checks if the output directory 
specified by Out_dir exists. If the directory does not exist, it creates the directory using the os.mkdir() function. 

Reading data information from a CSV file: This section reads data information from a CSV file using the pd.read_csv() 
function. It reads the data from the file named './../RawData/data_info.csv' and stores it in the df_AllInfo 
DataFrame. The separator used in the CSV file is specified as ';'. 

Initializing lists to store the average changes in PSD for each path: This section initializes empty lists to store 
the average changes in PSD (Power Spectral Density) for each path. These lists are m_DiffAbsPSDAllPaths_MtvsBs, 
m_DiffAbsPSDAllPaths_AfvsBs, and m_DiffAbsPSDAllPaths_AfvsMt. 

Looping over each path to load and process the data:
This section uses a loop to iterate over each path in v_strPaths. For each path, it performs the following steps:

Constructs the filename based on the path. Checks if the file exists in the Data_dir directory. If the file exists, 
loads the data from the mat file using scipy.io.loadmat() and stores it in the d_AllData dictionary. Extracts 
relevant data from d_AllData, such as absolute PSD (m_AbsPSD), channel names (v_ChanNames), session times (
v_SessionTimes), and frequency bands (v_FreqBands). Calculates the average changes in PSD for the current path using 
the f_AverageChange() function and stores the results in the respective lists. Storing the average changes in PSD for 
further analysis: After the loop completes, the average changes in PSD for each path are stored in the respective 
lists. The lists m_DiffAbsPSDAllPaths_MtvsBs, m_DiffAbsPSDAllPaths_AfvsBs, and m_DiffAbsPSDAllPaths_AfvsMt now 
contain the average changes in PSD for each path. 

Handling the case when a file is missing, and printing a warning: If a file is missing for a particular path, 
this section is triggered. It prints a warning message indicating the missing file name. The script then breaks out 
of the loop. 

Calculating the mean differences in PSD across all paths: This section calculates the mean differences in PSD across 
all paths. It uses the np.mean() function to compute the mean along the first dimension of the arrays 
m_DiffAbsPSDAllPaths_MtvsBs, m_DiffAbsPSDAllPaths_AfvsBs, and m_DiffAbsPSDAllPaths_AfvsMt. The results are stored in 
the arrays m_AbsPsdDiff_MTvsBS, m_AbsPsdDiff_AfvsBs, and m_AbsPsdDiff_AfvsMt. 

Iterating over each frequency band: This section uses a loop to iterate over each frequency band in v_FreqBands. It 
performs the following steps for each frequency band: 

Extracts the mean differences in PSD for the current frequency band from m_AbsPsdDiff and stores it in 
m_AbsPsdDiff_Band. Extracting the mean differences in PSD for the current frequency band: This section extracts the 
mean differences in PSD for the current frequency band from m_AbsPsdDiff and stores them in the variable 
m_AbsPsdDiff_Band. 

Generating a topography plot for the mean differences in PSD: This section calls the TopoPlotMEANS() function to 
generate a topography plot for the mean differences in PSD for the current frequency band. It passes 
m_AbsPsdDiff_Band, False, and 0 as arguments to the function. The generated plot is stored in the sc variable. 

Saving the topography plot as an image: This section saves the topography plot generated in the previous step as an 
image file. The file path and name are constructed based on the current frequency band and saved using the 
screenshot() method of the sc object. 

Converting the list of average changes in PSD to a numpy array for further analysis: This section converts the list 
m_DiffAbsPSDAllPaths_MtvsBs to a numpy array using np.array(). The resulting numpy array is stored in 
m_DiffAbsPSDAllPaths_MtvsBs. 

"""

