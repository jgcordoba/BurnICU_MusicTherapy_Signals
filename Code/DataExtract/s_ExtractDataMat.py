import pandas as pd
from Functions.f_TRC_Reader import *
from Functions.f_SignalProcFuncLibs import *
import scipy.io as sio
import mne
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# -#######################################
# -              Paths
# -#######################################
str_DataPath = './../RawData/'  # Path to the raw data folder
str_OutPath = './../Data/'  # Path to the output folder for EMG features
if not os.path.isdir(str_OutPath):
    os.mkdir(str_OutPath)  # Create the path if this doesn't exist
df_AllInfo = pd.read_csv('./../RawData/data_info.csv', sep=';')  # Read the CSV file 'data_info.csv' into a DataFrame
v_PathNames = df_AllInfo['name_path'].tolist()  # Extract the 'name_path' column as a list

# -#######################################
# -         Boolean Conditions
# -#######################################
b_saveRawData = False  # Save Raw data in .Mat
b_saveData = False
# v_PathNames = ['FSFB_0202']
v_PathNames = v_PathNames[13::]
# -#######################################
# -            Main Loop
# -#######################################
# Iterate over each path in v_PathNames
for i_path in range(len(v_PathNames)):
    print(f'############################################################')
    print(f'Processing path: {v_PathNames[i_path]} - {i_path + 1}/{len(v_PathNames)}')
    print(f'############################################################')

    # Construct the file paths and names based on the current path name
    str_ReadName = str_DataPath + v_PathNames[i_path] + '.TRC'

    # Set the channel-specific information based on the current path name
    if v_PathNames[i_path] == 'FSFB_0101':
        str_ChanStr = ['Fp2', 'Fp1', 'T3', 'T4', 'C3', 'C4', 'O2', 'O1']  # Electrode names for path 0101
        str_SaveChan = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2']  # Channels to extract for path 0101
        str_ChanTyp = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg']  # Channel types for path 0101
        b_ECGInvert = False  # Flag indicating ECG inversion for path 0101
        b_multipleRecs = False  # Flag indicating multiple records for path 0101
    elif v_PathNames[i_path] == 'FSFB_0102':
        # Channel information for path 0102
        str_ChanStr = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'ECG1+']
        str_SaveChan = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'ECG']
        str_ChanTyp = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'ecg']
        b_ECGInvert = True  # ECG inversion for path 0102
        b_multipleRecs = False  # Single record for path 0102
    # ... and so on for other path names
    elif v_PathNames[i_path] == 'FSFB_0201':
        str_ChanStr = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'EMG2+', 'EMG1+']  # 0201
        str_SaveChan = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'ECG', 'EMG']  # Channels to extract
        str_ChanTyp = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'ecg', 'emg']
        b_ECGInvert = False
        b_multipleRecs = False
    elif v_PathNames[i_path] == 'FSFB_0202':
        str_ChanStr = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'EMG1+', 'EMG2+']  # 0202
        str_SaveChan = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'ECG', 'EMG']  # Channels to extract
        str_ChanTyp = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'ecg', 'emg']
        b_ECGInvert = True
        b_multipleRecs = False
    # Handle specific path names with common channel information
    elif v_PathNames[i_path] in ['FSFB_0301', 'FSFB_0302', 'FSFB_0601', 'FSFB_0701', 'FSFB_0901']:
        # Common channel information for certain path names
        str_ChanStr = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'ECG1+', 'EMG1+']
        str_SaveChan = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'ECG', 'EMG']
        str_ChanTyp = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'ecg', 'emg']
        b_ECGInvert = True
        b_multipleRecs = False

    elif v_PathNames[i_path] in ['FSFB_0401', 'FSFB_0402', 'FSFB_0602', 'FSFB_0702', 'FSFB_0802']:
        str_ChanStr = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'ECG1+', 'EMG1+']  # 0202
        str_SaveChan = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'ECG', 'EMG']  # Channels to extract
        str_ChanTyp = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'ecg', 'emg']
        b_ECGInvert = False
        b_multipleRecs = False
    # Handle specific path names with different variations
    elif v_PathNames[i_path] in ['FSFB_0501']:
        # Channel information for path 0501
        str_ChanStr = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'ECG1+', 'EMG1+']
        str_SaveChan = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'ECG', 'EMG']
        str_ChanTyp = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'ecg', 'emg']
        v_RecNames = ['FSFB_0501_A', 'FSFB_0501_B', 'FSFB_0501_C']  # Multiple record names for path 0501
        b_ECGInvert = True
        b_multipleRecs = True  # Multiple records for path 0501
    # ... and so on for other path names

    # Additional path names with multiple records
    elif v_PathNames[i_path] in ['FSFB_0502']:
        str_ChanStr = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'ECG1+', 'EMG1+']
        str_SaveChan = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'ECG', 'EMG']
        str_ChanTyp = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'ecg', 'emg']
        v_RecNames = ['FSFB_0502_A', 'FSFB_0502_B']  # Multiple record names for path 0502
        b_ECGInvert = True
        b_multipleRecs = True  # Multiple records for path 0502
    # ... and so on for other path names

    # More path names with multiple records
    elif v_PathNames[i_path] in ['FSFB_0801']:
        str_ChanStr = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'ECG1+', 'EMG1+']
        str_SaveChan = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2', 'ECG', 'EMG']
        str_ChanTyp = ['eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'eeg', 'ecg', 'emg']
        v_RecNames = ['FSFB_0801_A', 'FSFB_0801_B']  # Multiple record names for path 0801
        b_ECGInvert = False
        b_multipleRecs = True  # Multiple records for path 0801

    m_AllData = []  # Initialize an empty list to store the extracted signals

    # Check if there are multiple records
    if b_multipleRecs:
        # Iterate over each record name
        for i_str in v_RecNames:
            # Check if no data has been extracted yet
            if len(m_AllData) == 0:
                str_ReadName = str_DataPath + i_str + '.TRC'  # Generate the TRC file path for the current record
                st_TRCHead = f_GetTRCHeader(str_ReadName)  # Extract the header of the TRC file
                d_SampleRate = st_TRCHead['RecFreq']  # Get the sample frequency from the header
                print(f'Sampling rate: {d_SampleRate}')  # Print the sampling rate
                m_AllData = f_GetSignalsTRC(str_ReadName, str_ChanStr)  # Extract the data from the TRC file
            else:
                str_ReadName = str_DataPath + i_str + '.TRC'  # Generate the TRC file path for the current record
                m_NewData = f_GetSignalsTRC(str_ReadName, str_ChanStr)  # Extract the data from the TRC file
                m_AllData = np.concatenate((m_AllData, m_NewData),
                                           1)  # Concatenate the newly extracted data with the existing data
    # If there is only one record
    else:
        str_ReadName = str_DataPath + v_PathNames[i_path] + '.TRC'  # Generate the TRC file path for the record
        st_TRCHead = f_GetTRCHeader(str_ReadName)  # Extract the header of the TRC file
        d_SampleRate = st_TRCHead['RecFreq']  # Get the sample frequency from the header
        print(f'Sampling rate: {d_SampleRate}')  # Print the sampling rate
        m_AllData = f_GetSignalsTRC(str_ReadName, str_ChanStr)  # Extract the data from the TRC file

    # Check if ECG inversion is enabled
    if b_ECGInvert:
        m_AllData[8] = m_AllData[8] * -1  # Invert the ECG data by multiplying it by -1

    print(len(m_AllData[0])/d_SampleRate)
    # -#######################################
    # -         Save Raw Data .Mat
    # -#######################################
    if b_saveRawData:

        str_OutPathRawMat = str_OutPath + 'RawMatData/'  # Path to the output folder for EMG features
        str_SaveName = 'RawDataMat_' + v_PathNames[i_path]
        if not os.path.isdir(str_OutPathRawMat):
            os.mkdir(str_OutPathRawMat)  # Create the path if this doesn't exist

        df_AllInfo = pd.read_csv('./../RawData/data_info.csv', sep=';')  # Patient information
        df_PathInfo = df_AllInfo[df_AllInfo['name_path'] == v_PathNames[i_path]]
        # SessionTimes = [int(df_PathInfo['start_time']), int(df_PathInfo['end_time'])]  # Start and End of music
        # sio.savemat(str_OutPathRawMat + str_SaveName + '.mat', mdict={'m_Data': m_AllData,
        #                                                               'v_ChanNames': str_SaveChan,
        #                                                               'd_sampleRate': d_SampleRate,
        #                                                               'v_sessionTimes': SessionTimes})

        print(f'---------------------------------------------')
        print(f'{str_SaveName} successfully saved in path.')
        print(f'---------------------------------------------')

#     m_DataFilt = []
#     # Iterate over each channel
#     for i_chan in range(len(str_ChanStr)):
#         if str_SaveChan[i_chan] == 'ECG':
#             # If the channel is ECG
#             v_Data = m_AllData[i_chan]  # Get the data for the ECG channel
#             st_Filt = f_GetIIRFilter(d_SampleRate, [1, 45], [0.95, 46])  # Obtain the IIR filter for ECG data
#             v_DataFilt = f_IIRBiFilter(st_Filt, v_Data)  # Apply the IIR filter to the ECG data
#             m_DataFilt.append(v_DataFilt)  # Add the filtered ECG data to the list
#         elif str_SaveChan[i_chan] == 'EMG':
#             # If the channel is EMG
#             if d_SampleRate >= 512:
#                 v_Data = m_AllData[i_chan]  # Get the data for the EMG channel
#                 st_Filt = f_GetIIRFilter(d_SampleRate, [20, 200], [19, 201])  # Obtain the IIR filter for EMG data
#                 v_DataFilt = f_IIRBiFilter(st_Filt, v_Data)  # Apply the IIR filter to the EMG data
#                 m_DataFilt.append(v_DataFilt)  # Add the filtered EMG data to the list
#                 st_Filt = f_GetIIRFilter(d_SampleRate, [1, 10], [0.95, 11])  # Obtain the IIR filter for EOG data
#                 v_DataFilt = f_IIRBiFilter(st_Filt, v_Data)  # Apply the IIR filter to the EOG data
#                 m_DataFilt.append(v_DataFilt)  # Add the filtered EOG data to the list
#                 str_SaveChan.append('EOG')  # Add channel name
#                 str_ChanTyp.append('eog')  # Add chanel type
#
#             else:
#                 v_Data = m_AllData[i_chan]  # Get the data for the EMG channel
#                 st_Filt = f_GetIIRFilter(d_SampleRate, [20, 100], [19, 101])  # Obtain the IIR filter for EMG data
#                 v_DataFilt = f_IIRBiFilter(st_Filt, v_Data)  # Apply the IIR filter to the EMG data
#                 m_DataFilt.append(v_DataFilt)  # Add the filtered EMG data to the list
#                 st_Filt = f_GetIIRFilter(d_SampleRate, [1, 10], [0.95, 11])  # Obtain the IIR filter for EOG data
#                 v_DataFilt = f_IIRBiFilter(st_Filt, v_Data)  # Apply the IIR filter to the EOG data
#                 m_DataFilt.append(v_DataFilt)  # Add the filtered EOG data to the list
#                 str_SaveChan.append('EOG')  # Add channel name
#                 str_ChanTyp.append('eog')  # Add chanel type
#         else:
#             # For other channels
#             v_Data = m_AllData[i_chan]  # Get the data for the current channel
#             st_Filt = f_GetIIRFilter(d_SampleRate, [1, 30], [0.95, 31])  # Obtain the default IIR filter
#             v_DataFilt = f_IIRBiFilter(st_Filt, v_Data)  # Apply the default IIR filter to the data
#             m_DataFilt.append(v_DataFilt)  # Add the filtered data to the list
#     ##
#     info = mne.create_info(ch_names=str_SaveChan, sfreq=d_SampleRate, ch_types=str_ChanTyp)  # Create MNE Info object
#     mne_rawData = mne.io.RawArray(m_DataFilt, info)  # Create MNE RawArray object with filtered data
#     # scalings = {'eeg': 100, 'emg': 50, 'ecg': 300, 'eog': 50}  # Define scaling values for plotting
#     # mne_rawData.plot(n_channels=len(str_SaveChan), scalings=scalings, title='Data from arrays', duration=45, show=True,
#     #                  block=True)  # Plot raw data
#
#     n_components = 8  # Number of components to use for ICA
#     ica = mne.preprocessing.ICA(n_components=n_components, method='fastica', random_state=97)  # Initialize ICA object
#     ica.fit(mne_rawData, picks=['eeg'])  # Fit ICA on the EEG data
#     if 'EOG' in str_SaveChan:
#         eog_indices, eog_scores = ica.find_bads_eog(mne_rawData, threshold=0.9,
#                                                     ch_name=['EOG'], measure='correlation')  # Find EOG artifacts
#         eog_indices = [np.argmax(np.abs(eog_scores))]# Select the component with the highest EOG score as artifact
#     else:
#         eog_indices, eog_scores = ica.find_bads_eog(mne_rawData, threshold=0.9, ch_name=['Fp1', 'Fp2'])  # Find EOG artifacts
#         eog_indices = [np.argmax(np.mean(np.abs(eog_scores), 0))]# Select the component with the highest EOG score as artifact
#
#     # ica.plot_sources(mne_rawData, start=0., stop=60)  # Plot IC sources
#     eog_indices = [0,1]
#     ica.plot_overlay(mne_rawData, exclude=eog_indices, start=0.,
#                      stop=len(mne_rawData) / d_SampleRate)  # Plot IC overlay on raw data
#     ica.exclude = eog_indices  # Exclude the EOG component
#     ica.apply(mne_rawData)  # Apply ICA to remove identified artifacts
#     m_DataICA = mne_rawData.get_data()  # Get the ICA-corrected data
# ##
#     # fig, ax = plt.subplots(len(str_SaveChan), 1, sharex=True)  # Create subplots for each channel
#     # for i_chann in range(len(str_SaveChan)):  # Iterate over each channel
#     #     ax[i_chann].plot(m_DataFilt[i_chann])  # Plot the original filtered data
#     #     ax[i_chann].plot(m_DataICA[i_chann])  # Plot the ICA-corrected data
#     # plt.show()
# ##
#     if b_saveData:
#
#         str_OutPathMat = str_OutPath + 'MatData/'  # Path to the output folder for EMG features
#         str_SaveName = 'DataMat_' + v_PathNames[i_path]
#         if not os.path.isdir(str_OutPathMat):
#             os.mkdir(str_OutPathMat)  # Create the path if this doesn't exist
#         df_AllInfo = pd.read_csv('./../RawData/data_info.csv', sep=';')
#         df_PathInfo = df_AllInfo[df_AllInfo['name_path'] == v_PathNames[i_path]]
#         SessionTimes = [int(df_PathInfo['start_time']), int(df_PathInfo['end_time'])]
#
#         sio.savemat(str_OutPathMat + str_SaveName + '.mat', mdict={'m_Data': m_DataICA,
#                                                               'v_ChanNames': str_SaveChan,
#                                                               'd_sampleRate': d_SampleRate,
#                                                               'v_sessionTimes': SessionTimes})
#
#         print(f'---------------------------------------------')
#         print(f'{str_SaveName} successfully saved in path.')
#         print(f'---------------------------------------------')