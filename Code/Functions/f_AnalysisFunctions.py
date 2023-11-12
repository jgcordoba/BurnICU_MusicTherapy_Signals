import numpy as np

def FullfillData(m_AbsPSD, v_SessionTimes, d_stepSize):
    """
    Fill missing data in the PSD matrix by extending the available data based on session times.

    Parameters:
    - m_AbsPSD (array-like): Matrix containing the power spectral density (PSD) data.
    - v_SessionTimes (array-like): Array containing the start and end times of different sessions.
    - d_stepSize (float): Step size used for data collection.

    Returns:
    - m_AbsPSDFull (array-like): Matrix with filled data, extending the available data based on session times.
    """
    d_maxBs = int(5 * 60 / d_stepSize)  # Maximum length of the "Before Session" period in number of data points
    d_maxMt = int(15 * 60 / d_stepSize)  # Maximum length of the "During Session" period in number of data points
    d_maxAf = int(5 * 60 / d_stepSize)  # Maximum length of the "After Session" period in number of data points
    m_AbsPSDFull = []  # Initialize list for the filled PSD data

    for i_band in range(len(m_AbsPSD[0])):
        m_AbsPSDBand = m_AbsPSD[:, i_band]  # Extract the PSD data for a specific band

        # Fill "Before Session" data
        m_AbsPSDBand_Bs = m_AbsPSDBand[:, 0:int(v_SessionTimes[0])]
        m_AbsPSDBandAll_Bs = np.zeros((len(m_AbsPSD), d_maxBs))
        d_med = int((d_maxBs / 2))
        d_pos = int(len(m_AbsPSDBand_Bs[0]) / 2)
        m_AbsPSDBandAll_Bs[:, d_med - d_pos:d_med + d_pos] = m_AbsPSDBand_Bs

        # Fill "During Session" data
        m_AbsPSDBand_Mt = m_AbsPSDBand[:, int(v_SessionTimes[0]):int(v_SessionTimes[1])]
        m_AbsPSDBandAll_Mt = np.zeros((len(m_AbsPSD), d_maxMt))
        d_med = int((d_maxMt / 2))
        d_pos = int(len(m_AbsPSDBand_Mt[0]) / 2)
        m_AbsPSDBandAll_Mt[:, d_med - d_pos:d_med + d_pos] = m_AbsPSDBand_Mt

        # Fill "After Session" data
        m_AbsPSDBand_Af = m_AbsPSDBand[:, int(v_SessionTimes[1])::]
        m_AbsPSDBandAll_Af = np.zeros((len(m_AbsPSD), d_maxAf))
        d_med = int((d_maxAf / 2))
        d_pos = int(len(m_AbsPSDBand_Af[0]) / 2)
        m_AbsPSDBandAll_Af[:, d_med - d_pos:d_med + d_pos] = m_AbsPSDBand_Af

        # Concatenate the filled data for all periods
        m_AbsPSDBandFull = np.concatenate((m_AbsPSDBandAll_Bs, m_AbsPSDBandAll_Mt, m_AbsPSDBandAll_Af), 1)
        m_AbsPSDFull.append(m_AbsPSDBandFull)  # Add the filled data for the band to the main list

    m_AbsPSDFull = np.array(m_AbsPSDFull)  # Convert the filled data to a numpy array
    return m_AbsPSDFull  # Return the filled PSD data


def cleanDataExract(m_PatAbsPsd, v_SessionTimes, v_ChanNamesEEG, v_FreqBands, s_windStep, d_BslWind, d_MtlWind, d_AflWind, d_factor, relative):
    """
    Clean and extract data from the power spectral density (PSD) matrix based on specified parameters.

    Parameters:
    - m_PatAbsPsd (array-like): Matrix containing the absolute PSD data.
    - v_SessionTimes (array-like): Array containing the start and end times of different sessions.
    - v_ChanNamesEEG (array-like): Array containing the names of EEG channels.
    - v_FreqBands (array-like): Array containing the frequency bands.
    - s_windStep (float): Window step size for data extraction.
    - d_BslWind (float): Length of the baseline window.
    - d_MtlWind (float): Length of the during-session window.
    - d_AflWind (float): Length of the after-session window.
    - d_factor (float): Factor used for data cleaning.
    - relative (bool): Flag indicating whether to return relative or absolute PSD.

    Returns:
    - m_PatFinalPsd (array-like): Matrix containing the cleaned and extracted PSD data.
    """

    m_PatCleanPsd = []  # Initialize list for the cleaned PSD data

    for i_band in range(len(v_FreqBands)):
        m_BandPsd = m_PatAbsPsd[:, i_band]  # Extract the PSD data for a specific frequency band

        # Data cleaning
        m_cleanIndx = m_BandPsd < np.mean(m_BandPsd) + (np.std(m_BandPsd) * d_factor)
        m_cleanIndx = np.mean(m_cleanIndx, 0) != 1

        # Data extraction for different periods
        m_BandPatPsd_Bs = m_BandPsd[:, 0:int(v_SessionTimes[0] / s_windStep)]
        m_BandPatPsd_Mt = m_BandPsd[:, int(v_SessionTimes[0] / s_windStep):int(v_SessionTimes[1] / s_windStep)]
        m_BandPatPsd_Af = m_BandPsd[:, int(v_SessionTimes[1] / s_windStep):]

        # Cleaned data for different periods
        m_BandPatCleanPsd_Bs = np.delete(m_BandPatPsd_Bs, m_cleanIndx[0:int(v_SessionTimes[0] / s_windStep)], axis=1)
        print(f'Tiempo BS: {len(m_BandPatCleanPsd_Bs[0]) * 1.5 / 60}')  # Print the time for the "Before Session" period
        m_BandPatCleanPsd_Mt = np.delete(m_BandPatPsd_Mt, m_cleanIndx[int(v_SessionTimes[0] / s_windStep):int(
            v_SessionTimes[1] / s_windStep)], axis=1)
        print(f'Tiempo Mt: {len(m_BandPatCleanPsd_Mt[0]) * 1.5 / 60}')  # Print the time for the "During Session" period
        m_BandPatCleanPsd_Af = np.delete(m_BandPatPsd_Af, m_cleanIndx[int(v_SessionTimes[1] / s_windStep):], axis=1)
        print(f'Tiempo Af: {len(m_BandPatCleanPsd_Af[0]) * 1.5 / 60}')  # Print the time for the "After Session" period

        # Extracted central data for different periods
        m_BandWindPsd_Bs = m_BandPatCleanPsd_Bs[:, int((len(m_BandPatCleanPsd_Bs[0]) / 2) - (d_BslWind / 2)):int(
            (len(m_BandPatCleanPsd_Bs[0]) / 2) + (d_BslWind / 2))]
        m_BandWindPsd_Mt = m_BandPatCleanPsd_Mt[:, int((len(m_BandPatCleanPsd_Mt[0]) / 2) - (d_MtlWind / 2)):int(
            (len(m_BandPatCleanPsd_Mt[0]) / 2) + (d_MtlWind / 2))]
        m_BandWindPsd_Af = m_BandPatCleanPsd_Af[:, int((len(m_BandPatCleanPsd_Af[0]) / 2) - (d_AflWind / 2)):int(
            (len(m_BandPatCleanPsd_Af[0]) / 2) + (d_AflWind / 2))]

        m_BandAllCleanPsd = np.concatenate((m_BandWindPsd_Bs, m_BandWindPsd_Mt, m_BandWindPsd_Af), axis=1)

        if relative:
            m_PatCleanPsd.append(m_BandAllCleanPsd)
        else:
            m_BandNormPsd = (np.transpose(m_BandAllCleanPsd) - np.mean(m_BandWindPsd_Bs, 1)) / (
                np.std(m_BandWindPsd_Bs, 1))
            m_PatCleanPsd.append(np.transpose(m_BandNormPsd))

    m_PatCleanPsd = np.array(m_PatCleanPsd)  # Convert the cleaned PSD data to a numpy array
    m_PatFinalPsd = []

    for i_chan in range(len(v_ChanNamesEEG)):
        m_ChannBandCleanPsd = m_PatCleanPsd[:, i_chan]
        m_PatFinalPsd.append(m_ChannBandCleanPsd)

    m_PatFinalPsd = np.array(m_PatFinalPsd)  # Convert the final PSD data to a numpy array

    return m_PatFinalPsd  # Return the cleaned and extracted PSD data

import numpy as np

def f_AveragePSD(m_Data, v_SessionTimes, changes):
    """
    Calculate the average change between different segments of data.

    Args:
        m_Data (list): A list of arrays representing the data.
        v_SessionTimes (list): A list of two values representing the session times.

    Returns:
        tuple: A tuple containing three arrays: m_DiffDataMT_BS, m_DiffDataAF_BS, m_DiffDataAF_MT.
    """


    m_DiffData_BS = []
    m_DiffData_Mt = []
    m_DiffData_AF = []

    m_DiffDataMT_BS = []
    m_DiffDataAF_BS = []
    m_DiffDataAF_MT = []

    for i_chann in range(len(m_Data)):     # Iterate over each channel of the data
        v_Data = m_Data[i_chann]         # Get the data for the current channel
        # Calculate the mean of the data for the first session time
        v_MeanData_Bs = np.mean(v_Data[:, 0:int(v_SessionTimes[0])], 1)
        # Calculate the mean of the data for the second session time
        v_MeanData_mt = np.mean(v_Data[:, int(v_SessionTimes[0]):int(v_SessionTimes[1])], 1)
        # Calculate the mean of the data for the third session time
        v_MeanData_Af = np.mean(v_Data[:, int(v_SessionTimes[1])::], 1)

        m_DiffData_BS.append(v_MeanData_Bs)
        m_DiffData_Mt.append(v_MeanData_mt)
        m_DiffData_AF.append(v_MeanData_Af)

        if changes:
            # Append the difference between v_MeanData_mt and v_MeanData_Bs to the m_DiffDataMT_BS list
            m_DiffDataMT_BS.append(v_MeanData_mt - v_MeanData_Bs)
            # Append the difference between v_MeanData_Af and v_MeanData_Bs to the m_DiffDataAF_BS list
            m_DiffDataAF_BS.append(v_MeanData_Af - v_MeanData_Bs)
            # Append the difference between v_MeanData_Af and v_MeanData_mt to the m_DiffDataAF_MT list
            m_DiffDataAF_MT.append(v_MeanData_Af - v_MeanData_mt)

    if changes:
        # Convert the difference lists to numpy arrays
        m_DiffDataMT_BS = np.array(m_DiffDataMT_BS)
        m_DiffDataAF_BS = np.array(m_DiffDataAF_BS)
        m_DiffDataAF_MT = np.array(m_DiffDataAF_MT)
        # Return the calculated differences as a tuple
        return m_DiffDataMT_BS, m_DiffDataAF_BS, m_DiffDataAF_MT

    else:
        m_DiffData_BS = np.array(m_DiffData_BS)
        m_DiffData_Mt = np.array(m_DiffData_Mt)
        m_DiffData_AF = np.array(m_DiffData_AF)
        return m_DiffData_BS, m_DiffData_Mt, m_DiffData_AF


def OneDimFullFill(m_measureEvolution, v_SessionTimes, d_sampleRate ):

    d_maxBs = int(5 * 60 * d_sampleRate)
    d_maxMt = int(15 * 60 * d_sampleRate)
    d_maxAf = int(5 * 60 * d_sampleRate)

    m_measure_Bs = m_measureEvolution[0:int(v_SessionTimes[0] * d_sampleRate)]
    m_measure_Mt = m_measureEvolution[int(v_SessionTimes[0] * d_sampleRate):int(v_SessionTimes[1] * d_sampleRate)]
    m_measure_Af = m_measureEvolution[int(v_SessionTimes[1] * d_sampleRate)::]

    m_measureFull_Bs = np.zeros(d_maxBs)
    m_measureFull_Mt = np.zeros(d_maxMt)
    m_measureFull_Af = np.zeros(d_maxAf)

    d_med = int((d_maxBs / 2))
    d_pos = len(m_measure_Bs) / 2 + 0.1
    m_measureFull_Bs[d_med - int(d_pos):d_med + round(d_pos)] = m_measure_Bs

    d_med = int((d_maxMt / 2))
    d_pos = len(m_measure_Mt) / 2 + 0.1
    m_measureFull_Mt[d_med - int(d_pos):d_med + round(d_pos)] = m_measure_Mt

    d_med = int((d_maxAf / 2))
    d_pos = (len(m_measure_Af) / 2) + 0.1

    m_measureFull_Af[d_med - int(d_pos):d_med + round(d_pos)] = m_measure_Af

    m_measureFull = np.concatenate((m_measureFull_Bs, m_measureFull_Mt, m_measureFull_Af))
    return(m_measureFull)