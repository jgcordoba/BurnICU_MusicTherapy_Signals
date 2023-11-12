import numpy as np
from scipy.signal import welch
from scipy.integrate import simps
from Functions.f_SignalProcFuncLibs import *

def f_bandPower(v_data, sf, v_freqBand):
    """
    Compute the power of a given frequency band in a signal.
    Args:
        v_data (array-like): The input signal data.
            - Should be in the time domain.
        sf (float): The sampling frequency of the signal.
            - Should be in Hz.
        band (array-like): The frequency band of interest [low, high].
            - The lower and upper frequency limits of the band in Hz.
    Returns:
        float: The power of the specified frequency band in the signal.
    """

    band = np.asarray(v_freqBand)  # Convert band to NumPy array for consistency
    low, high = band  # Extract lower and upper frequency limits from band
    d_length = len(v_data)  # Get the length of the input signal
    v_freqs, v_psd = welch(v_data, sf, nfft=d_length) # Estimate the (PSD) of the signal using Welch's method
    v_freq_res = v_freqs[1] - v_freqs[0]  # Compute the frequency resolution
    idx_band = np.logical_and(v_freqs >= low, v_freqs <= high) # Determine indices corresponding frequency band
    d_bandPower = simps(v_psd[idx_band], dx=v_freq_res) # Integrate the PSD within the frequency band

    return d_bandPower


def f_powerEvolution(m_dataChann, s_FreqBand, s_SRate, s_windSize, s_windStep):
    """
    Compute the frequency evolution of a signal within a specific frequency band.
    Args:
        m_dataChann (array-like): The input signal data for a single channel.
            - Should be in the time domain.
        s_FreqBand (array-like): The frequency band of interest [low, high].
            - The lower and upper frequency limits of the band in Hz.
        s_SRate (float): The sampling rate of the signal.
            - Should be in Hz.
        s_windSize (float): The size of the analysis window in seconds.
        s_windStep (float): The step size between consecutive analysis windows in seconds.

    Returns:
        array-like: The frequency evolution of the signal within the specified frequency band.
    Notes:
    - The input signal data should be in the time domain.
    - The frequency band should be specified as [low, high], where `low` and `high` are in Hz.
    - The sampling rate (`s_SRate`) should be in Hz.
    - The size of the analysis window (`s_windSize`) and the step size (`s_windStep`) should be in seconds.
    - The function computes the frequency evolution of the signal by applying a sliding window approach.
    - The power within the specified frequency band is calculated using the `f_bandPower` function.
    - The resulting frequency evolution is returned as an array.
    """

    v_EvolutionConvBand = []
    # Apply a sliding window approach to compute the frequency evolution
    for i_wind in np.arange(0, int(len(m_dataChann) / s_SRate), s_windStep):
        # Extract the data within the current window
        m_dataChann_wind = m_dataChann[int(i_wind * s_SRate):int((s_windSize + i_wind) * s_SRate)]
        # Compute the power within the frequency band for the current window
        i_psdEvo = f_bandPower(m_dataChann_wind, s_SRate, s_FreqBand)
        # Append the computed power to the frequency evolution list
        v_EvolutionConvBand.append(i_psdEvo)
    v_EvolutionConvBand = np.array(v_EvolutionConvBand)  # Convert the frequency evolution list to a NumPy array
    return v_EvolutionConvBand

def MeanPsdEvolution(m_AbsPSD_AllPaths):
    """
    Calculate the mean and standard deviation of the power spectral density (PSD) evolution across multiple paths.

    Parameters:
    - m_AbsPSD_AllPaths (array-like): Multi-dimensional array containing the PSD evolution for each path, band, and channel.

    Returns:
    - m_dataAllPathsMeanFinal (array-like): Mean PSD evolution across all paths and channels.
    - m_dataAllPathsStdFinal (array-like): Standard deviation of the PSD evolution across all paths and channels.
    """
    m_dataAllPathsMean = []  # Initialize list for mean PSD evolution across all paths
    m_dataAllPathsStd = []  # Initialize list for standard deviation of PSD evolution across all paths

    for i_band in range(len(m_AbsPSD_AllPaths[0])):
        m_dataAllPathsMeanBand = []  # Initialize list for mean PSD evolution per band
        m_dataAllPathsStdBand = []  # Initialize list for standard deviation of PSD evolution per band

        for i_chan in range(len(m_AbsPSD_AllPaths[0][0])):
            v_dataAllPaths = []  # Initialize list for PSD evolution across all paths for a specific band and channel

            for i_path in range(len(m_AbsPSD_AllPaths)):
                v_data = m_AbsPSD_AllPaths[i_path][i_band][i_chan]  # Get the PSD evolution for a specific path, band, and channel
                v_data = f_averageMean(v_data, 30)  # Apply a moving average to the PSD evolution
                v_dataAllPaths.append(v_data)  # Add the processed PSD evolution to the list

            v_dataAllPaths = np.array(v_dataAllPaths)
            v_dataAllPaths[v_dataAllPaths == 0] = np.nan  # Replace zeros with NaN values
            v_dataAllPathsMean = np.nanmean(v_dataAllPaths, axis=0)  # Calculate the mean PSD evolution across paths
            v_dataAllPathsStd = np.nanstd(v_dataAllPaths, axis=0)  # Calculate the standard deviation of the PSD evolution across paths

            m_dataAllPathsMeanBand.append(v_dataAllPathsMean)  # Add the mean PSD evolution to the list per band
            m_dataAllPathsStdBand.append(v_dataAllPathsStd)  # Add the standard deviation of the PSD evolution to the list per band

        m_dataAllPathsMean.append(m_dataAllPathsMeanBand)  # Add the mean PSD evolution per band to the main list
        m_dataAllPathsStd.append(m_dataAllPathsStdBand)  # Add the standard deviation of the PSD evolution per band to the main list

    m_dataAllPathsMean = np.array(m_dataAllPathsMean)
    m_dataAllPathsMeanFinal = []

    for i_chan in range(len(m_dataAllPathsMean[0])):
        m_dataAllPathsMeanChann = m_dataAllPathsMean[:, i_chan]  # Extract the mean PSD evolution for a specific channel across all bands
        m_dataAllPathsMeanFinal.append(m_dataAllPathsMeanChann)  # Add the mean PSD evolution for the channel to the final list

    m_dataAllPathsMeanFinal = np.array(m_dataAllPathsMeanFinal)

    m_dataAllPathsStd = np.array(m_dataAllPathsStd)
    m_dataAllPathsStdFinal = []

    for i_chan in range(len(m_dataAllPathsStd[0])):
        m_dataAllPathsStdChann = m_dataAllPathsStd[:, i_chan]  # Extract the standard deviation of the PSD evolution for a specific channel across all bands
        m_dataAllPathsStdFinal.append(m_dataAllPathsStdChann)  # Add the standard deviation of the PSD evolution for the channel to the final list

    m_dataAllPathsStdFinal = np.array(m_dataAllPathsStdFinal)

    return (m_dataAllPathsMeanFinal, m_dataAllPathsStdFinal)  # Return the mean and standard deviation of the PSD evolution
