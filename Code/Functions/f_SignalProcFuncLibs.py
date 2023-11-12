import numpy as np
from scipy import signal


def f_RMSFilter(v_signal, d_windSize):
    """
    Applies a Root Mean Square (RMS) filter to the signal.

    Args:
        signal (numpy.ndarray): The input signal.
        window_size (int): The size of the sliding window for RMS calculation.

    Returns:
        numpy.ndarray: The filtered signal after applying the RMS filter.
    """
    v_squaredSignal = np.square(v_signal)  # Square the signal
    v_squaredMean = np.convolve(v_squaredSignal, np.ones(d_windSize)/d_windSize, mode='same')  # Calculate the mean of squared signal using convolution
    rms_signal = np.sqrt(v_squaredMean)  # Take the square root to get the RMS values

    return rms_signal

def f_averageMean(dataArray, wind):
    """
    Calculate the moving average of a given array.

    Parameters:
    - dataArray (array-like): Input data array.
    - wind (int): Size of the moving window.

    Returns:
    - average_y (list): List of average values.
    """
    average_y = []  # Initialize list for average values
    for ind in range(len(dataArray)):  # Iterate over each index of the input array
        minPos = ind - wind  # Calculate the minimum position for the moving window
        maxPos = ind + wind  # Calculate the maximum position for the moving window

        # Adjust the positions if they go beyond the array boundaries
        if minPos < 0:
            minPos = 0
        if maxPos > len(dataArray) - 1:
            maxPos = len(dataArray) - 1

        average_y.append(np.mean(dataArray[minPos:maxPos]))  # Calculate the average and add it to the list

    return average_y  # Return the list of average values

    return np.array(average_y)

def f_GetIIRFilter(p_FsHz, p_PassFreqHz, p_StopFreqsHz):
    """
    Generates an IIR filter given the sampling frequency and passband/stopband frequencies.

    Args:
        p_FsHz (float): The sampling frequency in Hz.
        p_PassFreqHz (list or array): Passband frequencies in Hz.
        p_StopFreqsHz (list or array): Stopband frequencies in Hz.

    Returns:
        filt_FiltSOS (ndarray): The filter coefficients represented as second-order sections (SOS).

    """
    s_AMaxPassDb = 0.5  # Maximum passband ripple in decibels
    s_AMinstopDb = 120  # Minimum stopband attenuation in decibels
    s_NyFreqHz = p_FsHz / 2  # Nyquist frequency in Hz

    p_PassFreqHz = np.array(p_PassFreqHz) / s_NyFreqHz  # Normalize passband frequencies
    p_StopFreqsHz = np.array(p_StopFreqsHz) / s_NyFreqHz  # Normalize stopband frequencies

    s_N, v_Wn = signal.cheb2ord(p_PassFreqHz, p_StopFreqsHz, s_AMaxPassDb, s_AMinstopDb)  # Determine filter order and frequencies
    filt_FiltSOS = signal.cheby2(s_N, s_AMinstopDb, v_Wn, btype='bandpass', output='sos')  # Generate filter coefficients

    return filt_FiltSOS


def f_IIRBiFilter(p_FiltSOS, p_XIn):
    """
    Applies a bidirectional IIR filter to the input signal.

    Args:
        p_FiltSOS (ndarray): The filter coefficients represented as second-order sections (SOS).
        p_XIn (ndarray): The input signal to be filtered.

    Returns:
        filtered_signal (ndarray): The filtered output signal.

    """
    return signal.sosfiltfilt(p_FiltSOS, p_XIn)  # Apply bidirectional IIR filter



def f_GaborTFTransform(p_XIn, p_FsHz, p_FsHz2, p_F1Hz, p_F2Hz, p_FreqResHz, p_NumCycles):
    v_TimeArray = np.arange(0, np.size(p_XIn)) / p_FsHz
    b, a = signal.butter(4, [1, 30], fs=p_FsHz, btype='band')
    p_XIn = signal.filtfilt(b, a, p_XIn)
    p_XIn = signal.resample(p_XIn, int(v_TimeArray[-1] * p_FsHz2))
    p_FsHz = p_FsHz2
    # Creamos un vector de tiempo en segundos
    v_TimeArray = np.arange(0, np.size(p_XIn))
    v_TimeArray = v_TimeArray - v_TimeArray[np.int(np.floor(np.size(v_TimeArray) / 2))]
    v_TimeArray = v_TimeArray / p_FsHz

    # Definimos un rango de frecuencias
    # las cuales usaremos para crear nuestros
    # patrones oscilatorios de prueba
    # En este caso generaremos patrones para
    # frecuencias entre 1 y 50 Hz con pasos
    # de 0.25 Hz
    v_FreqTestHz = np.arange(p_F1Hz, p_F2Hz + p_FreqResHz, p_FreqResHz)

    # Creamos una matriz que usaremos para
    # almacenar el resultado de las
    # convoluciones sucesivas. En esta matriz,
    # cada fila corresponde al resultado de
    # una convolución y cada columna a todos
    # los desplazamientos de tiempo.
    m_ConvMat = np.zeros([np.size(v_FreqTestHz), np.size(p_XIn)], dtype=complex)

    # Se obtiene la transformada de Fourier
    # de la señal p_XIn para usarla en cada iteración
    p_XInfft = np.fft.fft(p_XIn)

    # Ahora creamos un procedimiento iterativo
    # que recorra todas las frecuencias de prueba
    # definidas en el arreglo v_FreqTestHz
    for s_FreqIter in range(np.size(v_FreqTestHz)):
        # Generamos una señal sinusoidal de prueba
        # que oscile a la frecuencia de la iteración
        # s_FreqIter (v_FreqTestHz[s_FreqIter]) y que tenga
        # la misma longitud que la señal p_XIn.
        # En este caso usamos una exponencial compleja.
        xtest = np.exp(1j * 2.0 * np.pi * v_FreqTestHz[s_FreqIter] * v_TimeArray)

        # Creamos una ventana gaussina para
        # limitar nuestro patrón en el tiempo
        # Definimos la desviación estándar de
        # acuerdo al número de ciclos definidos
        # Dividimos entre 2 porque para un ventana
        # gaussiana, una desviación estándar
        # corresponde a la mitad del ancho de la ventana
        xtestwinstd = ((1.0 / v_FreqTestHz[s_FreqIter]) * p_NumCycles) / 2.0
        # Definimos nuestra ventana gaussiana
        xtestwin = np.exp(-0.5 * (v_TimeArray / xtestwinstd) ** 2.0)
        # Multiplicamos la señal patrón por
        # la ventana gaussiana
        xtest = xtest * xtestwin

        # Para cada sinusoidal de prueba obtenemos
        # el resultado de la convolución con la señal p_XIn
        # En este caso nos toca calcular la convolución
        # separadamente para la parte real e imaginaria
        # m_ConvMat[s_FreqIter, :] = np.convolve(p_XIn, np.real(xtest), 'same') + \
        #                        1j * np.convolve(p_XIn, np.imag(xtest), 'same')

        # Se obtine la transformada de Fourier del patrón
        fftxtest = np.fft.fft(xtest)
        # Se toma únicamente la parte real para evitar
        # corrimientos de fase
        fftxtest = abs(fftxtest)
        # Se obtine el resultado de la convolución realizando
        # la multiplicación de las transformadas de Fourier de
        # la señal p_XIn por la del patrón
        m_ConvMat[s_FreqIter, :] = np.fft.ifft(p_XInfft * fftxtest)

    v_TimeArray = v_TimeArray - v_TimeArray[0]
    return m_ConvMat, v_TimeArray, v_FreqTestHz

def f_Matrix2RelAmplitud(pm_InMatrix):
    m_RelPowAmp = pm_InMatrix / np.transpose(np.tile(np.sum(pm_InMatrix, 1), (np.size(pm_InMatrix, 1), 1)))
    m_RelPowAmp = (m_RelPowAmp - np.min(m_RelPowAmp)) / (np.max(m_RelPowAmp) - np.min(m_RelPowAmp))
    return (m_RelPowAmp)

def f_Matrix2RemLogTrend(pm_InMatrix, pv_FreqAxisHz):
    m_RemLogTrendMat = pm_InMatrix
    v_LogFreq = np.log10(pv_FreqAxisHz)

    for s_Count in range(np.size(m_RemLogTrendMat, 1)):
        v_LogTF = np.log10(m_RemLogTrendMat[:, s_Count])
        s_Pol = np.polyfit(np.transpose(v_LogFreq), v_LogTF, 1)
        v_TrendLine = np.polyval(s_Pol, v_LogFreq)
        m_RemLogTrendMat[:, s_Count] = 10 ** (v_LogTF - np.transpose(v_TrendLine))

    return (m_RemLogTrendMat)