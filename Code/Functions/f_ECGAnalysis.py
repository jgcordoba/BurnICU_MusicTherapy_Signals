from Functions.f_SignalProcFuncLibs import *

def confidence_ellipse(x, y, ax, color, n_std):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor='none', edgecolor='k', alpha=1, linestyle='--', linewidth=1.5)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    # print(scale_x,scale_y)
    return ax.add_patch(ellipse)
def extractECGfreq(v_tacoInterpol,d_FsHzNew,nperseg = 1900):
    from scipy import signal
    import numpy as np
    from scipy.integrate import simps
    d_freqMin = 0.04
    d_freqMid = 0.15
    d_freqMax = 0.4
    v_freqRes, v_density = signal.welch(v_tacoInterpol, d_FsHzNew, nperseg=nperseg)
    v_density = v_density[np.array(v_freqRes > d_freqMin) * np.array(v_freqRes < d_freqMax)]
    v_freqRes = v_freqRes[np.array(v_freqRes > d_freqMin) * np.array(v_freqRes < d_freqMax)]
    v_density = (v_density / np.sum(v_density))*100
    d_LFPower = np.sum(v_density[v_freqRes < d_freqMid])
    d_HFPower = np.sum(v_density[v_freqRes > d_freqMid])
    # v_density = 10 * np.log10(v_density)
    # d_LFPower = simps(v_density[v_freqRes < d_freqMid], dx=(v_freqRes[1] - v_freqRes[0]))
    # d_HFPower = simps(v_density[v_freqRes > d_freqMid], dx=(v_freqRes[1] - v_freqRes[0]))
    v_density = f_averageMean(v_density,3)

    return v_freqRes, v_density,[d_LFPower,d_HFPower]


def cleanTachogram(v_RRTaco, v_TacoTime, d_adjuster=1):
    """
    Cleans the tachogram by removing outliers and limiting values based on given thresholds.

    Args:
        v_RRTaco (numpy.ndarray): Array of RR intervals representing the tachogram.
        v_TacoTime (numpy.ndarray): Array of time intervals corresponding to the RR intervals.
        d_adjuster (float, optional): Adjustment factor for threshold values. Defaults to 1.

    Returns:
        tuple: Cleaned RR intervals (v_RRTacoClean) and corresponding time intervals (v_RRTimeClean).
    """

    import numpy as np

    # Calculate the threshold for RR interval change
    d_changeRRThres = 0.25 / d_adjuster

    # Remove RR intervals and corresponding time intervals that exceed the change threshold
    v_RRTacoClean = np.delete(v_RRTaco[:-1], np.abs(np.diff(v_RRTaco)) > d_changeRRThres)
    v_RRTimeClean = np.delete(v_TacoTime[:-1], np.abs(np.diff(v_RRTaco)) > d_changeRRThres)

    # Calculate the maximum and minimum thresholds for RR intervals
    d_maxValueRRthres = 1.5 / d_adjuster
    d_minValueRRthres = 0.4 * d_adjuster

    # Remove RR intervals and corresponding time intervals that exceed the maximum threshold
    v_RRTimeClean = np.delete(v_RRTimeClean, v_RRTacoClean >= d_maxValueRRthres)
    v_RRTacoClean = np.delete(v_RRTacoClean, v_RRTacoClean >= d_maxValueRRthres)

    # Remove RR intervals and corresponding time intervals that are below the minimum threshold
    v_RRTimeClean = np.delete(v_RRTimeClean, v_RRTacoClean <= d_minValueRRthres)
    v_RRTacoClean = np.delete(v_RRTacoClean, v_RRTacoClean <= d_minValueRRthres)

    return v_RRTacoClean, v_RRTimeClean

def cleanECGbyTachogram(v_ECGData, d_sampleRate, b_showPlot=False):
    import numpy as np
    import biosppy
    import neurokit2 as nk
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')

    v_Time = np.arange(0, len(v_ECGData)) / d_sampleRate

    m_ECGResults = biosppy.signals.ecg.ecg(v_ECGData, sampling_rate=d_sampleRate, show=False)
    m_ECGResults2 = nk.ecg_peaks(v_ECGData, sampling_rate=d_sampleRate)

    v_rPeaks = np.sort(np.unique(np.concatenate((
        np.intersect1d(m_ECGResults[2], m_ECGResults2[1]['ECG_R_Peaks']),
        np.intersect1d(m_ECGResults[2] - 1, m_ECGResults2[1]['ECG_R_Peaks']),
        np.intersect1d(m_ECGResults[2] + 1, m_ECGResults2[1]['ECG_R_Peaks'])))))

    v_RRTaco = np.diff(v_rPeaks) / d_sampleRate
    v_TacoTime = v_Time[v_rPeaks[1::]]

    v_RRTacoClean, v_RRTimeClean = cleanTachogram(v_RRTaco, v_TacoTime)

    v_diffRRTimeClean = np.diff(v_RRTimeClean)
    v_timeRemove = v_RRTimeClean[1::][v_diffRRTimeClean > np.mean(v_RRTacoClean) * 5]
    v_lenTimeRemove = v_diffRRTimeClean[v_diffRRTimeClean > np.mean(v_RRTacoClean) * 5]

    v_ECGDataClean = v_ECGData
    v_ECGTimeClean = v_Time
    m_timeRemoveLimits = np.array([])
    v_rPeaksClean = v_rPeaks

    if len(v_timeRemove) != 0:
        m_timeRemoveLimits = np.array([v_timeRemove - v_lenTimeRemove, v_timeRemove])
        v_timeRemoveIndx = np.where((m_timeRemoveLimits[0][1:] - m_timeRemoveLimits[1][:-1] > 15) == 0)[0]

        v_firtsTimeValues = np.delete(m_timeRemoveLimits[0], v_timeRemoveIndx + 1)
        v_lastTimeValues = np.delete(m_timeRemoveLimits[1], v_timeRemoveIndx)
        m_timeRemoveLimits = np.array([v_firtsTimeValues[::-1], v_lastTimeValues[::-1]])

        for i_start, i_end in np.transpose(m_timeRemoveLimits) * d_sampleRate:
            v_ECGDataClean = np.concatenate((v_ECGDataClean[:int(i_start)], v_ECGDataClean[int(i_end) + 1:]))
            v_ECGTimeClean = np.concatenate((v_ECGTimeClean[:int(i_start)], v_ECGTimeClean[int(i_end) + 1:]))

        m_ECGResultsClean = biosppy.signals.ecg.ecg(v_ECGDataClean, sampling_rate=d_sampleRate, show=False)
        m_ECGResultsClean2 = nk.ecg_peaks(v_ECGDataClean, sampling_rate=d_sampleRate)

        v_rPeaksClean = np.sort(np.unique(np.concatenate((
            np.intersect1d(m_ECGResultsClean[2], m_ECGResultsClean2[1]['ECG_R_Peaks']),
            np.intersect1d(m_ECGResultsClean[2] - 1, m_ECGResultsClean2[1]['ECG_R_Peaks']),
            np.intersect1d(m_ECGResultsClean[2] + 1, m_ECGResultsClean2[1]['ECG_R_Peaks'])))))

        v_RRTacoClean = np.diff(v_rPeaksClean) / d_sampleRate
        v_RRTimeClean = v_ECGTimeClean[v_rPeaksClean[1::]]

    if b_showPlot:
        fig, ax = plt.subplots(2, 1, sharex=True)
        ax[0].plot(v_Time, v_ECGData, 'gray')
        ax[0].scatter(v_Time[v_rPeaks], v_ECGData[v_rPeaks], color='salmon', marker='.', linewidth=0.25, zorder=2)
        # if len(v_timeRemove) != 0:
        ax[0].plot(v_ECGTimeClean, v_ECGDataClean, 'k')
        ax[0].scatter(v_ECGTimeClean[v_rPeaksClean], v_ECGDataClean[v_rPeaksClean], color='red', marker='.',
                      linewidth=0.25, zorder=2)

        if len(m_timeRemoveLimits) > 0:
            for i in range(len(m_timeRemoveLimits[0])):
                ax[0].fill_between(m_timeRemoveLimits[:, i], np.min(v_ECGData), np.max(v_ECGData), color='r',
                                   alpha=0.25, linewidth=0)

        ax[1].plot(v_TacoTime, v_RRTaco)
        ax[1].plot(v_RRTimeClean, v_RRTacoClean)
        fig.show()

    return (v_ECGDataClean, v_ECGTimeClean, v_rPeaksClean, m_timeRemoveLimits)


def confidence_ellipse(x, y, ax, n_std):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)

    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor='none', edgecolor='k', alpha=1, linestyle='-', linewidth=1)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    # print(scale_x,scale_y)
    return ax.add_patch(ellipse)

def poincare_parameters(v_rr):
    # Shift the RR differences by one position
    v_rr = v_rr[:-1]
    v_rr_diff = np.diff(v_rr)
    # Calculate SD1 and SD2 using linear regression analysis
    d_SDNN = np.std(v_rr)
    d_SDSD = np.std(v_rr_diff)
    np.std(v_rr_diff) / np.sqrt(2)

    d_SD1 = np.sqrt(0.5 * (d_SDSD ** 2))
    d_SD2 = np.sqrt((2 * (d_SDNN ** 2)) - (0.5 * (d_SDSD ** 2)))
    return d_SD1, d_SD2