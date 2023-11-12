import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from Functions.f_SignalProcFuncLibs import *
from Functions.f_ECGAnalysis import *
import matplotlib.colors as colors
import numpy as np

def PlotBandEvolutionAllPaths(m_Data, m_DataStd, s_windStep, v_FreqBands_Names, v_FreqBands, v_ChanNames,
                              v_SessionTimes, title, relative):
    fig, ax = plt.subplots(len(v_FreqBands), 1, sharex=True, gridspec_kw=dict(hspace=0), figsize=(5, 14))
    v_TimeArray = np.arange(0, len(m_Data[0][0])) * s_windStep
    AverageWind = 5
    colors = pl.cm.Reds(np.linspace(0, 1, len(m_Data) + 3))

    for i_band in range(len(v_FreqBands)):
        i_max = 0
        i_min = 1
        for i_chan in range(len(v_ChanNames)):
            v_Data = m_Data[i_chan][i_band]
            v_DataStd = m_DataStd[i_chan][i_band]
            v_DataStd_Soft = np.array(f_averageMean(v_DataStd, AverageWind))
            v_Data_Soft = np.array(f_averageMean(v_Data, AverageWind))

            if i_max < max(v_Data_Soft):
                i_max = max(v_Data_Soft)
            if i_min > min(v_Data_Soft):
                i_min = min(v_Data_Soft)

            ax[i_band].plot(v_TimeArray, v_Data_Soft, color=colors[i_chan + 3], label=v_ChanNames[i_chan])
            ax[i_band].fill_between(v_TimeArray, v_Data_Soft - v_DataStd_Soft, v_Data_Soft + v_DataStd_Soft,
                                    color=colors[i_chan + 3], alpha=0.15)
            [ax[i_band].spines[i].set_visible(False) for i in ['top', 'right', 'bottom', 'left']]

        # if i_band == 0:
        #     ax[i_band].legend(ncol=1, bbox_to_anchor=(1.125, -0.5), fancybox=True, shadow=True, fontsize=7)
        #     if len(v_SessionTimes) == 2:
        #         ax[i_band].text(v_SessionTimes[0] - 5, i_max * 1.5, 'Start', rotation=90,
        #                         fontsize=8)  # Se hubica un texto indicando la banda de seguridad ,bbox=bbox_props
        #         ax[i_band].text(v_SessionTimes[1] - 5, i_max * 1.5, 'End', rotation=90,
        #                         fontsize=8)  # Se hubica un texto indicando la banda de seguridad ,bbox=bbox_props
        #     else:
        #         ax[i_band].text(v_SessionTimes[0] - 5, i_max * 1.5, 'Start', rotation=90,
        #                         fontsize=8)  # Se hubica un texto indicando la banda de seguridad ,bbox=bbox_props

        if len(v_SessionTimes) == 2:
            ax[i_band].axvline(v_SessionTimes[0], linewidth=1, linestyle='--', color='k')
            ax[i_band].axvline(v_SessionTimes[1], linewidth=1, linestyle='--', color='k')
        else:
            ax[i_band].axvline(v_SessionTimes[0], linewidth=1, linestyle='--', color='k')

        # ax[i_band].grid(linewidth=0.4, linestyle=':', color='k', which='both')
        # ax[i_band].set_ylabel(f'{v_FreqBands_Names[i_band]}', fontsize=16)

        ax[i_band].yaxis.tick_right()
        # if i_min < 0:
        #     ax[i_band].set_ylim([i_min * 1.4, i_max * 1.4])
        # else:
        #     ax[i_band].set_ylim([i_min * 0.6, i_max * 1.4])
        ax[i_band].yaxis.set_tick_params(labelsize=10)
        ax[i_band].xaxis.set_tick_params(labelsize=12)
        if i_band == len(v_FreqBands) - 1:
            ax[i_band].set_xlabel('Time (s)')


    # fig.supylabel('Z-Score Normalized PSD', fontsize=16)
    # fig.suptitle(title, fontsize='18')
    # fig.subplots_adjust(hspace=0, wspace=0.1)
    plt.tight_layout()

def psdEvolution(m_Data, s_windStep, v_FreqBands_Names, v_FreqBands, v_ChanNames, v_SessionTimes, title, relative):
    fig, ax = plt.subplots(len(v_FreqBands), 1, sharex=True, figsize=(8, 10))
    v_TimeArray = np.arange(0, len(m_Data[0][0])) * s_windStep
    AverageWind = 2
    colors = pl.cm.Reds(np.linspace(0, 1, len(m_Data) + 2))

    for i_band in range(len(v_FreqBands)):
        i_max = 0
        i_min = 1
        for i_chan in range(len(v_ChanNames)):
            v_Data = m_Data[i_chan][i_band]
            v_Data_Soft = f_averageMean(v_Data, AverageWind)

            if i_max < max(v_Data_Soft):
                i_max = max(v_Data_Soft)
            if i_min > min(v_Data_Soft):
                i_min = min(v_Data_Soft)

            ax[i_band].plot(v_TimeArray, v_Data_Soft, color=colors[i_chan + 2], label=v_ChanNames[i_chan])

        if i_band == 0:
            ax[i_band].legend(ncol=1, bbox_to_anchor=(1.125, -0.5), fancybox=True, shadow=True, fontsize=7)
            if len(v_SessionTimes) == 2:
                ax[i_band].text(v_SessionTimes[0] - 5, i_max * 1.5, 'Start', rotation=90,
                                fontsize=8)  # Se hubica un texto indicando la banda de seguridad ,bbox=bbox_props
                ax[i_band].text(v_SessionTimes[1] - 5, i_max * 1.5, 'End', rotation=90,
                                fontsize=8)  # Se hubica un texto indicando la banda de seguridad ,bbox=bbox_props
            else:
                ax[i_band].text(v_SessionTimes[0] - 5, i_max * 1.5, 'Start', rotation=90,
                                fontsize=8)  # Se hubica un texto indicando la banda de seguridad ,bbox=bbox_props

        if len(v_SessionTimes) == 2:
            ax[i_band].axvline(v_SessionTimes[0], linewidth=1, linestyle='--', color='k')
            ax[i_band].axvline(v_SessionTimes[1], linewidth=1, linestyle='--', color='k')
        else:
            ax[i_band].axvline(v_SessionTimes[0], linewidth=1, linestyle='--', color='k')

        ax[i_band].grid(linewidth=0.4, linestyle=':', color='k', which='both')
        ax[i_band].set_ylabel(f'{v_FreqBands_Names[i_band]}')
        ax[i_band].yaxis.tick_right()
        if i_min < 0:
            ax[i_band].set_ylim([i_min * 1.4, i_max * 1.4])
        else:
            ax[i_band].set_ylim([i_min * 0.6, i_max * 1.4])
        ax[i_band].yaxis.set_tick_params(labelsize=6)
        ax[i_band].set_xlabel('Time (s)')

    if relative:
        fig.supylabel('Relative PSD')
        fig.suptitle(title, fontsize='18')
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
    else:
        fig.supylabel('Z-Score Normalized PSD')
        fig.suptitle(title, fontsize='18')
        fig.subplots_adjust(hspace=0.1, wspace=0.1)


def TopoPlotMEANS(m_Data, relative, showPlots):
    # Args:
    #   m_Data (list): A list of data arrays, where each array represents the data for a test.
    #   relative (bool): A boolean flag indicating whether the data is relative or not.
    #   showPlots (bool): A boolean flag indicating whether to show the plots or not.

    # Import necessary libraries
    from visbrain.objects import TopoObj, ColorbarObj, SceneObj
    from matplotlib.colors import ListedColormap

    # Create a scene object for visualization
    sc = SceneObj(bgcolor='white', size=(450 * len(m_Data) + 45, 600))

    # Iterate over each test in the data
    for i_test in range(len(m_Data)):
        # Get the data for the current test
        v_Data = m_Data[i_test]

        # Calculate the data range for color mapping
        d_clim = np.max(np.abs([np.min(m_Data), np.max(m_Data)]))
        v_clim = [-d_clim, d_clim]

        # Set the parameters for the topography plot
        kw_top = dict(margin=30 / 100, chan_offset=(0.1, 0.1, 0.), chan_size=0, levels=7, line_width=12,
                      cmap='coolwarm', level_colors='k', clim=v_clim)

        # Set the parameters for the colorbar
        kw_cbar = dict(cbtxtsz=40, txtsz=40., width=.5, txtcolor='black', cbtxtsh=1.8, rect=(0., -1.65, 1., 3.5),
                       border=True)

        def freqECGBoxPlot(v_Data_Rat, b_plotLines):
            colors = pl.cm.Reds(np.linspace(0, 1, 9))
            colors[:, -1] = 0.65

            fig, ax = plt.subplots(figsize=(7, 5))
            if b_plotLines:
                ax.plot(np.arange(len(v_Data_Rat[0])) + 1, np.transpose(v_Data_Rat), color='gray', linewidth=1.5)
                ax.plot(np.arange(len(v_Data_Rat[0])) + 1, np.transpose(v_Data_Rat), '.k')
            [ax.spines[i].set_visible(False) for i in ['top', 'right', 'bottom', 'left']]
            ax.set_xticklabels(['', '', ''])
            ax.tick_params(axis="y", labelsize=18)
            # ax[0].spines['top'].set_visible(False)
            # ax[0].spines['right'].set_visible(False)
            # ax[0].spines['bottom'].set_linewidth(2)
            # ax[i_step].spines['left'].set_linewidth(2)

            dict_bp = ax.boxplot(v_Data_Rat,
                                 patch_artist=True,
                                 whiskerprops=dict(linewidth=1.25),
                                 capprops=dict(linewidth=1.25),
                                 medianprops=dict(linewidth=3, color='w'),
                                 widths=0.75)

            i_count = 2
            for patch in dict_bp['boxes']:
                patch.set(facecolor=colors[i_count * 2], linewidth=0)
                i_count += 1

        # Set the parameters for the title
        kw_title = dict(title_color='black', title_size=9.0, width_max=950)

        # Define the channel names
        ch_names = ['Fp1', 'Fp2', 'T3', 'T4', 'C3', 'C4', 'O1', 'O2']

        # Create the topography object
        t_obj = TopoObj('topo', v_Data, channels=ch_names, **kw_top)

        # Add the topography object to the scene
        sc.add_to_subplot(t_obj, row=0, col=i_test, zoom=1.075, **kw_title)

        # Set the label for the colorbar based on the 'relative' parameter
        if relative:
            clabel = ' %-Change '
        else:
            clabel = 'Z-Score'

    # Create the colorbar object
    cb_obj1 = ColorbarObj(t_obj, bw=10, cblabel=clabel, **kw_cbar)

    # Add the colorbar object to the scene
    sc.add_to_subplot(cb_obj1, row=0, col=(i_test + 1), width_max=200)

    # Preview the scene if 'showPlots' is True
    if showPlots:
        sc.preview()

    # Returns:
    #   sc (SceneObj): A scene object containing the topography plots and colorbar.
    return sc
def MyBoxPlot(v_Data_Rat, b_plotLines):
    # Generate a color map for the boxplot
    colors = pl.cm.Reds(np.linspace(0, 1, 9))
    colors[:, -1] = 0.65  # Adjust the alpha channel for transparency

    # Create a figure and axis for the boxplot
    fig, ax = plt.subplots(figsize=(int(2 * 3), 6))

    # Remove spines (axis lines) on all sides and hide x-axis labels
    [ax.spines[i].set_visible(False) for i in ['top', 'right', 'bottom', 'left']]
    ax.set_xticklabels(['', '', ''])

    # Set the font size for y-axis labels
    ax.tick_params(axis="y", labelsize=18)

    # Create a boxplot with custom properties
    dict_bp = ax.boxplot(v_Data_Rat,
                         patch_artist=True,
                         whiskerprops=dict(linewidth=1.25),
                         capprops=dict(linewidth=1.25),
                         medianprops=dict(linewidth=3, color='w'),
                         widths=0.75)

    i_count = 2
    for patch in dict_bp['boxes']:
        # Set the facecolor of each box using the predefined colors
        patch.set(facecolor=colors[i_count * 2], linewidth=0)
        i_count += 1

    if b_plotLines:
        # Plot gray lines and black points on top of the boxplot

        # Create a matrix with 10 rows, each containing [1, 2, 3]
        matrix = np.transpose([[1, 2, 3] for _ in range(len(v_Data_Rat))])
        # Generate random values in the range [-0.25, 0.25]
        random_factors = np.random.uniform(-0.25, 0.25, size=matrix.shape)
        matrix_with_random = matrix + random_factors
        # Add the random factors to the matrix
        ax.plot(matrix_with_random, np.transpose(v_Data_Rat), color='gray', linewidth=0.5)
        ax.plot(matrix_with_random, np.transpose(v_Data_Rat), 'ok')



def ECGPSDPlot(m_allDensity, v_freqRes, d_windSize=2):
    colors = pl.cm.Reds(np.linspace(0, 1, 12))

    d_indx = np.where(v_freqRes > 0.15)[0][0]
    fig, ax = plt.subplots(1, len(m_allDensity), figsize=(20, 7), sharey=True, sharex=True)

    for i_step in range(len(m_allDensity)):
        m_density = m_allDensity[i_step]
        ax[i_step].semilogy(v_freqRes, f_averageMean(m_density, d_windSize), color=colors[11])
        ax[i_step].fill_between(v_freqRes[:d_indx + 1], f_averageMean(m_density, d_windSize)[:d_indx + 1], 0,
                                color=colors[9],
                                alpha=0.3)
        ax[i_step].fill_between(v_freqRes[d_indx:], f_averageMean(m_density, d_windSize)[d_indx:], 0, color=colors[11],
                                alpha=0.3)
        ax[i_step].tick_params(axis='both', which='major', labelsize=20)
        ax[i_step].spines['top'].set_visible(False)
        ax[i_step].spines['right'].set_visible(False)
        ax[i_step].spines['bottom'].set_linewidth(2)
        ax[i_step].spines['left'].set_linewidth(2)
        ax[i_step].set_xlim([0.04, 0.4])
        # ax[i_step].set_ylim([0.0025, np.max(m_density)])

def PlotPoincareComparisons(v_tacoAll):
    """
    Plot Poincaré comparisons for a list of data sequences.

    Parameters:
    - v_tacoAll: A list of sequences to be plotted.

    This function generates a grid of subplots with Poincaré plots for each sequence in v_tacoAll.
    """
    # Create a color map for the plots
    colors = plt.cm.Reds(np.linspace(0, 1, 10))
    colors = ['#9D0E15', '#DC696C', '#FEAC9C']

    # Create a figure with 3 subplots
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for i_step in range(len(v_tacoAll)):
        # Add a diagonal reference line to each subplot
        ax[i_step].axline((0, 0), slope=1, color='black', linewidth=1, linestyle='--')

        # Get the current sequence
        v_itaco = v_tacoAll[i_step]

        # Scatter plot of the sequence with specified settings
        ax[i_step].scatter(v_itaco[:-1], v_itaco[1:], color=colors[i_step], alpha=0.5, linewidth=0,
                           marker='.',
                           s=35)

        # Plot a confidence ellipse for the data points
        confidence_ellipse(v_itaco[:-1], v_itaco[1:], ax[i_step], n_std=1.0)

        # Customize subplot appearance
        ax[i_step].spines['top'].set_visible(False)
        ax[i_step].spines['right'].set_visible(False)
        ax[i_step].spines['bottom'].set_linewidth(2)
        ax[i_step].spines['left'].set_linewidth(2)
        ax[i_step].set_xticks([-5, 0, 5])
        ax[i_step].set_yticks([-5, 0, 5])
        ax[i_step].set_xlim(-8, 8)
        ax[i_step].set_ylim(-8, 8)
        ax[i_step].set_aspect('equal', 'box')

    # Display the plot
    plt.show()

def EMGPSDplot(v_dataPlot, v_freq, m_allPathsMNF, d_softWind):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True, sharex=True)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    colors = pl.cm.Reds(np.linspace(0, 1, 12))
    v_colorIndx = [colors[5], colors[9], colors[11]]
    v_name = ['BS', 'MT', 'AF']
    for i_indx in range(len(v_dataPlot)):
        d_meanMNF = np.mean(m_allPathsMNF[:, i_indx])
        d_errorMNF = np.std(m_allPathsMNF[:, i_indx]) / np.sqrt(len(m_allPathsMNF[:, i_indx]))
        ax[i_indx].plot(v_freq, f_averageMean(v_dataPlot[i_indx] * 100, d_softWind), color=v_colorIndx[i_indx],
                        linewidth=1.5,
                        label=v_name[i_indx])

        ax[i_indx].fill_between(v_freq, f_averageMean(v_dataPlot[i_indx] * 100, d_softWind), 0, color=v_colorIndx[i_indx],
                                alpha=0.5)
        ax[i_indx].axvline(d_meanMNF, color='k', linewidth=2)
        # ax[i_indx].axvline(d_meanMNF-d_errorMNF, color=v_colorIndx[i_indx], linewidth=1)
        # ax[i_indx].axvline(d_meanMNF+d_errorMNF, color=v_colorIndx[i_indx], linewidth=1)
        # ax[i_indx].fill_between([d_meanMNF-d_errorMNF,d_meanMNF+d_errorMNF],2.6, color='k', alpha=1)

        # ax[i_indx].legend()
        ax[i_indx].tick_params(axis='both', which='major', labelsize=16)
        ax[i_indx].spines['top'].set_visible(False)
        ax[i_indx].spines['right'].set_visible(False)
        ax[i_indx].spines['bottom'].set_linewidth(2)
        ax[i_indx].spines['left'].set_linewidth(2)
        ax[i_indx].set_ylim(0.1, 2.5)
        ax[i_indx].set_xlim(20, 200)