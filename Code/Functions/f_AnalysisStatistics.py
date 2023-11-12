import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.multitest import fdrcorrection
import copy


def f_PermTest(pv_Dist1, pv_Dist2, ps_Alpha=0.05, ps_PermNum=1600):
    '''
    function f_PermTest2.m

    Description:

    Inputs:
    pv_Dist1: First distribution
    pv_Dist2: Second distribution
    ps_Alpha: Performs the test at the (100 * ps_Alpha) significance level

    Outputs:
    s_H: 1 indicates a rejection of the null hypothesis; 0 otherwise [Mean, Median, MeanDev, Tdis]
    s_P: p-value
    v_TDist: T values obtained for all permutations
    '''

    pv_Dist1 = pv_Dist1.flatten()
    pv_Dist2 = pv_Dist2.flatten()

    s_LenX = len(pv_Dist1)
    s_LenY = len(pv_Dist2)

    v_ArrayTemp = np.concatenate((pv_Dist1, pv_Dist2))
    # v_ArrayTemp = np.vstack([pv_Dist1, pv_Dist2]).ravel('F')
    s_LenDbl = s_LenX + s_LenY

    v_Ind = np.zeros((s_LenDbl, ps_PermNum))
    v_MeanDiffDist = np.zeros((ps_PermNum, 1))
    v_MedianDiffDist = np.zeros((ps_PermNum, 1))
    v_MeanDevDist = np.zeros((ps_PermNum, 1))
    v_TDist = np.zeros((ps_PermNum, 1))

    v_IndOrd = np.zeros((s_LenDbl, 1))
    v_IndAux = np.random.permutation(s_LenDbl)
    v_IndOrd[v_IndAux[1:s_LenX]] = 1

    for s_PermCounter in range(ps_PermNum):
        while 1:
            # print('a')
            v_IndOrd = np.zeros((s_LenDbl, 1))
            v_IndAux = np.random.permutation(s_LenDbl)
            v_IndOrd[v_IndAux[0:s_LenX]] = 1
            v_IndOrd = v_Ind[:, 0: s_PermCounter] * np.tile(v_IndOrd, (1, s_PermCounter))
            v_IndOrd = np.sum(v_IndOrd, 0)
            if len(np.where(v_IndOrd == s_LenX)[0]) >= 1:
                continue
            v_IndAux = v_IndAux[0:s_LenX]
            break

        if s_PermCounter + 1 == 1 or (s_PermCounter + 1) % 400 == 0:
            print(f'[f_PermTest2] - Permutation {s_PermCounter + 1}/{ps_PermNum}')
        v_Ind[v_IndAux, s_PermCounter] = 1

        s_MeanX = np.mean(v_ArrayTemp[v_Ind[:, s_PermCounter] == 1], 0)
        s_MeanY = np.mean(v_ArrayTemp[v_Ind[:, s_PermCounter] == 0], 0)
        v_MeanDiffDist[s_PermCounter] = s_MeanX - s_MeanY

        s_MedianX = np.median(v_ArrayTemp[v_Ind[:, s_PermCounter] == 1], 0)
        s_MedianY = np.median(v_ArrayTemp[v_Ind[:, s_PermCounter] == 0], 0)
        v_MedianDiffDist[s_PermCounter] = s_MeanX - s_MeanY

        v_MeanDevDist[s_PermCounter] = np.mean(abs(v_ArrayTemp[v_Ind[:, s_PermCounter] == 1] - s_MedianX), 0) \
                                       / np.mean(abs(v_ArrayTemp[v_Ind[:, s_PermCounter] == 0] - s_MedianY), 0)

        s_VarX = np.var(v_ArrayTemp[v_Ind[:, s_PermCounter] == 1])
        s_VarY = np.var(v_ArrayTemp[v_Ind[:, s_PermCounter] == 0])

        v_TDist[s_PermCounter] = (v_MeanDiffDist[s_PermCounter]) / np.sqrt((s_VarX / s_LenX) + (s_VarY / s_LenY))

    v_MeanDiffDistOrg = np.sort(v_MeanDiffDist[:, 0])
    # v_MeanDiffDist = np.sort(abs(v_MeanDiffDist[:, 0]))
    v_MeanDiffDist = np.sort(abs(v_MeanDiffDist[:, 0]))

    v_MedianDiffDist = np.sort(abs(v_MedianDiffDist[:, 0]))
    v_MeanDevDist = np.sort(abs(v_MeanDevDist[:, 0]))
    v_TDist = np.sort(abs(v_TDist[:, 0]))

    s_MeanX = np.mean(v_ArrayTemp[0:s_LenX])
    s_MeanY = np.mean(v_ArrayTemp[s_LenX::])
    s_MeanDiffRef = s_MeanX - s_MeanY

    s_MedianX = np.median(v_ArrayTemp[0:s_LenX])
    s_MedianY = np.median(v_ArrayTemp[s_LenX::])
    s_MedianDiffRef = s_MedianX - s_MedianY

    s_MeanDevRef = np.mean(abs(v_ArrayTemp[0: s_LenX] - s_MedianX)) / np.mean(abs(v_ArrayTemp[s_LenX::] - s_MedianY))

    s_VarX = np.var(v_ArrayTemp[0:s_LenX])
    s_VarY = np.var(v_ArrayTemp[s_LenX::])
    s_TRef = (s_MeanX - s_MeanY) / np.sqrt((s_VarX / s_LenX) + (s_VarY / s_LenY))

    s_TotalStats = 4
    s_StatCounter = 0
    v_H = np.zeros(s_TotalStats)
    v_P = np.zeros(s_TotalStats)

    s_MeanDiffRef = np.abs(s_MeanDiffRef)
    s_MedianDiffRef = np.abs(s_MedianDiffRef)
    s_MeanDevRef = np.abs(s_MeanDevRef)
    s_TRef = np.abs(s_TRef)

    v_StatsDist = [v_MeanDiffDist, v_MedianDiffDist, v_MeanDevDist, v_TDist]
    v_StatsRef = [s_MeanDiffRef, s_MedianDiffRef, s_MeanDevRef, s_TRef]

    for s_StatCounter in range(len(v_StatsDist)):
        s_H = 0
        s_P = np.where(v_StatsDist[s_StatCounter] >= v_StatsRef[s_StatCounter])[0]

        if len(s_P) == 0:
            s_P = 1 / (len(v_StatsDist[s_StatCounter]) + 1)
            s_H = 1
        else:
            s_P = s_P[0] + 1
            s_P = (len(v_StatsDist[s_StatCounter]) + 1 - s_P) / len(v_StatsDist[s_StatCounter])

            if s_P <= ps_Alpha:
                s_H = 1

        v_H[s_StatCounter] = s_H
        v_P[s_StatCounter] = s_P

    return (v_H == 1), v_P, v_StatsRef,


def confidence_interval_t(sorted_permuted_stats, alpha):
    """
    Calculate a confidence interval for a t-distribution.

    Args:
    sorted_permuted_stats (array-like): Sorted list of permuted test statistics.
    alpha (float): Significance level (e.g., 0.05 for a 95% confidence interval).

    Returns:
    confidence_interval (tuple): Tuple containing the lower and upper bounds of the confidence interval.
    """

    # Calculate the degrees of freedom (df) for the t-distribution
    df = len(sorted_permuted_stats) - 1

    # Find the critical t-values based on the alpha/2 and 1-alpha/2 percentiles
    t_alpha_over_2 = stats.t.ppf(1 - alpha / 2, df)
    t_1_minus_alpha_over_2 = stats.t.ppf(alpha / 2, df)

    # Calculate the confidence interval bounds
    lower_bound = sorted_permuted_stats[int(alpha / 2 * len(sorted_permuted_stats))]
    upper_bound = sorted_permuted_stats[int((1 - alpha / 2) * len(sorted_permuted_stats))]

    return (lower_bound, upper_bound)


# Example usage:
# sorted_permuted_stats is a list of permuted test statistics, and alpha is the significance level.
# Replace these with your actual data and significance level.


def extractPermTestData(m_Data1, m_Data2, v_FreqBands_Names, v_ChanNames, ps_PermNum):
    d_count = 1
    d_cols = 3
    m_PermTest = np.zeros((len(v_ChanNames), len(v_FreqBands_Names) * d_cols))
    v_FrameNames = []
    for i_band in range(len(v_FreqBands_Names)):
        for i_chann in range(len(v_ChanNames)):
            print(f'##################################################')
            print(
                f'Processing channel {v_ChanNames[i_chann]} in band {v_FreqBands_Names[i_band]} | {d_count}/{len(v_ChanNames) * len(v_FreqBands_Names)} ')
            print(f'##################################################')
            v_chan1 = m_Data1[:, i_chann, i_band]
            v_chan2 = m_Data2[:, i_chann, i_band]

            s_hyp, s_pvalues, v_StatsRef = f_PermTest(v_chan1, v_chan2, ps_PermNum=ps_PermNum)
            # Probar otro permtest

            m_PermTest[i_chann, (i_band * d_cols)] = s_hyp[0]
            m_PermTest[i_chann, (i_band * d_cols) + 1] = s_pvalues[0]
            m_PermTest[i_chann, (i_band * d_cols) + 2] = np.mean(v_chan1) - np.mean(v_chan2)

            d_count += 1

        s_pvalueName = f'{v_FreqBands_Names[i_band]} - Pvalue'
        v_pvalueValues = np.array(m_PermTest[:, 1 + (i_band * d_cols)])
        v_pvalueValuesFDR = fdrcorrection(v_pvalueValues, 0.05, method='indep')
        m_PermTest[:, (i_band * d_cols)] = v_pvalueValuesFDR[0]
        m_PermTest[:, (i_band * d_cols) + 1] = v_pvalueValuesFDR[1]

        v_FrameNames.append(f'{v_FreqBands_Names[i_band]} - FDR H0')
        v_FrameNames.append(f'{v_FreqBands_Names[i_band]} - FDR Pvalue')
        v_FrameNames.append(f'{v_FreqBands_Names[i_band]} - Mean Diff')

    return pd.DataFrame(m_PermTest, columns=v_FrameNames, index=v_ChanNames)

import numpy as np

