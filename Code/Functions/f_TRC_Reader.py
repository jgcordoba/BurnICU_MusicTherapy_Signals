import struct as st
import numpy as np
import os


class CustomError(Exception):
    pass


def f_GetTRCHeader(pstr_File):
    s_fileID = open(pstr_File, 'rb');
    s_Status = s_fileID.seek(175, 0)
    s_HeaderType = s_fileID.read(1)
    s_HeaderType = int(st.unpack('B' * int(len(s_HeaderType)), s_HeaderType)[0])

    if s_HeaderType != 4:
        raise CustomError("'[f_GetTRCHeader] - ERROR: Incorrect Type TRC Header!'")

    try:
        st_Header = {}

        s_Status = s_fileID.seek(0, 0)
        st_Header['Title'] = str(s_fileID.read(31))[1::]

        s_Status = s_fileID.seek(32, 0)
        st_Header['Lab'] = str(s_fileID.read(31))[1::]

        s_Status = s_fileID.seek(128, 0)
        act_read = s_fileID.read(3)
        st_Header['RecDate'] = np.array(st.unpack('B' * int(len(act_read)), act_read))
        st_Header['RecDate'][2] = st_Header['RecDate'][2] + 1900

        s_Status = s_fileID.seek(131, 0)
        act_read = s_fileID.read(3)
        st_Header['RecTime'] = np.array(st.unpack('B' * int(len(act_read)), act_read))

        s_Status = s_fileID.seek(136, 0)
        act_read = s_fileID.read(2)
        st_Header['AcqType'] = int(st.unpack('H' * int(len(act_read) / 2), act_read)[0])

        s_Status = s_fileID.seek(138, 0)
        act_read = s_fileID.read(4)
        st_Header['DataStart'] = int(st.unpack('L' * int(len(act_read) / 4), act_read)[0])

        s_Status = s_fileID.seek(142, 0)
        act_read = s_fileID.read(2)
        st_Header['ChanNum'] = int(st.unpack('H' * int(len(act_read) / 2), act_read)[0])

        s_Status = s_fileID.seek(144, 0)
        act_read = s_fileID.read(2)
        st_Header['Multiplex'] = int(st.unpack('H' * int(len(act_read) / 2), act_read)[0])

        s_Status = s_fileID.seek(146, 0)
        act_read = s_fileID.read(2)
        s_MinRate = int(st.unpack('H' * int(len(act_read) / 2), act_read)[0])
        st_Header['RecFreq'] = s_MinRate

        s_Status = s_fileID.seek(148, 0)
        act_read = s_fileID.read(2)
        st_Header['NumBytes'] = int(st.unpack('H' * int(len(act_read) / 2), act_read)[0])

        st_Code = {}

        s_Status = s_fileID.seek(176, 0)
        st_Code['Name'] = str(s_fileID.read(8))[1::]

        s_Status = s_fileID.seek(184, 0)
        act_read = s_fileID.read(4)
        st_Code['StartOf'] = int(st.unpack('L' * int(len(act_read) / 4), act_read)[0])

        s_Status = s_fileID.seek(188, 0)
        act_read = s_fileID.read(4)
        st_Code['Lenght'] = int(st.unpack('L' * int(len(act_read) / 4), act_read)[0])

        st_Elec = {}

        s_Status = s_fileID.seek(192, 0)
        st_Elec['Name'] = str(s_fileID.read(8))[1::]

        s_Status = s_fileID.seek(200, 0)
        act_read = s_fileID.read(4)
        st_Elec['StartOf'] = int(st.unpack('L' * int(len(act_read) / 4), act_read)[0])

        s_Status = s_fileID.seek(204, 0)
        act_read = s_fileID.read(4)
        st_Elec['Lenght'] = int(st.unpack('L' * int(len(act_read) / 4), act_read)[0])

        s_Status = s_fileID.seek(st_Code['StartOf'], 0);
        act_read = s_fileID.read(st_Code['Lenght'] * 2)
        v_ElectCh = np.array(st.unpack('H' * int(len(act_read) / 2), act_read))
        s_Idx = np.where(v_ElectCh == 0)[0][0]
        v_ElectCh = v_ElectCh[0:s_Idx]

    except:
        raise CustomError('[f_GetTRCHeader] - ERROR: fseek error!')

    v_ElectOffsetData = np.arange(st_Elec['StartOf'], st_Elec['StartOf'] + st_Elec['Lenght'] - 1, 128)
    v_ElectOffsetData = v_ElectOffsetData[v_ElectCh]
    v_ElectData = [None] * st_Header['ChanNum']
    st_Header['Ch'] = [None] * st_Header['ChanNum']

    s_CurrVal = -1;
    for s_CurrPos in v_ElectOffsetData:
        st_Info = {}
        s_Status = s_fileID.seek(s_CurrPos, 0)
        s_ElecStatus = s_fileID.read(1)
        s_ElecStatus = int(st.unpack('B' * int(len(s_ElecStatus)), s_ElecStatus)[0])
        if s_ElecStatus == 0:
            raise CustomError('[f_GetTRCHeader] - ERROR: fseek ElectStatus error!')
        else:
            s_CurrVal += 1

        s_Status = s_fileID.seek(s_CurrPos + 1, 0)
        s_ElecStatus = s_fileID.read(1)
        st_Info['Type'] = int(st.unpack('B' * int(len(s_ElecStatus)), s_ElecStatus)[0])

        s_Status = s_fileID.seek(s_CurrPos + 2, 0)
        s_ElecStatus = s_fileID.read(6)
        str_InLabelPos = str(s_ElecStatus, 'utf-8')
        str_InLabelPos = str(str_InLabelPos.split('\x00')[0])

        s_Status = s_fileID.seek(s_CurrPos + 8, 0)
        s_ElecStatus = s_fileID.read(6)
        str_InLabelNeg = str(s_ElecStatus, 'utf-8')
        str_InLabelNeg = str(str_InLabelNeg.split('\x00')[0])

        st_Info['Ch'] = str_InLabelPos
        st_Info['Label'] = str_InLabelPos + '_' + str_InLabelNeg

        st_Conv = {}
        s_Status = s_fileID.seek(s_CurrPos + 14, 0)
        act_read = s_fileID.read(4)
        st_Conv['LogicMin'] = int(st.unpack('i' * int(len(act_read) / 4), act_read)[0])

        s_Status = s_fileID.seek(s_CurrPos + 18, 0)
        act_read = s_fileID.read(4)
        st_Conv['LogicMax'] = int(st.unpack('i' * int(len(act_read) / 4), act_read)[0])

        s_Status = s_fileID.seek(s_CurrPos + 22, 0)
        act_read = s_fileID.read(4)
        st_Conv['LogicGnd'] = int(st.unpack('i' * int(len(act_read) / 4), act_read)[0])

        s_Status = s_fileID.seek(s_CurrPos + 26, 0)
        act_read = s_fileID.read(4)
        st_Conv['PhysicMin'] = int(st.unpack('i' * int(len(act_read) / 4), act_read)[0])

        s_Status = s_fileID.seek(s_CurrPos + 30, 0)
        act_read = s_fileID.read(4)
        st_Conv['PhysicMax'] = int(st.unpack('i' * int(len(act_read) / 4), act_read)[0])

        s_Status = s_fileID.seek(s_CurrPos + 34, 0)
        act_read = s_fileID.read(1)
        s_Units = int(st.unpack('B' * int(len(act_read)), act_read)[0])

        if s_Units == -1:
            st_Info['Units'] = 'nV'
        elif s_Units == 0:
            st_Info['Units'] = 'uV'
        elif s_Units == 1:
            st_Info['Units'] = 'mV'
        elif s_Units == 2:
            st_Info['Units'] = 'V'
        elif s_Units == 100:
            st_Info['Units'] = '%'
        elif s_Units == 101:
            st_Info['Units'] = 'bpm'
        elif s_Units == 102:
            st_Info['Units'] = 'Adim'
        else:
            raise CustomError('[f_GetTRCHeader] - ERROR: Electrode units unknown!')

        s_Status = s_fileID.seek(s_CurrPos + 44, 0)
        act_read = s_fileID.read(2)
        s_SamplingCoef = int(st.unpack('H' * int(len(act_read) / 2), act_read)[0])

        st_Info['Sampling'] = s_MinRate * s_SamplingCoef
        st_Info['Conversion'] = st_Conv

        v_ElectData[s_CurrVal] = st_Info
        st_Header['Ch'][s_CurrVal] = st_Info['Ch']

    v_ElectData = v_ElectData[0:s_CurrVal + 1]
    st_Header['ElectData'] = v_ElectData

    try:
        s_Status = s_fileID.seek(st_Header['DataStart'], 0)
        s_DatStart = s_fileID.tell()
        s_Status = s_fileID.seek(0, 2)
        s_DatEnd = s_fileID.tell()
    except:
        raise CustomError('[f_GetTRCHeader] - ERROR: fseek error!')

    st_Header['s_SampleNum'] = int((s_DatEnd - s_DatStart) / (st_Header['NumBytes'] * st_Header['ChanNum']));
    s_fileID.close()
    return (st_Header)


def f_LoadData(pstr_File, ps_Start, ps_FirstInd, ps_LastInd, s_Bytes):
    v_Data = None
    s_Size = (ps_LastInd - ps_FirstInd) + 1
    if ps_FirstInd < 1 or (ps_FirstInd > 0 and s_Size < 1):
        return None

    s_fileID = open(pstr_File, 'rb');

    if s_Bytes == 1:
        str_Type = 'B'
        s_Bytes_size = 1
    elif s_Bytes == 2:
        str_Type = 'H'
        s_Bytes_size = 2
    elif s_Bytes == 3:
        str_Type = 'I'
        s_Bytes_size = 4

    s_Status = s_fileID.seek(ps_Start, 0)
    if ps_FirstInd > 1:
        s_fileID.seek(s_Bytes * (ps_FirstInd - 1), 1);

    if ps_FirstInd > 0:
        act_read = s_fileID.read()
        v_Data = np.array(st.unpack(str_Type * int(len(act_read) / s_Bytes_size), act_read))
    s_fileID.close()
    return v_Data


def f_GetSignalsTRC(pstr_FullPath, pstr_EEGSigStr=[], ps_FirstSam=[], ps_LastSam=[], ps_ScaleData=1):
    st_Header = f_GetTRCHeader(pstr_FullPath)

    if type(pstr_EEGSigStr) == str:
        v_EEGSigCell = [pstr_EEGSigStr]
    elif len(pstr_EEGSigStr) == 0:
        v_EEGSigCell = st_Header['Ch']
    else:
        v_EEGSigCell = pstr_EEGSigStr

    [_, str_FileNamePrefix] = os.path.split(pstr_FullPath)[::]
    str_FileNamePrefix = str_FileNamePrefix.split('.')[0]

    s_MaxElem = 50 * (10 ** 12)
    s_MaxElem = s_MaxElem / st_Header['NumBytes']
    s_MaxElem = int(s_MaxElem / st_Header['ChanNum']) * st_Header['ChanNum']

    if len(ps_FirstSam) == 0:
        s_FirstInd = 1
    else:
        s_FirstInd = (ps_FirstSam - 1) * st_Header['ChanNum'] + 1

    if len(v_EEGSigCell) == 0:
        v_EEGSigCell = st_Header['Ch'];

    v_ChIndexes = [st_Header['Ch'].index(i) for i in v_EEGSigCell if i in st_Header['Ch']]

    if len(v_ChIndexes) == 0:
        raise CustomError('[f_GetSignalsTRC] - ERROR: no channel found!')

    if len(ps_LastSam) == 0:
        s_LastInd = s_FirstInd + s_MaxElem - 1
    else:
        s_LastInd = ps_LastSam * st_Header.ChanNum;
        if (s_LastInd - s_FirstInd) + 1 > s_MaxElem:
            s_LastInd = s_FirstInd + s_MaxElem - 1

    m_FileSig = f_LoadData(pstr_FullPath, st_Header['DataStart'], s_FirstInd, s_LastInd, st_Header['NumBytes'])
    m_FileSig = np.reshape(m_FileSig, (st_Header['ChanNum'], -1), order='F')
    m_Signal = m_FileSig[v_ChIndexes]

    if ps_ScaleData:
        for i in range(len(v_ChIndexes)):
            st_Scale = st_Header['ElectData'][v_ChIndexes[i]];
            m_Signal[i, :] = ((m_Signal[i, :] - st_Scale['Conversion']['LogicGnd']) \
                              / (st_Scale['Conversion']['LogicMax'] - st_Scale['Conversion']['LogicMin'] + 1)) \
                             * (st_Scale['Conversion']['PhysicMax'] - st_Scale['Conversion']['PhysicMin'])

    return (m_Signal)
