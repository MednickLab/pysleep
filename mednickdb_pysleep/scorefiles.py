import pandas as pd
import re
import math
import xml.etree.ElementTree as ET
import numpy as np
import scipy.interpolate
from scipy.io import loadmat
from datetime import datetime, timedelta
from mednickdb_pysleep import pysleep_defaults, pysleep_utils
import wonambi.ioeeg.edf as wnbi
import mne
from typing import Tuple
import numpy as np


def extract_epochstages_from_scorefile(file, stagemap) -> Tuple[np.ndarray, float, datetime]:
    """
    Extract score data from file, and pass to the appropriate scoring reading/conversion function
    :param file: file to extract scoring from
    :param stagemap: stagemap to use for convert
    :return: parsed data, epoch offset (difference between start of scoring and start of recording), startime (startime as a datetime)
    """
    if file.endswith("xls") or file.endswith("xlsx") or file.endswith(".csv"):
        xl = pd.ExcelFile(file)
        if 'GraphData' in xl.sheet_names:  # Then we have a mednick type scorefile
            epochdict = _parse_grass_scorefile(file)
        else:
            raise ValueError('Unknown xlsx scorefile and not able to be parsed.')

    # these are the scoring files (txt)
    elif file.endswith(".txt"):
        file_data = open(file, 'r')
        epochdict = _txtfile_select_parser_function(file_data)  # This determines which type of txt file is present

    # EDF+ files which contain scoring data
    elif file.endswith(".edf"):
        epochdict = _parse_edf_scorefile(file, stagemap)

    elif file.endswith('.xml'):  # Some score file were xml...
        epochdict = _nsrr_xml_parse(file, stagemap)

    elif file.endswith('.mat'):  # assume that all .mat are hume type
        epochdict = _hume_parse(file)

    elif file.endswith('.vrmk'):  # assume that all .mat are hume type
        epochdict = _parse_vrmk_scorfile(file, stagemap)

    else:
        raise ValueError('ScoreFile not able to be parsed.')

    #Do stagemap
    epochstages = [stagemap[str(x)] if str(x) in stagemap else pysleep_defaults.unknown_stage for x in epochdict['epochstages']]
    if all([stage == pysleep_defaults.unknown_stage for stage in epochdict['epochstages']]):
        raise ValueError('All stages are unknown, this is probably an error, maybe the stagemap was not found. Make sure the study name is correct.')

    return epochstages, epochdict['epochoffset'], epochdict['starttime'] if 'starttime' in epochdict else None


def score_wake_as_waso_wbso_wase(epochstages, wake_base='wake',
                                 waso=pysleep_defaults.waso_stage,
                                 wbso=pysleep_defaults.wbso_stage,
                                 wase=pysleep_defaults.wase_stage,
                                 sleep_stages=pysleep_defaults.sleep_stages):
    """
    Convert all the wake before sleep onset to wbso, all the after the last epoch of sleep to wase, and the rest to waso
    :param epochstages: epoch stages
    :param wake_base: name of wake before conversion to wbso, waso, wase
    :param waso: name of waso
    :param wbso: name of wbso
    :param wase: name of wase
    :param sleep_stages: list of stages considered as sleep
    :return: converted epochstages
    """
    epochstages = np.array(epochstages)
    epochs_of_sleep = [1 if epoch in sleep_stages else 0 for epoch in epochstages]
    trans_to_sleep = np.where(np.diff(epochs_of_sleep) == 1)[0]
    trans_from_sleep = np.where(np.diff(epochs_of_sleep) == -1)[0]
    epochs_of_wake = np.where([1 if epoch == wake_base else 0 for epoch in epochstages])[0]
    epochstages[epochs_of_wake] = wbso
    if len(trans_to_sleep) > 0:
        epochs_of_wbso = epochs_of_wake[epochs_of_wake <= trans_to_sleep[0]]
        epochstages[epochs_of_wbso] = wbso
        epochs_of_waso = epochs_of_wake[epochs_of_wake > trans_to_sleep[0]]
        epochstages[epochs_of_waso] = waso
    if len(trans_from_sleep) > 0:
        epochs_of_wase = epochs_of_wake[trans_from_sleep[-1] < epochs_of_wake]
        epochstages[epochs_of_wase] = wase
    return epochstages.tolist()


def _hume_parse(file, epoch_len=pysleep_defaults.epoch_len):
    """
    Parse HUME type matlab file
    :param file: file to parse
    :param epoch_len: length of an epoch in seconds
    :return: dict with epochstage, epochoffset, starttime keys
    """

    hume_dict = hume_matfile_loader(file)

    dict_obj = {"epochstages": hume_dict['stages'], 'epochoffset':0}

    if 'win' in hume_dict:
        assert epoch_len == hume_dict['win'], "Hume epoch_len is different to requested epoch_len TODO convert."

    if 'recStart' in hume_dict:
        dict_obj['starttime'] = mat_datenum_to_py_datetime(hume_dict['recStart'])

    if 'lightsOFF' in hume_dict:
        dict_obj['epochoffset'] = (mat_datenum_to_py_datetime(hume_dict['lightsOFF']) - mat_datenum_to_py_datetime(hume_dict['recStart'])).seconds

    if 'stageTime' in hume_dict:
        dict_obj['epochoffset'] += hume_dict['win'] * hume_dict['stageTime'][0]

    return dict_obj


def _read_edf_annotations(fname, annotation_format="edf/edf+"):
    """
    Read EDF file annotations.
    # CODE PROVIDED BY M N E TO READ KEMP FILES
    :param fname: Path to file.
    :param annotation_format: one of ['edf/edf+', 'edf++']
    :return: annotations to be converted to epochstage format
    """
    with open(fname, 'r', encoding='utf-8',
              errors='ignore') as annotions_file:
        tal_str = annotions_file.read()

    if "edf" in annotation_format:
        if annotation_format == "edf/edf+":
            exp = '(?P<onset>[+\-]\d+(?:\.\d*)?)' + \
                  '(?:\x15(?P<duration>\d+(?:\.\d*)?))?' + \
                  '(\x14(?P<description>[^\x00]*))?' + '(?:\x14\x00)'

        elif annotation_format == "edf++":
            exp = '(?P<onset>[+\-]\d+.\d+)' + \
                  '(?:(?:\x15(?P<duration>\d+.\d+)))' + \
                  '(?:\x14\x00|\x14(?P<description>.*?)\x14\x00)'

        annot = [m.groupdict() for m in re.finditer(exp, tal_str)]
        good_annot = pd.DataFrame(annot)
        good_annot = good_annot.query('description != ""').copy()
        good_annot.loc[:, 'duration'] = good_annot['duration'].astype(float)
        good_annot.loc[:, 'onset'] = good_annot['onset'].astype(float)
    else:
        raise ValueError('Type not supported')

    return good_annot


def _resample_to_new_epoch_len(annot, new_epoch_len=pysleep_defaults.epoch_len):
    """
    Some scorefiles have 20 second epochs, this will resample to some other length (30 generally).
    :param annot: a dataframe with onset, duration and description columns.
    :param new_epoch_len: the new epoch length to resample to.
    :return: A dataframe in the same format as the annot input, with the resampled epoch len
    """
    # This is coupled with stagemaps because we need some way of removing the non stage entries of the annotations
    stage_map_dict = {v:k for k,v in enumerate(annot['description'].unique())}
    stage_rev_map_dict = {k:v for k, v in enumerate(annot['description'].unique())}
    annot.loc[:, 'description'] = annot.loc[:, 'description'].map(stage_map_dict)

    sub_second_offset = annot['onset'].values[0] - int(annot['onset'].values[0])
    onset = list(annot['onset'].astype(int))
    window_onsets = np.array((range(onset[0], onset[-1], new_epoch_len))) + sub_second_offset
    hypno = scipy.interpolate.interp1d(annot['onset'], annot['description'], kind='zero')
    window_stages = [stage_rev_map_dict[stage] for stage in hypno(window_onsets)]
    durations = [new_epoch_len for i in window_stages]
    return pd.DataFrame({'onset': list(window_onsets), 'description': window_stages, 'duration': durations})


def _parse_edf_scorefile(path, stage_map_dict, epoch_len=pysleep_defaults.epoch_len):
    """
    Load edf file extract relevant meta data including epoch stages.
    :param path: file to parse
    :param stage_map_dict: stagemap
    :return:
    """

    dictObj = {}

    edf = wnbi.Edf(path)
    dictObj['starttime'] = edf.hdr['start_time']

    try: #type1
        annot = _read_edf_annotations(path)
    except ValueError: #type3
        annot = _read_edf_annotations(path, annotation_format="edf++") #FIXME how to get edf++ ..?

    assert annot.shape[0] > 0, "no sleep stage annotations found, this is probably an error"

    valid_stages = annot['description'].isin(stage_map_dict.keys())
    annot = annot.loc[valid_stages, :]

    annot = _resample_to_new_epoch_len(annot, epoch_len)
    dictObj['epochstages'] = list(annot['description'].values)
    dictObj['epochoffset'] = annot['onset'].values[0]

    return dictObj


def _xml_repeater(node):
    """
    Helper to parse xml
    :param node: input xml to parse
    :return:
    """
    temp = {}
    for child in node:
        J = (_xml_repeater(child))
        if len(J) != 0:
            for key in J.keys():
                if key in temp.keys():
                    temp[key].append(J[key])
                else:  # if J[key] != None:
                    temp[key] = []
                    temp[key].append(J[key])
        dict = {child.tag: child.text}
        if (child.text != '\n'):
            for key in dict.keys():
                if key in temp.keys():
                    temp[key].append(dict[key])
                else:  # if dict[key] != None:
                    temp[key] = []
                    temp[key].append(dict[key])
    return temp


def _nsrr_xml_parse(file, stage_map_dict, epoch_len=pysleep_defaults.epoch_len):
    """
    Parsing for NSRR formated xml scorefiles
    :param file: file object to parse
    :param stage_map_dict: stage map
    :return: dict with epochstage, etc
    """

    # characters that we will strip
    STRIP = "' ', ',', '\'', '(', '[', '{', ')', '}', ']'"

    tree = ET.parse(file)
    root = tree.getroot()
    dict_xml = _xml_repeater(root)
    temp_dict = {'description': [], 'onset': [], 'duration': []}

    for key in dict_xml.keys():
        needToStrip = str(dict_xml[key]).split(',')
        for i in range(len(needToStrip)):
            needToStrip[i] = needToStrip[i].lstrip(STRIP).rstrip(STRIP)
        dict_xml[key] = needToStrip

    # Need to change this maybe	right now only includes the important stuff
    # Need to fix the time
    # get dictionary with sleepevent, start time, and duration
    # need to expand so it will see every 30 sec and have it in epoch time
    for i in range(len(dict_xml['EventType'])):
        if "Stages" in dict_xml['EventType'][i]:
            temp_dict['description'].append(dict_xml['EventConcept'][i].split('|')[0])
            temp_dict['duration'].append(float(dict_xml['Duration'][i]))
            temp_dict['onset'].append(float(dict_xml['Start'][i]))
    annot = pd.DataFrame(temp_dict)
    valid_stages = annot['description'].isin(stage_map_dict.keys())
    assert valid_stages.any(), "No valid stages, are you sure the stage map in study settings is correct?"
    annot = annot.loc[valid_stages, :]
    annot_resampled = _resample_to_new_epoch_len(annot, epoch_len)

    return_dict = {}
    return_dict['epochstages'] = list(annot_resampled['description'].values)
    return_dict['epochoffset'] = annot_resampled['onset'].values[0]
    return_dict['starttime'] = datetime.strptime(dict_xml['ClockTime'][0].split(' ')[-1], '%H.%M.%S')

    return return_dict


def _txtfile_select_parser_function(file):
    """
    Returns an integer determing which parse method to use
       if found == 0 file contain only s and 0s
       if found == 1 file contain latency and type(sleep stage mode)
       if found == 2 file contain sleep stage , and time
    :param file: file to decide for
    :return: int to select txt file parse method
    """

    parsers = {0: _parse_basic_txt_scorefile,
               1: _parse_lat_type_txt_scorefile,
               2: _parse_full_type_txt_scorefile}

    key_words = ["latency", "RemLogic"]
    found = 0
    firstline = file.readline()
    file.seek(0)
    for count in range(len(key_words)):
        if firstline.find(key_words[count]) != -1:
            found = count + 1

    try:
        return parsers[found](file)
    except KeyError:
        raise ValueError('txt ScoreFile not able to be parsed.')


def _parse_basic_txt_scorefile(file, epoch_len=pysleep_defaults.epoch_len):
    """
    Parse the super basic sleep files from Dinklmann
    No starttime is available.
    :param file:
    :return:
    """
    dict_obj = {"epochstages": [], "epochoffset": 0}
    for line in file:
        temp = line.split(' ')
        temp = temp[0].split('\t')
        temp[0] = temp[0].strip('\n')
        dict_obj["epochstages"].append(temp[0])
    return dict_obj


def _parse_lat_type_txt_scorefile(file):
    """
    Example: SpencerLab
    These files give time in seconds in 30 sec interval
    Start of sleep time is not available
    :param file:
    :return:
    """
    dict_obj = {"epochstages": [], "epochoffset": 0}
    file.readline()  # done so that we can ignore the first line which just contain variable names
    times = []
    for line in file:
        temp = line.split('  ')
        if len(temp) == 1:
            temp = line.split('\t')
        temp[-1] = temp[-1].strip('\n')
        dict_obj["epochstages"].append(temp[-1].lstrip(" ").rstrip(" "))
        time = temp[0]
        time = int(time)
        times.append(time)
    dict_obj["epochoffset"] = times[0]
    return dict_obj


def _parse_full_type_txt_scorefile(file, epoch_len=pysleep_defaults.epoch_len):
    """
    Parse full type txt file. Example: CAPStudy, maybe other phsyionet stuff...
    :param file:
    :return:
    """
    # Type 2
    dict_obj = {"epochstages": [], "epochoffset": 0}
    # find line with SleepStage
    # find position of SleepStage and Time
    start_split = False
    get_starttime = True
    sleep_stage_pos = 0
    time_pos = 0
    event_pos = 0

    for line in file:
        if start_split and line.strip() != '':
            full_line = line.split('\t')
            if get_starttime:
                starttime = full_line[time_pos]
                get_starttime = False
            if len(full_line) > event_pos and full_line[event_pos].find("MCAP") == -1:
                dict_obj["epochstages"].append(full_line[sleep_stage_pos])

        if line.find("Sleep Stage") != -1:
            start_split = True
            full_line = line.split('\t')
            for i in range(len(full_line)):
                if full_line[i] == "Sleep Stage":
                    sleep_stage_pos = i
                if full_line[i].find("Time") != -1:
                    time_pos = i
                if full_line[i].find("Event") != -1:
                    event_pos = i

        if line.find('Recording Date:') != -1:
            full_line = line.split('\t')
            date = full_line[1]
            print(date)

    dict_obj['starttime'] = datetime.strptime(date + ' ' + starttime, '%d/%m/%Y %H.%M.%S')
    return dict_obj


def _parse_vrmk_scorfile(file, stage_map_dict, epoch_len=pysleep_defaults.epoch_len):
    """Parses a vmrk file from brainvision to epochstages and epochoffset"""
    annot = mne.read_annotaitons(file)
    assert annot.shape[0] > 0, "no sleep stage annotations found, this is probably an error"

    valid_stages = annot['description'].isin(stage_map_dict.keys())
    annot = annot.loc[valid_stages, :]
    annot = _resample_to_new_epoch_len(annot, epoch_len)

    #dictObj['starttime'] = edf.hdr['start_time'] #TODO deal with startime

    return {'epochstages':list(annot['description'].values), 'epochoffset':annot['onset'].values[0]}


def _parse_grass_scorefile(file):
    """
    Parse the grass type scorefile
    :param file: the file to parse
    :return:
    """
    dict_obj = {"epochoffset": 0, 'epochstages': []}

    list_data = pd.read_excel(file, sheet_name="list")
    graph_data = pd.read_excel(file, sheet_name="GraphData")

    time = None
    date = None
    for row_idx, row in list_data.iterrows():
        if (row.iloc[1] == "RecordingStartTime"):
            time = row.iloc[2]
        if (row.iloc[1] == "TestDate"):
            date = row.iloc[2]
        if date is not None and time is not None:
            break

    dict_obj['starttime'] = datetime.strptime(date + ' ' + time, '%m/%d/%y %H:%M:%S')

    for _, i in graph_data.iterrows():
        if not (math.isnan(i[1])):
            dict_obj['epochstages'].append(int(i[1]))
        else:
            break

    return dict_obj


def hume_matfile_loader(matfile_path):
    """
    Loads a hume matlab .mat file which contains a struct, and writes fields and values to a dictionary
    :param matfile_path: path of matlab file to load
    :return: dict of matlab information
    """
    mat_struct = loadmat(matfile_path)

    # build a list of keys and values for each entry in the structure
    if 'stageData' in mat_struct:
        vals = mat_struct['stageData'][0, 0]
        keys = mat_struct['stageData'][0, 0].dtype.descr
    elif 'mrk' in mat_struct:
        mat_dict = {'stages':mat_struct['mrk'][:, 0]}
        return mat_dict

    # Assemble the keys and values into variables with the same name as that used in MATLAB
    mat_dict = {}
    for i in range(len(keys)):
        key = keys[i][0]
        if len(vals[key].shape) > 1 and vals[key].shape[0] > vals[key].shape[1]:
            vals[key] = vals[key].T
        if len(vals[key][0]) > 1:
            val = np.squeeze(vals[key][0])
        else:
            val = np.squeeze(vals[key][0][0])  # squeeze is used to covert matlat (1,n) arrays into numpy (1,) arrays.
        mat_dict[key] = val

    return mat_dict


def mat_datenum_to_py_datetime(mat_datenum):
    """
    Converts a matlab "datenum" type to a python datetime type
    :param mat_datenum: matlab datenum to conver
    :return: converted datetime
    """
    return datetime.fromordinal(int(mat_datenum)) + timedelta(days=float(mat_datenum) % 1) - timedelta(days=366)