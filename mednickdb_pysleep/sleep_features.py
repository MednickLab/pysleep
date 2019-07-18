from wonambi import Dataset
from wonambi.detect import DetectSpindle, DetectSlowWave
from mednickdb_pysleep import pysleep_utils
from mednickdb_pysleep import pysleep_defaults
from mednickdb_pysleep.sleep_architecture import sleep_stage_architecture
from typing import List, Tuple, Dict, Union
import pandas as pd
import numpy as np
import warnings

if pysleep_defaults.load_matlab_detectors:
    import yetton_rem_detector
    rem_detector = yetton_rem_detector.initialize()


class EEGError(BaseException):
    """Some error with EEG has occured"""
    pass


def extract_features(edf_filepath: str,
                     epochstages: List[str],
                     offset_between_epochstages_and_edf: Union[float, None],
                     end_offset: Union[float, None],
                     do_slow_osc: bool=True,
                     do_spindles: bool=True,
                     chans_for_spindles: List[str]=None,
                     chans_for_slow_osc: List[str]=None,
                     epochs_with_artifacts: List[int]=None,
                     do_rem: bool=False,
                     spindle_algo: str='Wamsley2012'):
    """
    Run full feature extraction (rem, spindles and SO) on an edf
    :param edf_filepath: path to edf file
    :param epochstages: epochstages list e.g. ['waso', 'waso', 'n1', 'n2', 'n2', etc]
    :param offset_between_epochstages_and_edf: difference between the start of the epochstages and the edf in seconds
    :param end_offset: end time to stop extraction for (seconds since start of edf)
    :param do_slow_osc: whether to extract slow oscillations or not
    :param do_spindles: whether to extract spindles or not
    :param do_rem: whether to extract rem or not, note that matlab detectors must be turned on in pysleep_defaults
    :param chans_for_spindles: which channels to extract for, can be empty list (no channels), None or all (all channels) or a list of channel names
    :param chans_for_slow_osc: which channels to extract for, can be empty list (no channels), None or all (all channels) or a list of channel names
    :param epochs_with_artifacts: idx of epochstages that are bad or should be skipped
    :param spindle_algo: which spindle algo to run, see the list in Wonambi docs
    :return: dataframe of all events, with description, stage, onset (seconds since epochstages start), duration, and feature properties
    """
    features_detected = []
    start_offset = offset_between_epochstages_and_edf

    chans_for_slow_osc = None if chans_for_slow_osc == 'all' else chans_for_slow_osc
    chans_for_spindles = None if chans_for_spindles == 'all' else chans_for_spindles

    if do_spindles:
        data = load_and_slice_data_for_feature_extraction(edf_filepath=edf_filepath,
                                                          epochstages=epochstages,
                                                          start_offset=start_offset,
                                                          bad_segments=epochs_with_artifacts,
                                                          end_offset=end_offset,
                                                          chans_to_consider=chans_for_spindles)
        spindles = detect_spindles(data, start_offset=start_offset, algo=spindle_algo)
        n_spindles = spindles.shape[0]
        print('Detected', n_spindles, 'spindles')
        spindles = assign_stage_to_feature_events(spindles, epochstages)
        features_detected.append(spindles)

    if do_slow_osc:
        data = load_and_slice_data_for_feature_extraction(edf_filepath=edf_filepath,
                                                          epochstages=epochstages,
                                                          start_offset=start_offset,
                                                          bad_segments=epochs_with_artifacts,
                                                          end_offset=end_offset,
                                                          chans_to_consider=chans_for_slow_osc)
        sos = detect_slow_oscillation(data, start_offset=start_offset)
        n_sos = sos.shape[0]
        print('Detected',n_sos, 'spindles')
        sos = assign_stage_to_feature_events(sos, epochstages)
        features_detected.append(sos)

    if do_rem:
        if not pysleep_defaults.load_matlab_detectors:
            warnings.warn('Requested REM, but matlab detectors are turned off. Turn on in pysleep defaults.')

        try:
            data = load_and_slice_data_for_feature_extraction(edf_filepath=edf_filepath,
                                                              epochstages=epochstages,
                                                              start_offset=start_offset,
                                                              bad_segments = epochs_with_artifacts,
                                                              end_offset=end_offset,
                                                              chans_to_consider=['LOC','ROC'],
                                                              stages_to_consider=['rem'])
        except ValueError as e:
            raise EEGError('LOC and ROC must be present in the record') from e
        rems = detect_rems(edf_filepath=edf_filepath, data=data)
        rems = assign_stage_to_feature_events(rems, epochstages)
        features_detected.append(rems)

    return pd.concat(features_detected, axis=0, sort=False)


def load_and_slice_data_for_feature_extraction(edf_filepath: str,
                                               epochstages: List[str],
                                               bad_segments: List[List[float]]=None,
                                               start_offset: float = None,
                                               end_offset: float = None,
                                               chans_to_consider: List[str] = None,
                                               epoch_len=pysleep_defaults.epoch_len,
                                               stages_to_consider=pysleep_defaults.nrem_stages):
    if start_offset is None:
        start_offset = 0
    if end_offset is not None:
        last_good_epoch = int((end_offset-start_offset)/epoch_len)
        epochstages = epochstages[0:last_good_epoch]

    d = Dataset(edf_filepath)

    eeg_data = d.read_data().data[0]
    if not (1 < np.sum(np.abs(eeg_data))/eeg_data.size < 100):
        raise EEGError("edf data should be in mV, please rescale units in edf file")

    epochstages = pysleep_utils.convert_epochstages_to_eegevents(epochstages, start_offset=start_offset)
    epochstages_to_consider = epochstages.loc[epochstages['description'].isin(stages_to_consider), :]
    starts = epochstages_to_consider['onset'].tolist()
    ends = (epochstages_to_consider['onset'] + epochstages_to_consider['duration']).tolist()

    if bad_segments is not None:
        starts = starts + bad_segments[1]
        sorted(starts)
        ends = bad_segments[0] + ends
        sorted(ends)

    for i, (s, e) in enumerate(zip(starts[::-1].copy(), ends[::-1].copy())):
        if s == e:
            del starts[i]
            del ends[i]

    data = d.read_data(begtime=starts, endtime=ends, chan=chans_to_consider)
    data.starts = starts
    data.ends = ends
    return data

def detect_spindles(data: Dataset, algo: str = 'Wamsley2012',
                    start_offset: float = None) ->pd.DataFrame:
    """
    Detect spindles locations in an edf file for each channel.
    :param edf_filepath: path of edf file to load. Will maybe work with other filetypes. untested.
    :param algo: which algorithm to use to detect spindles. See wonambi methods: https://wonambi-python.github.io/gui/methods.html
    :param chans_to_consider: which channels to detect spindles on, must match edf channel names
    :param bad_segments:
    :param start_offset: offset between first epoch and edf - onset is measured from this
    :return: returns dataframe of spindle locations, with columns for chan, start, duration and other spindle properties, sorted by onset
    """

    detection = DetectSpindle(algo)
    spindles_detected = detection(data)
    spindles_df = pd.DataFrame(spindles_detected.events, dtype=float)
    col_map = {'start': 'onset',
               'end': None,
               'peak_time': 'peak_time',
               'peak_val_det': 'peak_uV',
               'peak_val_orig': None,
               'dur': 'duration',
               'auc_det': None,
               'auc_orig': None,
               'rms_det': None,
               'rms_orig': None,
               'power_orig': None,
               'peak_freq': 'freq_peak',
               'ptp_det': None,
               'ptp_orig': None,
               'chan': 'chan'}
    cols_to_keep = set(spindles_df.columns) - set([k for k, v in col_map.items() if v is None])
    spindles_df = spindles_df.loc[:, cols_to_keep]
    spindles_df.columns = [col_map[k] for k in spindles_df.columns]
    spindles_df['peak_time'] = spindles_df['peak_time'] - spindles_df['onset']
    spindles_df['description'] = 'spindle'
    if start_offset is not None:
        spindles_df['onset'] = spindles_df['onset'] - start_offset
        spindles_df = spindles_df.loc[spindles_df['onset'] >= 0, :]
    spindles_df = spindles_df.loc[(spindles_df['freq_peak'] > 11) & (spindles_df['freq_peak'] < 16),:]
    return spindles_df.sort_values('onset')


def detect_slow_oscillation(data: Dataset, algo: str = 'AASM/Massimini2004', start_offset: float = None) ->pd.DataFrame:
    """
    Detect slow waves (slow oscillations) locations in an edf file for each channel
    :param edf_filepath: path of edf file to load. Will maybe work with other filetypes. untested.
    :param algo: which algorithm to use to detect spindles. See wonambi methods: https://wonambi-python.github.io/gui/methods.html
    :param chans_to_consider: which channels to detect spindles on, must match edf channel names
    :param bad_segments:
    :param start_offset: offset between first epoch and edf - onset is measured from this
    :return: returns dataframe of spindle locations, with columns for chan, start, duration and other spindle properties, sorted by onset
    """
    detection = DetectSlowWave(algo)
    sos_detected = detection(data)
    sos_df = pd.DataFrame(sos_detected.events, dtype=float)
    col_map = {'start': 'onset',
               'end': None,
               'trough_time': 'trough_time',
               'zero_time': 'zero_time',
               'peak_time': 'peak_time',
               'trough_val': 'trough_val',
               'peak_val': 'peak_val',
               'dur': 'duration',
               'ptp': None,
               'chan': 'chan'}
    cols_to_keep = set(sos_df.columns) - set([k for k, v in col_map.items() if v is None])
    sos_df = sos_df.loc[:, cols_to_keep]
    sos_df.columns = [col_map[k] for k in sos_df.columns]
    sos_df['peak_time'] = sos_df['peak_time'] - sos_df['onset']
    sos_df['trough_time'] = sos_df['trough_time'] - sos_df['onset']
    sos_df['zero_time'] = sos_df['zero_time'] - sos_df['onset']
    sos_df['description'] = 'slow_osc'
    if start_offset is not None:
        sos_df['onset'] = sos_df['onset'] - start_offset
        sos_df = sos_df.loc[sos_df['onset']>=0,:]
    return sos_df.sort_values('onset')


def detect_rems(edf_filepath: str,
                data: Dataset,
                loc_chan: str = 'LOC',
                roc_chan: str = 'ROC',
                start_time=0,
                algo: str='HatzilabrouEtAl'):
    """
    Detect rapid eye movement events in an edf file from loc and roc channels (only REM stage considered). Sample Freq must be converted to 256Hz!
    :param edf_filepath: path of edf file to load. Will maybe work with other filetypes. untested.
    :param algo: which algorithm to use to detect spindles. See wonambi methods: https://wonambi-python.github.io/gui/methods.html
    :param chans_to_consider: which channels to detect spindles on, must match edf channel names
    :param epochstages: list of stages for each epoch
    :param start_offset: offset between first epoch and edf - onset is measured from this
    :return: returns dataframe of spindle locations, with columns for chan, start, duration and other spindle properties, sorted by onscdet
    """
    if data.header['s_freq'] != 256:
        raise EEGError("edf should be 256Hz. Please resample.")
    rem_starts = [d*256 for d in data.starts] #must be in samples, must be 256Hz.
    rem_ends = [d*256 for d in data.ends]
    onsets, _, _, _, _ = rem_detector.runDetectorCommandLine(edf_filepath, [rem_starts, rem_ends], algo, loc_chan, roc_chan)
    rem_df = pd.DataFrame({'onsets': onsets}, dtype=float)
    rem_df['description'] = 'rem_event'
    if start_time is not None:
        start_time = 0
    rem_df['onset'] = rem_df['onset'] - start_time
    rem_df = rem_df.loc[rem_df['onset']>=0,:]
    return rem_df


def assign_stage_to_feature_events(feature_events: pd.DataFrame,
                                   epochstages: list,
                                   epoch_stage_offset: int = 0,
                                   epoch_len=pysleep_defaults.epoch_len,
                                   ) -> pd.DataFrame:
    """
    :param feature_events: events dataframe, with start and duration columns
    :param epochstages: stages, where the first stage starts at 0 seconds on the events timeline
    :param epoch_stage_offset: the offset in seconds between the epoch stages and the features events
     (i.e. when epoch stages start compared to the time where feature_events['onset']==0)
    :return: the modified events df, with a stage column
    """
    feature_events['onset'] -= epoch_stage_offset
    if isinstance(epochstages, list):
        stage_events = pd.DataFrame({'onset':np.arange(0, len(epochstages))*epoch_len,
                                     'stage_idx':np.arange(0, len(epochstages)),
                                     'stage':epochstages})
        stage_events['duration'] = epoch_len
    else:
        raise ValueError('epochstages is of unknown type. Should be list.')

    def check_overlap(start, duration, events):
        end = start + duration
        for idx, stage in events.iterrows():
            if pysleep_utils.overlap(start, end, stage['onset'], stage['onset']+stage['duration']):
                return stage.loc[['stage', 'stage_idx']]
        return pd.Series({'stage':pysleep_defaults.unknown_stage, 'stage_idx':-1})

    stage_events_to_cat = feature_events.apply(lambda x: check_overlap(x['onset'], x['duration'], stage_events), axis=1)
    feature_events = pd.concat([feature_events, stage_events_to_cat], axis=1, sort=False)
    return feature_events


def sleep_feature_variables_per_stage(feature_events: pd.DataFrame,
                                      mins_in_stage_df: pd.DataFrame=None,
                                      stages_to_consider: List[str]=pysleep_defaults.stages_to_consider,
                                      channels: List[str]=None,
                                      av_across_channels: bool=True):
    """
    Calculate the density, and mean of other important sleep feature variables (amp, power, peak freq, etc)
    :param feature_events: dataframe of a single event type (spindle, slow osc, rem, etc)
    :param stages_to_consider: The stages to extract for, i.e. you probably want to leave out REM when doing spindles
    :param channels: if None consider all channels THAT HAVE DETECTED SPINDLES, to include 0 density and count for
        channels that have no spindles, make sure to inlcude this channel list argument.
    :param av_across_channels: whether to average across channels, or return separate for each channel
    :return: dataframe of with len(stage)*len(chan) or len(stage) rows with density + mean of each feature as columns
    """
    if 'quartile' in feature_events.columns:
        by_quart = True
        index_vars = ['stage','description', 'quartile']
    else:
        index_vars = ['stage','description']
        by_quart = False
    pos_non_var_cols = ['stage', 'onset', 'description', 'chan'] + index_vars
    non_var_cols = [col for col in feature_events.columns if col in pos_non_var_cols]
    features_per_stage_cont = []
    for stage_and_other_idx, feature_data_per_stage in feature_events.groupby(index_vars):
        stage = stage_and_other_idx[0]
        if by_quart:
            quart = stage_and_other_idx[-1]
            mins_in_stage = mins_in_stage_df.loc[(quart,stage),'minutes_in_stage']
        else:
            mins_in_stage = mins_in_stage_df.loc[stage, 'minutes_in_stage']
        if stage in stages_to_consider:
            per_chan_cont = []
            channels_without_events = set(feature_data_per_stage['chan'].unique() if channels is None else channels)
            for chan, feature_data_per_stage_chan in feature_data_per_stage.groupby('chan'):
                if channels is None or chan in channels:
                    channels_without_events = channels_without_events - {chan}
                    features_per_chan = feature_data_per_stage_chan.drop(non_var_cols, axis=1).agg(np.nanmean)
                    features_per_chan.index = ['av_'+col for col in features_per_chan.index]
                    features_per_chan['density'] = feature_data_per_stage_chan.shape[0] / mins_in_stage
                    features_per_chan['count'] = feature_data_per_stage_chan.shape[0]
                    features_per_chan['chan'] = chan
                    per_chan_cont.append(features_per_chan)
            if len(channels_without_events) > 0: #if there were channels that didnt have any spindles
                for chan in channels_without_events:
                    per_chan_cont.append(pd.Series({'chan': chan, 'density': 0, 'count': 0}))
            if len(per_chan_cont) > 0:
                features_per_stage = pd.concat(per_chan_cont, axis=1, sort=False).T
                if av_across_channels:
                    features_per_stage = features_per_stage.drop('chan', axis=1).agg(np.nanmean)
                for idx_idx, idx in enumerate(index_vars):
                    features_per_stage[idx] = stage_and_other_idx[idx_idx]
                features_per_stage_cont.append(features_per_stage)
    if len(features_per_stage_cont) > 0:
        if av_across_channels:
            features_df = pd.concat(features_per_stage_cont, axis=1).T
        else:
            features_df = pd.concat(features_per_stage_cont, axis=0)
        return features_df
    else:
        print('No events in given stages')
        return pd.DataFrame() #return empty df
