from wonambi import Dataset
from wonambi.detect import DetectSpindle, DetectSlowWave
from mednickdb_pysleep import pysleep_utils
from mednickdb_pysleep import pysleep_defaults
from mednickdb_pysleep.error_handling import EEGError
from typing import List, Tuple, Dict, Union
import pandas as pd
import numpy as np
import warnings
import datetime
import os
import contextlib
import sys
import logging
import inspect

try:
    logger = inspect.currentframe().f_back.f_globals['logger']
except KeyError:
    logger = logging.getLogger('errorlog')
    logger.info = print#

class DummyFile(object):
    def write(self, x): pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

if 'skip_rem' not in os.environ:
    if pysleep_defaults.load_matlab_detectors:
        import yetton_rem_detector
        yetton_rem_detector.initialize_runtime(['-nojvm', '-nodisplay'])
        rem_detector = yetton_rem_detector.initialize()

def extract_features(edf_filepath: str,
                     epochstages: List[str],
                     epochoffset_secs: Union[float, None]=None,
                     end_offset: Union[float, None]=None,
                     do_slow_osc: bool=True,
                     do_spindles: bool=True,
                     chans_for_spindles: List[str]=None,
                     chans_for_slow_osc: List[str]=None,
                     epochs_with_artifacts: List[int]=None,
                     do_rem: bool=False,
                     spindle_algo: str='Wamsley2012',
                     timeit=False):
    """
    Run full feature extraction (rem, spindles and SO) on an edf
    :param edf_filepath: path to edf file
    :param epochstages: epochstages list e.g. ['waso', 'waso', 'n1', 'n2', 'n2', etc]
    :param epochoffset_secs: difference between the start of the epochstages and the edf in seconds
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
    start_offset = epochoffset_secs

    chans_for_slow_osc = None if chans_for_slow_osc == 'all' else chans_for_slow_osc
    chans_for_spindles = None if chans_for_spindles == 'all' else chans_for_spindles

    if do_spindles:
        if timeit:
            starttime = datetime.datetime.now()
        logger.info('Spindle Extraction starting for ' + edf_filepath)
        data = load_and_slice_data_for_feature_extraction(edf_filepath=edf_filepath,
                                                          epochstages=epochstages,
                                                          epochoffset_secs=start_offset,
                                                          bad_segments=epochs_with_artifacts,
                                                          end_offset=end_offset,
                                                          chans_to_consider=chans_for_spindles)
        spindles = detect_spindles(data, start_offset=start_offset, algo=spindle_algo)
        if spindles is None or spindles.shape[0]==0:
            logger.warning('No Spindles detected for ' + edf_filepath)
        else:
            n_spindles = spindles.shape[0]
            logger.info('Detected '+ str(n_spindles) + ' spindles on ' + edf_filepath)
            if timeit:
                logger.info('Spindle extraction took '+str(datetime.datetime.now()-starttime))
                donetime = datetime.datetime.now()
            spindles = assign_stage_to_feature_events(spindles, epochstages)
            assert all(spindles['stage'].isin(pysleep_defaults.nrem_stages)), "All stages must be nrem. If missmatch maybe epochoffset is incorrect?"
            if spindles.shape[0]:
                features_detected.append(spindles)
            if timeit:
                logger.info('Bundeling extraction took '+str(datetime.datetime.now()-donetime))

    if do_slow_osc:
        if timeit:
            starttime = datetime.datetime.now()
        logger.info('Slow Osc Extraction starting for '+edf_filepath)
        if not do_spindles or chans_for_slow_osc != chans_for_spindles:
            data = load_and_slice_data_for_feature_extraction(edf_filepath=edf_filepath,
                                                              epochstages=epochstages,
                                                              epochoffset_secs=start_offset,
                                                              bad_segments=epochs_with_artifacts,
                                                              end_offset=end_offset,
                                                              chans_to_consider=chans_for_slow_osc)
        sos = detect_slow_oscillation(data, start_offset=start_offset)
        if sos is None:
            logger.warning('No SO detected for ' + edf_filepath)
        else:
            n_sos = sos.shape[0]
            logger.info('Detected '+str(n_sos)+ ' slow osc for ' + edf_filepath)
            sos = assign_stage_to_feature_events(sos, epochstages)

            assert all(sos['stage'].isin(pysleep_defaults.nrem_stages)), "All stages must be nrem. If missmatch maybe epochoffset is incorrect?"
            if sos.shape[0]:
                features_detected.append(sos)
            if timeit:
                logger.info('Slow Osc extraction took '+str(datetime.datetime.now()-starttime))

    if do_rem:
        if not pysleep_defaults.load_matlab_detectors:
            warnings.warn('Requested REM, but matlab detectors are turned off. Turn on in pysleep defaults.')
        else:
            if timeit:
                starttime = datetime.datetime.now()
            try:
                logger.info('REM Extraction starting for '+ edf_filepath)
                data = load_and_slice_data_for_feature_extraction(edf_filepath=edf_filepath,
                                                                  epochstages=epochstages,
                                                                  epochoffset_secs=start_offset,
                                                                  bad_segments = epochs_with_artifacts,
                                                                  end_offset=end_offset,
                                                                  chans_to_consider=['LOC','ROC'],
                                                                  stages_to_consider=['rem'])
            except ValueError:
                warnings.warn('LOC and ROC must be present in the record. Cannot do REM')
                rems = None
            else:
                rems = detect_rems(edf_filepath=edf_filepath, data=data, start_time=start_offset)
            if rems is None:
                logger.warning('No REM detected for ' + edf_filepath)
            else:
                rems = assign_stage_to_feature_events(rems, epochstages)
                assert all(rems['stage'] == 'rem'), "All stages for rem must be rem. If missmatch maybe epochoffset is incorrect?"
                logger.info('Detected '+ str(rems.shape[0]) + ' REMs for ' + edf_filepath)
                if rems.shape[0]:
                    features_detected.append(rems)
                if timeit:
                    logger.info('REM extraction took'+ str(datetime.datetime.now() - starttime))

    if features_detected:
        sleep_features_df = pd.concat(features_detected, axis=0, sort=False)
        if do_spindles and do_slow_osc:
            sleep_features_df = detect_slow_osc_spindle_overlap(sleep_features_df,
                                            coupling_secs=pysleep_defaults.so_spindle_overlap,
                                            as_bool=True)
        return sleep_features_df
    else:
        return None


def load_and_slice_data_for_feature_extraction(edf_filepath: str,
                                               epochstages: List[str],
                                               bad_segments: List[int]=None,
                                               epochoffset_secs: float = None,
                                               end_offset: float = None,
                                               chans_to_consider: List[str] = None,
                                               epoch_len=pysleep_defaults.epoch_len,
                                               stages_to_consider=pysleep_defaults.nrem_stages):
    if epochoffset_secs is None:
        epochoffset_secs = 0
    if end_offset is not None:
        last_good_epoch = int((end_offset - epochoffset_secs) / epoch_len)
        epochstages = epochstages[0:last_good_epoch]

    d = Dataset(edf_filepath)

    eeg_data = d.read_data().data[0]
    if not (1 < np.sum(np.abs(eeg_data))/eeg_data.size < 200):
        raise EEGError("edf data should be in mV, please rescale units in edf file")

    if bad_segments is not None:
        for bad_epoch in bad_segments:
            epochstages[bad_epoch]='artifact'
    epochstages = pysleep_utils.convert_epochstages_to_eegevents(epochstages, start_offset=epochoffset_secs)
    epochstages_to_consider = epochstages.loc[epochstages['description'].isin(stages_to_consider), :]
    starts = epochstages_to_consider['onset'].tolist()
    ends = (epochstages_to_consider['onset'] + epochstages_to_consider['duration']).tolist()

    for i in range(len(starts)-1,0,-1):
        if starts[i] == ends[i-1]:
            del starts[i]
            del ends[i-1]

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
               'peak_val_det': 'peak_uV', #peak in the band of interest (removing DC, and other signal components)
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
    if spindles_df.shape[0] == 0:
        return None #empty df
    spindles_df['peak_time'] = spindles_df['peak_time'] - spindles_df['onset']
    spindles_df['description'] = 'spindle'
    if start_offset is not None:
        spindles_df['onset'] = spindles_df['onset'] - start_offset
        spindles_df = spindles_df.loc[spindles_df['onset'] >= 0, :]
    return spindles_df.sort_values('onset')


def detect_slow_oscillation(data: Dataset, algo: str = 'AASM/Massimini2004', start_offset: float = None) -> pd.DataFrame:
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
               'trough_val': 'trough_uV',
               'peak_val': 'peak_uV',
               'dur': 'duration',
               'ptp': None,
               'chan': 'chan'}
    cols_to_keep = set(sos_df.columns) - set([k for k, v in col_map.items() if v is None])
    sos_df = sos_df.loc[:, cols_to_keep]
    sos_df.columns = [col_map[k] for k in sos_df.columns]
    if sos_df.shape[0] == 0:
        return None #empty df
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
                start_time: float=0,
                std_rem_width: float = 0.1,
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
    # if data.header['s_freq'] != 256:
    #     raise EEGError("edf should be 256Hz. Please resample.")
    rem_starts = [int(d*256) for d in data.starts] #must be in samples, must be 256Hz.
    rem_ends = [int(d*256) for d in data.ends]
    if len(rem_starts) > 0:
        with nostdout():
            onsets = rem_detector.runDetectorCommandLine(edf_filepath, [rem_starts, rem_ends], algo, loc_chan, roc_chan, 0)
    else:
        return None
    if isinstance(onsets, float): #one rem?
        warnings.warn('Only a single rem was found, this may be an error')
        pass
    if len(onsets) > 0:
        onsets = onsets[0]
    else:
        return None
    rem_df = pd.DataFrame({'onset': onsets}, dtype=float)
    rem_df['description'] = 'rem_event'
    if start_time is not None:
        rem_df['onset'] = rem_df['onset'] - start_time
        rem_df = rem_df.loc[rem_df['onset'] >= 0, :]
    rem_df['duration'] = std_rem_width
    rem_df['chan'] = 'LOC'
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

    labels = [(s,i) for s,i in zip(stage_events['stage'], stage_events['stage_idx'])]
    end_point = stage_events['onset'].iloc[-1] + stage_events['duration'].iloc[-1]
    stages = pd.DataFrame(pd.cut(feature_events['onset'],
                                 stage_events['onset'].to_list()+[end_point],
                                 right=False,
                                 labels=labels).to_list(),
                          columns=['stage','stage_idx'])

    feature_events = pd.concat([feature_events.reset_index(drop=True), stages.reset_index(drop=True)], axis=1)

    return feature_events


def detect_slow_osc_spindle_overlap(features_df, coupling_secs=None, as_bool=False) -> pd.DataFrame:
    """
    Detect if a set of features (i.e. spindles) are close (with coupling_secs) to a set of base features (i.e. Slow Osc)
    :param base_onsets: the onsets of features to search around (Slow Osc)
    :param candidate_onsets:
    :param offset_secs:
    :return: closest infront of onset, closest behind onset, Nan if nothing within "coupling secs"
    """
    if as_bool:
        assert coupling_secs is not None, 'if you want a yes no returned for coupling, then there must be a couping secs'

    SO_df = features_df.loc[features_df['description'] == 'slow_osc', :]
    spindle_df = features_df.loc[features_df['description'] == 'spindle', :]

    def overlap_func(base, spindle_onsets, coupling_secs=None, as_bool=False):
        if spindle_onsets.shape[0] == 0:
            return pd.Series({'before':np.nan, 'after':np.nan})
        spindle_diff = spindle_onsets - base
        closest_before = spindle_diff[spindle_diff<0].iloc[-1] if spindle_diff[spindle_diff<0].shape[0]!=0 else np.nan
        closest_after = spindle_diff[spindle_diff>0].iloc[0] if spindle_diff[spindle_diff>0].shape[0]!=0 else np.nan
        if coupling_secs:
            if abs(closest_before) < coupling_secs:
                if as_bool:
                    closest_before = True
            else:
                closest_before = np.nan
                if as_bool:
                    closest_before = False

            if closest_after < coupling_secs:
                if as_bool:
                    closest_after = True
            else:
                closest_after = np.nan
                if as_bool:
                    closest_after = False
        return pd.Series({'before':closest_before, 'after':closest_after})

    for chan, chan_data in SO_df.groupby('chan'):
        overlap = chan_data.apply(lambda x: overlap_func(x['onset']+x['zero_time'],
                                                         spindle_df.loc[spindle_df['chan']==x['chan'],'onset'],
                                                         coupling_secs,
                                                         as_bool), axis=1)
        features_df.loc[(features_df['description'] == 'slow_osc') & (features_df['chan'] == chan), 'coupled_before'] = overlap['before']
        features_df.loc[(features_df['description'] == 'slow_osc') & (features_df['chan'] == chan), 'coupled_after'] = overlap['after']
    return features_df

def sleep_feature_variables_per_stage_old(feature_events: pd.DataFrame,
                                         mins_in_stage_df: pd.DataFrame=None,
                                      stages_to_consider: List[str]=pysleep_defaults.stages_to_consider,
                                      channels: List[str]=None,
                                      av_across_channels: bool=True):
    """
    Calculate the density, and mean of other important sleep feature variables (amp, power, peak freq, etc)
    :param feature_events: dataframe of a single event type (spindle, slow osc, rem, etc)
    :param stages_to_consider: The stages to extract for, i.e. you probably want to leave out REM when doing spindles
    :param channels: if None consider all channels THAT HAVE DETECTED SPINDLES, to include 0 density and count for
        channels that have no spindles, make sure to include this channel list argument.
    :param av_across_channels: whether to average across channels, or return separate for each channel
    :return: dataframe of with len(stage)*len(chan) or len(stage) rows with density + mean of each feature as columns
    """
    if 'stage_idx' in feature_events.columns:
        feature_events = feature_events.drop('stage_idx', axis=1)
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
                    features_per_stage = features_per_stage.drop('chan', axis=1).apply(
                        lambda x: pd.to_numeric(x, errors='ignore')).agg(np.nanmean)
                for idx_idx, idx in enumerate(index_vars):
                    features_per_stage[idx] = stage_and_other_idx[idx_idx]
                features_per_stage_cont.append(features_per_stage)
    if len(features_per_stage_cont) > 0:
        if av_across_channels:
            features_df = pd.concat(features_per_stage_cont, axis=1, sort=False).T
        else:
            features_df = pd.concat(features_per_stage_cont, axis=0, sort=False)
        return features_df
    else:
        logger.info('No events in given stages')
        return pd.DataFrame() #return empty df


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
        channels that have no spindles, make sure to include this channel list argument.
    :param av_across_channels: whether to average across channels, or return separate for each channel
    :return: dataframe of with len(stage)*len(chan) or len(stage) rows with density + mean of each feature as columns
    """
    cont = []
    for feature_name, a_feature_event in feature_events.groupby('description'):
        to_drop = [col for col in a_feature_event.columns if col in ['stage_idx','onset']]
        a_feature_event = a_feature_event.drop(to_drop, axis=1)
        if stages_to_consider is not None:
            a_feature_event = a_feature_event.loc[a_feature_event['stage'].isin(stages_to_consider),:]
        if channels is not None:
            a_feature_event = a_feature_event.loc[a_feature_event['chan'].isin(channels),:]

        index_vars = ['chan', 'stage','description', 'quartile']
        if av_across_channels:
            index_vars.remove('chan')
        if 'quartile' not in a_feature_event.columns:
            index_vars.remove('quartile')
            mins_idx_vars = [index_vars.index('stage')]
        else:
            mins_idx_vars = [index_vars.index('stage'), index_vars.index('quartile')]

        var_values = {}
        for var in index_vars:
            values = a_feature_event[var].unique()
            var_values[var] = values

        data_cols = [col for col in a_feature_event.columns if col not in index_vars]
        if av_across_channels:
            data_cols.remove('chan')
        multi_idx = pd.MultiIndex.from_product(var_values.values(), names=var_values.keys())
        features_df = pd.DataFrame(index=multi_idx, columns=data_cols)
        features_df['count'] = 0
        features_df['density'] = 0
        for idxs, feature_data in a_feature_event.groupby(index_vars):
            mins_idx = [idxs[i] for i in mins_idx_vars]
            mins_in_stage = mins_in_stage_df.loc[mins_idx, 'minutes_in_stage'][0]
            features_df.loc[idxs, 'density'] = feature_data.shape[0] / mins_in_stage
            features_df.loc[idxs, 'count'] = feature_data.shape[0]
            agg_data = feature_data.agg(np.nanmean)
            features_df.loc[idxs, agg_data.index] = agg_data.values
        features_df = features_df.rename({col:'av_'+col for col in features_df.columns if col in data_cols})
        cont.append(features_df.reset_index())
    return pd.concat(cont, axis=0)
