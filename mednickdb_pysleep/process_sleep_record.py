#%% Import the tools we need
from mednickdb_pysleep import sleep_features, pysleep_defaults, pysleep_utils, scorefiles, \
    sleep_architecture, frequency_features, artifact_detection
import pandas as pd
import numpy as np
import yaml
import mne
import warnings
import datetime
from mednickdb_pysleep.error_handling import EEGError
#warnings.filterwarnings("ignore")


def extract_eeg_variables(edf_filepath,
                          epochstages,
                          epochoffset_secs=None,
                          end_offset=None,
                          do_artifacting=True,
                          do_spindles=True,
                          do_slow_osc=True,
                          do_rem=False,
                          do_band_power=True,
                          artifacting_channels=None,
                          spindle_channels=None,
                          slow_osc_channels=None,
                          band_power_channels=None,
                          do_quartiles=False,
                          timeit=False):
    """
    Runs a set of algorithms to artifact, extract band power, spindles, slow osc, rem from record. Master function for pysleep...
    :param edf_filepath:
    :param epochstages:
    :param epochoffset_secs:
    :param end_offset:
    :param do_artifacting:
    :param do_spindles:
    :param do_slow_osc:
    :param do_rem:
    :return:
    """
    chs = []
    for chans in [spindle_channels, slow_osc_channels, band_power_channels, artifacting_channels]:
        if chans is not None:
            chs += chans
    ch_names = mne.io.read_raw_edf(edf_filepath).ch_names
    missing_chans = [ch for ch in set(chs) if ch not in ch_names]
    if len(missing_chans):
        raise EEGError("Some channels are missing: "+str(missing_chans))
    # get the start and end of where we want to extract spindles from (lights off->lights on)
    og_starttime = datetime.datetime.now()
    if do_artifacting:
        if timeit:
            starttime = datetime.datetime.now()
            print(og_starttime-starttime, '\tDetecting artifacts')
        epochs_with_artifacts = artifact_detection.detect_artifacts(edf_filepath=edf_filepath,
                                                                    epochstages=epochstages,
                                                                    epochoffset_secs=epochoffset_secs,
                                                                    end_offset=end_offset,
                                                                    chans_to_consider=artifacting_channels)

        print('\t\tRecord contains ', 100 * len(epochs_with_artifacts) / len(epochstages),'% bad epochs that will be ignored')
        if timeit:
            print('\tDetect artifacts took',datetime.datetime.now()-starttime)
    else:
        epochs_with_artifacts = None

    if do_band_power:
        if timeit:
            starttime = datetime.datetime.now()
            print(datetime.datetime.now()-og_starttime, '\tBand power extraction started')
        band_power = frequency_features.extract_band_power(edf_filepath=edf_filepath,
                                                           epochoffset_secs=epochoffset_secs,
                                                           end_time=end_offset,
                                                           chans_to_consider=band_power_channels,
                                                           epoch_len=pysleep_defaults.band_power_epoch_len)

        band_power_per_epoch = frequency_features.extract_band_power_per_epoch(band_power,
                                                                               epoch_len=pysleep_defaults.epoch_len)

        power_df = frequency_features.assign_band_power_stage(band_power_per_epoch,
                                                                        epochstages, bad_epochs=epochs_with_artifacts)

        if do_quartiles:
            power_df, _ = pysleep_utils.assign_quartiles(power_df, epochstages)
            groupby = ['quartile', 'stage', 'chan', 'band']
        else:
            groupby = ['stage', 'chan', 'band']

        power_df = power_df.drop(['onset', 'duration'], axis=1)
        band_power_w_stage_to_mean = power_df.loc[
                                     power_df['stage'].isin(pysleep_defaults.stages_to_consider), :]
        power_averages_df = band_power_w_stage_to_mean.groupby(groupby).agg(np.nanmean)
        power_averages_df['power'] = pysleep_utils.trunc(power_averages_df['power'], 3)
        power_averages_df.index = ['_'.join(idx) for idx in power_averages_df.index.values]
        if timeit:
            print('\tBand power took',datetime.datetime.now()-starttime)
    else:
        power_averages_df = power_df = None

    if do_spindles or do_slow_osc or do_rem:
        if timeit:
            starttime = datetime.datetime.now()
            print(datetime.datetime.now()-og_starttime, '\tFeature extraction started')
        mins_df = sleep_architecture.sleep_stage_architecture(epochstages,
                                                              epochs_to_ignore=epochs_with_artifacts,
                                                              return_type='dataframe',
                                                              per_quartile=do_quartiles)

        # %% Sleep Features
        features_df = sleep_features.extract_features(edf_filepath=edf_filepath,
                                                      epochstages=epochstages,
                                                      epochoffset_secs=epochoffset_secs,
                                                      end_offset=end_offset,
                                                      chans_for_spindles=spindle_channels,
                                                      chans_for_slow_osc=slow_osc_channels,
                                                      epochs_with_artifacts=epochs_with_artifacts,
                                                      do_rem=do_rem,
                                                      do_spindles=do_spindles,
                                                      do_slow_osc=do_slow_osc,
                                                      timeit=timeit)

        if features_df is not None:
            if do_quartiles:
                features_df, _ = pysleep_utils.assign_quartiles(features_df, epochstages)
                groupby = ['quartile', 'stage', 'chan', 'description']
            else:
                groupby = ['stage', 'chan', 'description']

            feature_averages_df = sleep_features.sleep_feature_variables_per_stage(features_df,
                                                                                  mins_in_stage_df=mins_df,
                                                                                  av_across_channels=False,
                                                                                  stages_to_consider=pysleep_defaults.stages_to_consider)

            feature_averages_df = feature_averages_df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
            feature_averages_df = feature_averages_df.groupby(groupby).agg(np.nanmean)
            feature_averages_df = feature_averages_df.apply(lambda x: pysleep_utils.trunc(x, 3)).reset_index()
            if timeit:
                print('\tFeatures took',datetime.datetime.now()-starttime)
        else:
            feature_averages_df = None
    else:
        feature_averages_df = features_df = None

    return features_df, power_df, feature_averages_df, power_averages_df, epochs_with_artifacts

