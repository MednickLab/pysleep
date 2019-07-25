from mednickdb_pysleep import frequency_features, pysleep_defaults
import mne
import numpy as np
from typing import List, Union, Tuple
import pandas as pd


def employ_buckelmueller(edf_filepath, epochstages, epochoffset_secs=None,
                         end_offset=None, chans_to_consider=None, delta_thresh=2.5, beta_thresh=2.0):

    band_power = frequency_features.extract_band_power(edf_filepath=edf_filepath,
                                                       bands = {'delta': (0.75, 4.5),
                                                                'beta': (20, 40)},
                                                       epochoffset_secs=epochoffset_secs,
                                                       end_time=end_offset,
                                                       chans_to_consider=chans_to_consider,
                                                       epoch_len=pysleep_defaults.band_power_epoch_len)

    band_power_per_epoch = frequency_features.extract_band_power_per_epoch(band_power,
                                                                           epoch_len=pysleep_defaults.epoch_len)

    band_power_w_stage = frequency_features.assign_band_power_stage(band_power_per_epoch,
                                                                    epochstages)

    band_power_w_stage.sort_values(['chan', 'stage_idx'])

    band_power_piv = band_power_w_stage.pivot_table(columns=['band'], index=['stage_idx', 'chan', 'stage'], values=['power'])
    band_power_piv.columns = band_power_piv.columns.droplevel()
    band_power_piv = band_power_piv.reset_index()

    bad_epochs = []
    for chan_name, chan_data in band_power_piv.groupby('chan'):
        for mid_idx in range(0,chan_data.shape[0]):
            start_idx = mid_idx - 7 if mid_idx - 7 > 0 else 0
            end_idx = mid_idx + 7 if mid_idx + 7 < chan_data.shape[0] else chan_data.shape[0]
            slice_range = list(range(start_idx,end_idx))
            slice_range.remove(mid_idx)
            data_slice = chan_data.iloc[slice_range,:].loc[:,['delta','beta']].agg(np.nanmean)
            epoch_slice = chan_data.iloc[mid_idx,:].loc[['delta','beta']]
            if any(epoch_slice > data_slice*[delta_thresh, beta_thresh]):
                bad_epochs.append(band_power_piv['stage_idx'].iloc[mid_idx])

    return np.unique(bad_epochs).tolist()


def employ_hjorth(edf_filepath, epochoffset_secs=None, end_offset=None, chans_to_consider=None, return_events=False, hjorth_threshold=2.0):
    edf = mne.io.read_raw_edf(edf_filepath, preload=True)
    if chans_to_consider is not None:
        edf = edf.drop_channels([chan for chan in edf.ch_names if chan not in chans_to_consider])

    if epochoffset_secs is None:
        epochoffset_secs = 0
    edf = edf.crop(tmin=epochoffset_secs, tmax=end_offset)
    events = mne.make_fixed_length_events(edf, id=1, duration=pysleep_defaults.epoch_len)

    def activity(sig):
        return np.std(sig, axis=2)**2

    def mobility(sig):
        sig_diff = np.diff(epoch_chan_time_data, axis=2)
        return activity(sig_diff)/activity(sig)

    def complexity(sig):
        sig_diff = np.diff(epoch_chan_time_data, axis=2)
        return mobility(sig_diff)/mobility(sig)

    # Epoch length is 1.5 second
    epochs = mne.Epochs(edf, events, tmin=0., tmax=pysleep_defaults.epoch_len, baseline=None, detrend=0)
    epoch_chan_time_data = epochs.get_data()
    epoch_chan_rms = np.sqrt(np.nanmean(epoch_chan_time_data**2, axis=2))
    epoch_chan_activity = activity(epoch_chan_time_data)
    epoch_chan_mobility = mobility(epoch_chan_time_data)
    epoch_chan_complexity = complexity(epoch_chan_time_data)
    bad_epochs = []
    for meas in [epoch_chan_rms, epoch_chan_activity, epoch_chan_mobility, epoch_chan_complexity]:
        mean_meas = np.nanmean(meas, axis=0)
        std_meas = hjorth_threshold*np.nanstd(meas, axis=0)

        for mean_chan, std_chan in zip(mean_meas, std_meas):
            bad_epochs_per_meas = np.any((meas > mean_chan+std_chan) | (meas < mean_chan-std_chan), axis=1)
            bad_epochs += np.where(bad_epochs_per_meas)[0].tolist()

    if return_events:
        return np.unique(bad_epochs).tolist(), events
    else:
        return np.unique(bad_epochs).tolist()


def detect_artifacts(edf_filepath, epochstages,
                     epochoffset_secs=None, end_offset=None, chans_to_consider=None,
                     hjorth_threshold=3.5, delta_threshold=3.5, beta_threshold=3.5):
    bad_epochs = employ_buckelmueller(edf_filepath, epochstages,
                                      epochoffset_secs=epochoffset_secs,
                                      end_offset=end_offset,
                                      chans_to_consider=chans_to_consider,
                                      beta_thresh=beta_threshold,
                                      delta_thresh=delta_threshold
                                      )
    bad_epochs += employ_hjorth(edf_filepath,
                                epochoffset_secs=epochoffset_secs,
                                end_offset=end_offset,
                                hjorth_threshold=hjorth_threshold,
                                chans_to_consider=chans_to_consider)
    return np.unique(bad_epochs).tolist()


def epochs_with_artifacts_to_event_df(epochs_with_artifacts: List[int],
                                      epoch_len=pysleep_defaults.epoch_len,
                                      seconds_between_epochstages_and_edf=0):
    onsets = np.array(epochs_with_artifacts)*epoch_len + seconds_between_epochstages_and_edf
    event_df = pd.DataFrame({'onset': onsets})
    event_df['description'] = 'bad_epoch'
    event_df['duration'] = epoch_len
    return event_df


