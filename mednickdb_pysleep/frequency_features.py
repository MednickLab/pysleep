from wonambi import Dataset
from wonambi.trans import math, timefrequency
from typing import List, Tuple, Dict, Union
import numpy as np
from mednickdb_pysleep import pysleep_defaults, pysleep_utils
import pandas as pd
import warnings


def extract_band_power(edf_filepath: str,
                       bands: dict = pysleep_defaults.default_freq_bands,
                       chans_to_consider: List[str]=None,
                       start_time: float=None,
                       end_time: float=None,
                       epoch_len: int = pysleep_defaults.epoch_len) -> pd.DataFrame:
    """

    :param edf_filepath: The edf to extract bandpower for
    :param bands: bands to extract power in, if None, then defaults will be used i.e.
        bands = {
            'delta': (1, 4),
            'theta': (4, 7),
            'alpha': (8, 12),
            'sigma': (11, 16),
            'slow_sigma': (11, 13),
            'fast_sigma': (13, 16),
            'beta': (13, 30)
        }
    :param chans_to_consider: which channels to consider
    :param start_time: start time of the recording to extract band power for (when do epochs start), onset is measured from this
    :param end_time: end time of the recording to extract band power for
    :param epoch_len: how long a time bin you want your power to be averaged over
    :return: chan_epoch_band as a numpy array, and times, bands, chans
    """

    d = Dataset(edf_filepath)
    assert start_time is None or start_time >= 0
    assert end_time is None or end_time <= d.header['n_samples']/d.header['s_freq'], \
        "end time ("+ str(end_time) +") larger than record end!"+str(d.header['n_samples']/d.header['s_freq'])
    data = d.read_data(begtime=start_time, endtime=end_time, chan=chans_to_consider)
    power = timefrequency(data, method='spectrogram')
    abs_power = math(power, operator_name='abs')
    chan_time_freq = abs_power.data[0]
    all_chans = np.ones((chan_time_freq.shape[0],), dtype=bool)
    start_time = 0 if start_time is None else start_time
    time_axis = np.round(abs_power.axis['time'][0], 2) - start_time
    freq_axis = np.round(abs_power.axis['freq'][0], 2)
    chan_axis = abs_power.axis['chan'][0]
    freq_binsize = freq_axis[1] - freq_axis[0]
    assert epoch_len > 0, "epoch len must be greater than zero"
    times = np.arange(0,time_axis[-1],epoch_len)
    cont = []
    for band, freqs in bands.items():
        freq_mask = (freqs[0] <= freq_axis) & (freqs[1] >= freq_axis)
        for win_start in times:
            time_mask = (win_start < time_axis) & (time_axis < win_start + epoch_len)
            idx = np.ix_(all_chans, time_mask, freq_mask)
            if idx:
                chan_epoch_per_band = chan_time_freq[idx].mean(axis=1).mean(axis=1) / freq_binsize
            else:
                chan_epoch_per_band = np.zeros((len(chans_to_consider),))
            for chan, power in zip(chan_axis, chan_epoch_per_band):
                cont.append(pd.Series({'onset':win_start,
                                       'duration':epoch_len,
                                       'band':band.split('_')[0],
                                       'chan':chan,
                                       'power':power}))
    band_power = pd.concat(cont, axis=1).T.apply(lambda x: pd.to_numeric(x, errors='ignore'))
    return band_power


def extract_band_power_per_epoch(band_power_df: pd.DataFrame,
                                 epoch_offset: float=0,
                                 epoch_len: float=pysleep_defaults.epoch_len) -> pd.DataFrame:
    """
    Resample the bandpower df so that its an average per epoch
    :param band_power_df: band power df outputted from extract_band_power
    :param epoch_len: the new epoch len
    :param epoch_offset: the difference in seconds from when you want your sleep epochs to start compared to
    band power epochs (this is probably the differnece between epoch stsges and edf start)
    :return: resampled df
    """
    band_power_df['onset'] = band_power_df['onset'].apply(lambda x: pd.Timedelta(seconds=x))
    band_power_df = band_power_df.drop('duration', axis=1).set_index('onset')
    resampled_df = band_power_df.groupby(['chan', 'band']).resample(rule=str(epoch_len)+'s').mean().reset_index()
    resampled_df['onset'] = resampled_df['onset'].apply(lambda x: x.seconds)
    resampled_df['duration'] = epoch_len
    return resampled_df


def assign_band_power_stage(band_power_per_epoch_df: pd.DataFrame,
                            epochstages: list) -> Union[pd.DataFrame, dict]:
    """
    Extract band power per stage as a dataframe or dict
    :param band_power_per_epoch_df: Chan by Epoch by Band dataframe from extract_band_power_per_epoch
    :param epochstages: stages corresponding to each epoch
    :return: band power per stage
    """
    cont = []
    for (band, chan), power_df, in band_power_per_epoch_df.groupby(['band','chan']):
        assert power_df.shape[0] >= len(epochstages), "Missmatch in number of epochs and power per epoch, epoch len correct?"
        power_df = power_df.iloc[0:len(epochstages), :]
        power_df.loc[:, 'stage'] = np.array(epochstages)
        power_df.loc[:, 'stage_idx'] = np.arange(0,len(epochstages))
        cont.append(power_df)
    return pd.concat(cont, axis=0)


