from wonambi import Dataset
from wonambi.trans import math, timefrequency
from typing import List, Tuple, Dict, Union
import numpy as np
from mednickdb_pysleep.pysleep_defaults import epoch_len
import pandas as pd


def extract_band_power_per_epoch(edf_filepath: str,
                                 bands: dict = None,
                                 chans_to_consider: List[str]=None,
                                 start_time: float=None,
                                 end_time: float=None,
                                 epoch_len: int=epoch_len) -> Tuple[np.ndarray, dict, List[str]]:
    """

    :param edf_filepath: The edf to extract bandpower for
    :param bands: bands to extract power in, if None, then defaults will be used i.e.
        bands = {
            'delta': (1, 4),
            'theta': (4, 7),
            'alpha': (8, 12),
            'sigma': (11, 16),
            'slow_sigma': (11, 13),  # TODO whats the best fast/slow bands?
            'fast_sigma': (13, 16),
            'beta': (13, 30)
        }
    :param chans_to_consider: which channels to consider
    :param start_time: start time of the recording to extract band power for (when do epochs start)
    :param end_time: end time of the recording to extract band power for
    :return: chan_epoch_band as a numpy array, and bands
    """
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 7),
            'alpha': (8, 10),
            'sigma': (11, 16),
            'slow_sigma': (11, 13),  # TODO whats the best fast/slow bands?
            'fast_sigma': (13, 16),
            'beta': (16, 20)
        }
    d = Dataset(edf_filepath)
    data = d.read_data(begtime=start_time, endtime=end_time, chan=chans_to_consider)
    power = timefrequency(data, method='spectrogram')
    abs_power = math(power, operator_name='abs')
    chan_time_freq = abs_power.data[0]
    all_chans = np.ones((chan_time_freq.shape[0],), dtype=bool)

    band_cont = []
    time_axis = np.round(abs_power.axis['time'][0], 2)
    freq_axis = np.round(abs_power.axis['freq'][0], 2)
    chan_axis = abs_power.axis['chan'][0]
    freq_binsize = freq_axis[1] - freq_axis[0]
    assert epoch_len > 0, "epoch len must be greater than zero"
    for band, freqs in bands.items():
        freq_mask = (freqs[0] <= freq_axis) & (freqs[1] >= freq_axis)
        epoch_cont = []
        win_start = 0
        while win_start+epoch_len < time_axis[-1]:
            time_mask = (win_start < time_axis) & (time_axis < win_start + epoch_len)
            idx = np.ix_(all_chans, time_mask, freq_mask)
            chan_epoch_per_band = chan_time_freq[idx].mean(axis=1).mean(axis=1) / freq_binsize
            epoch_cont.append(chan_epoch_per_band)
            win_start += epoch_len
        band_cont.append(np.stack(epoch_cont, -1))
    chan_epoch_band = np.stack(band_cont, -1)
    return chan_epoch_band, bands, chan_axis


def extract_band_power_per_stage(chan_epoch_band_data: np.ndarray,
                                 epochstages: list,
                                 stages_to_consider: list=None,
                                 return_format: str= 'dict',
                                 ch_names: List[str]=None,
                                 band_names: List[str]=None) -> Union[pd.DataFrame, dict]:
    """
    Extract band power per stage as a dataframe or dict
    :param chan_epoch_band_data: Chan by Epoch by Band data
    :param epochstages: stages corresponding to each epoch
    :param stages_to_consider: which stages to extract for
    :param return_format: return format, either dataframe for a cols=[stage, chan, band, power] dataframe, or dict, for simpler {'stage':[chan*band]} dict
    :param ch_names: required if format is dataframe, channel names, with order that matches chan_epoch_band_data
    :param band_names: required if format is dataframe, band names, with order that matches chan_epoch_band_data
    :return: band power per stage
    """
    if return_format != 'dict':
        assert (ch_names is not None) and (band_names is not None), "freq and chan names required for dataframe output"

    epochstages = np.array(epochstages)
    if stages_to_consider is None:
        stages_to_consider = np.unique(epochstages)

    power_per_stage_dict = {}
    power_cont = []
    for stage in stages_to_consider:
        epochs_of_stage = (epochstages == stage).nonzero()[0]
        per_stage_data = chan_epoch_band_data[:, epochs_of_stage, :].mean(axis=1)
        if return_format == 'dict':
            power_per_stage_dict[stage] = per_stage_data
        else:
            for ch_idx, chan in enumerate(ch_names):
                for freq_idx, band in enumerate(band_names):
                    power_cont.append({'band': band, 'chan': chan, 'power': per_stage_data[ch_idx, freq_idx], 'stage':stage})

    if return_format == 'dict':
        return power_per_stage_dict
    else:
        return pd.DataFrame(power_cont)



