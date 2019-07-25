import pyedflib
import mne
from datetime import datetime
import warnings
import os
import pandas as pd
from typing import Tuple, List, Union
from mednickdb_pysleep.error_handling import EEGWarning

def write_edf_from_mne_raw_array(mne_raw: mne.io.RawArray, fname: str, ref_type='', annotations=False, new_date=False, picks=None, tmin=0, tmax=None, overwrite=True):
    """
    Saves the raw content of an MNE.io.Raw and its subclasses to
    a file using the EDF+ filetype
    pyEDFlib is used to save the raw contents of the RawArray to disk
    Parameters
    ----------
    mne_raw : mne.io.Raw
        An object with super class mne.io.Raw that contains the data
        to save
    fname : string
        File name of the new dataset. This has to be a new filename
        unless data have been preloaded. Filenames should end with .edf
    picks : array-like of int | None
        Indices of channels to include. If None all channels are kept.
    tmin : float | None
        Time in seconds of first sample to save. If None first sample
        is used.
    tmax : float | None
        Time in seconds of last sample to save. If None last sample
        is used.
    overwrite : bool
        If True, the destination file (if it exists) will be overwritten.
        If False (default), an error will be raised if the file exists.
    """
    if not issubclass(type(mne_raw), mne.io.BaseRaw):
        raise TypeError('Must be mne.io.Raw type')
    if not overwrite and os.path.exists(fname):
        raise OSError('File already exists. No overwrite.')
    # static settings
    if annotations:
        file_type = pyedflib.FILETYPE_EDFPLUS
    else:
        file_type = pyedflib.FILETYPE_EDF

    sfreq = mne_raw.info['sfreq']
    date = datetime.now().strftime('%d %b %Y %H:%M:%S') if new_date \
        else (datetime.fromtimestamp(mne_raw.info['meas_date'][0])).strftime('%d %b %Y %H:%M:%S')
    first_sample = int(sfreq*tmin)
    last_sample  = int(sfreq*tmax) if tmax is not None else None


    # convert data
    channels = mne_raw.get_data(picks,
                                start = first_sample,
                                stop  = last_sample)

    # convert to microvolts to scale up precision
    channels *= 1e6

    # set conversion parameters
    dmin, dmax = [-32768,  32767]
    pmin, pmax = [channels.min(), channels.max()]
    n_channels = len(channels)

    # create channel from this
    try:
        f = pyedflib.EdfWriter(fname,
                               n_channels=n_channels,
                               file_type=file_type)

        channel_info = []
        data_list = []

        for i in range(n_channels):
            ch_dict = {'label': mne_raw.ch_names[i],
                       'dimension': 'uV',
                       'sample_rate': sfreq,
                       'physical_min': pmin,
                       'physical_max': pmax,
                       'digital_min':  dmin,
                       'digital_max':  dmax,
                       'transducer': '',
                       'prefilter': ''}

            channel_info.append(ch_dict)
            data_list.append(channels[i])

        f.setTechnician('mednickdb')
        f.setEquipment('ref='+ref_type)
        f.setSignalHeaders(channel_info)
        f.setStartdatetime(date)
        f.writeSamples(data_list)
    except Exception as e:
        raise IOError('EDF could not be written') from e

    finally:
        f.close()
    return True


def rereference(edf: mne.io.RawArray, desired_ref: str, current_ref: str=None, pick_chans: list=None) -> Tuple[mne.io.RawArray, str]:
    """
    reference an edf. M1 and M2 should be included, but NOT as eeg stype in the mne.io.RawArray.
    contra_mastoid reref will be undone if needed but this requires M1 and M2 PRIOR to any referencing.
    :param edf: edf to re-ref
    :param current_ref: current reference type, can be "contra_mastoid",'linked_ear' or a channel name
    :param desired_ref: current reference type, can be "contra_mastoid",'linked_ear' or a channel name.
    If None, assumed it is a common channels
    :return:
    """
    if pick_chans is None:
        chans = edf.ch_names
    else:
        chans = pick_chans
    if current_ref == desired_ref:
        return edf, current_ref

    if desired_ref in ['linked_ear'] and 'M1' not in chans or 'M2' not in  chans:
        warnings.warn('Trying to reference to linked ear, but missing M1 and M2 channels. EEG file will not be re-referenced', EEGWarning)
        return edf, current_ref


    if current_ref == 'contra_mastoid':
        to_reref = [ch for ch in chans if ch not in ['M1','M2']]
        left = [ch for ch in to_reref if len([n for n in ['1','3','5','7','9'] if n in ch])>0]
        right = [ch for ch in to_reref if len([n for n in ['2','4','6','8','10'] if n in ch])>0]
        if len(left) > 0 and 'M2' not in chans:
            warnings.warn(
                'Trying to reference to left channels to M2 ear, but missing M2 channel. left channels cannot be unreferenced')
            left_ref = []
            left = []
        else:
            left_ref = ['M2'] * len(left)
        if len(right) > 0 and 'M1' not in chans:
            warnings.warn(
                'Trying to reference to right channels to M1 ear, but missing M1 channel. right channels cannot be unreferenced')
            right_ref = []
            right = []
        else:
            right_ref = ['M1'] * len(right)
        edf = edf.apply_function(lambda x: -x, picks=['M1', 'M2'])
        edf = mne.set_bipolar_reference(edf, left+right, left_ref+right_ref, drop_refs=False, verbose=False)
        edf = edf.drop_channels(left + right)
        edf.rename_channels({ch: ch.split('-')[0] for ch in edf.ch_names})
        edf = edf.apply_function(lambda x: -x, picks=['M1', 'M2'])

    ref_type = desired_ref
    if desired_ref == 'contra_mastoid':
        if current_ref == 'linked_ear':
            edf = edf.apply_function(lambda x: -x, picks=['M1','M2'])
            edf, _ = mne.set_eeg_reference(edf, ref_channels=['M1', 'M2'], verbose=False)
            edf = edf.apply_function(lambda x: -x, picks=['M1', 'M2'])
        to_reref = [ch for ch in chans if ch not in ['M1','M2']]
        left = [ch for ch in to_reref if len([n for n in ['1','3','5','7','9','z'] if n in ch])>0]
        right = [ch for ch in to_reref if len([n for n in ['2','4','6','8','10'] if n in ch])>0]
        if len(left) > 0 and 'M2' not in chans:
            warnings.warn(
                'Trying to reference to left channels to M2 ear, but missing M2 channel. left channels will not be re-referenced')
            left_ref = []
            left = []
            ref_type = 'contra_right_only'
        else:
            left_ref = ['M2'] * len(left)
        if len(right) > 0 and 'M1' not in chans:
            warnings.warn(
                'Trying to reference to right channels to M1 ear, but missing M1 channel. right channels will not be re-referenced')
            right_ref = []
            right = []
            ref_type = 'contra_left_only'
        else:
            right_ref = ['M1'] * len(right)
        edf = mne.set_bipolar_reference(edf, left+right, left_ref+right_ref, drop_refs=False, verbose=False)
        edf = edf.drop_channels(left + right)
        edf.rename_channels({ch:ch.split('-')[0] for ch in edf.ch_names})
    elif desired_ref == 'linked_ear':
        edf, _ = mne.set_eeg_reference(edf, ref_channels=['M1','M2'], verbose=False)
    else:
        edf, _ = mne.set_eeg_reference(edf, ref_channels=desired_ref, verbose=False)

    return edf, ref_type


def mne_annotations_to_dataframe(annot: mne.Annotations):
    return pd.DataFrame({'onset':annot.onset, 'description': annot.description, 'duration': annot.duration})


def mne_annotations_from_dataframe(df, orig_time=0):
    return mne.Annotations(df['onset'].values.astype(float)+orig_time,
                    df['duration'].values.astype(float),
                    df['description'].values.astype(str))


def add_events_df_to_mne_raw(mne_raw: mne.io.RawArray, events_df, orig_time):
    if mne_raw.annotations:
        mne_raw.annotations.append(onset=events_df['onset']+orig_time,
                                   duration=events_df['duration'],
                                   description=events_df['description'])
        return mne_raw
    return mne_raw.set_annotations(mne_annotations_from_dataframe(events_df, orig_time))
