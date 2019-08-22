import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + '/../mednickdb_pysleep/')
from sleep_features import detect_spindles, \
    detect_slow_oscillation, assign_stage_to_feature_events, \
    sleep_feature_variables_per_stage, detect_rems, \
    load_and_slice_data_for_feature_extraction, extract_features, \
    detect_slow_osc_spindle_overlap
from process_sleep_record import extract_eeg_variables
from sleep_architecture import sleep_stage_architecture
from pysleep_defaults import load_matlab_detectors
import time

import pytest
import pickle
import numpy as np
import pandas as pd
import yaml
import os

def test_spindle_detection():
    # TODO make a real test case...
    start_time = time.time()
    edf = os.path.join(file_dir, 'testfiles/example1_sleep_rec.edf')
    study_settings = yaml.safe_load(open(os.path.join(file_dir, 'testfiles/example1_study_settings.yaml'), 'rb'))
    chans_to_consider = list(study_settings['known_eeg_chans'].keys())
    epochstages_file = file_dir + '/testfiles/example1_epoch_stages.pkl'
    epochstages = pickle.load(open(epochstages_file, 'rb'))

    data = load_and_slice_data_for_feature_extraction(edf_filepath=edf,
                                                      epochstages=epochstages,
                                                      bad_segments = [i for i,s in enumerate(epochstages) if s == 'n2'],
                                                      epochoffset_secs=0,
                                                      end_offset=3000,
                                                      chans_to_consider=chans_to_consider)
    spindles = detect_spindles(data=data, algo='Ferrarelli2007', start_offset=0)

    spindles = assign_stage_to_feature_events(spindles, epochstages)
    assert not any(spindles['stage'] == 'n2')


    data = load_and_slice_data_for_feature_extraction(edf_filepath=edf,
                                                      epochstages=epochstages,
                                                      epochoffset_secs=0,
                                                      end_offset=3000,
                                                      chans_to_consider=chans_to_consider)

    spindles = detect_spindles(data=data, algo='Ferrarelli2007', start_offset=0)
    end_time = time.time()
    print('Spindles took', end_time-start_time)
    assert spindles is not None
    pytest.spindles_df = spindles


def test_so_detection():
    # TODO make a real test case...
    start_time = time.time()
    edf = os.path.join(file_dir,'testfiles/example1_sleep_rec.edf')
    study_settings = yaml.safe_load(open(os.path.join(file_dir, 'testfiles/example1_study_settings.yaml'), 'rb'))
    chans_to_consider = list(study_settings['known_eeg_chans'].keys())
    epochstages_file = file_dir + '/testfiles/example1_epoch_stages.pkl'
    epochstages = pickle.load(open(epochstages_file, 'rb'))
    data = load_and_slice_data_for_feature_extraction(edf_filepath=edf,
                                                      epochstages=epochstages,
                                                      epochoffset_secs=0,
                                                      end_offset=3000,
                                                      chans_to_consider=chans_to_consider)
    slow_oscillations = detect_slow_oscillation(data=data, start_offset=0)
    end_time = time.time()
    print('SO took', end_time - start_time)
    assert slow_oscillations is not None
    pytest.slow_oscillations_df = slow_oscillations


def test_coupling_detection():
    features_df = pd.concat([pytest.spindles_df, pytest.slow_oscillations_df], axis=0, sort=False)
    pytest.features_df = features_df
    coupling = detect_slow_osc_spindle_overlap(features_df,
                                               coupling_secs=1.2,
                                               as_bool=True)

    assert any(coupling['coupled_before']) and any(coupling['coupled_after'])

def test_rem_detection():
    if load_matlab_detectors and os.name != 'nt' and ('yetton_rem_detector' in sys.modules):
        edf = os.path.join(file_dir, 'testfiles/example2_sleep_rec.edf')
        epochstages_file = file_dir + '/testfiles/example2_epoch_stages.pkl'
        epochstages = pickle.load(open(epochstages_file, 'rb'))
        print('Beginning rem detection, this may take sometime')
        data = load_and_slice_data_for_feature_extraction(edf_filepath=edf,
                                                          epochstages=epochstages,
                                                          chans_to_consider=['Left Eye-A2','Right Eye-A1'],
                                                          stages_to_consider=['rem']
                                                          )
        rem_locs_df = detect_rems(edf, data, 'Left Eye-A2', 'Right Eye-A1')
        assert rem_locs_df is not None
        assert rem_locs_df.shape[0] > 0
        rem_locs_df = assign_stage_to_feature_events(rem_locs_df, epochstages)
        assert np.all(rem_locs_df['stage'] == 'rem')
        pytest.rem_locs_df = rem_locs_df


def test_feature_extraction():
    edf = os.path.join(file_dir, 'testfiles/example1_sleep_rec.edf')
    study_settings = yaml.safe_load(open(os.path.join(file_dir, 'testfiles/example1_study_settings.yaml'), 'rb'))
    chans_to_consider = list(study_settings['known_eeg_chans'].keys())
    epochstages_file = file_dir + '/testfiles/example1_epoch_stages.pkl'
    epochstages = pickle.load(open(epochstages_file, 'rb'))

    features_df = extract_features(edf_filepath= edf,
                                   epochstages=epochstages,
                                   epochoffset_secs=0,
                                   end_offset=3000,
                                   chans_for_spindles=chans_to_consider,
                                   chans_for_slow_osc=chans_to_consider,
                                   spindle_algo='Ferrarelli2007')

    features_df = features_df.drop(['stage','stage_idx','coupled_before','coupled_after'], axis=1)

    pytest.features_df = pytest.features_df.drop(['coupled_before','coupled_after'], axis=1)

    assert np.all(pytest.features_df.fillna(-1).values == features_df.fillna(-1).values)


def test_density_and_mean_features_calculations():
    #TODO should check actual spindle averages (assuming deterministic spindle algo)
    epochstages_file = os.path.join(file_dir, 'testfiles/example1_epoch_stages.pkl')
    epoch_stages = pickle.load(open(epochstages_file, 'rb'))
    spindle_events = assign_stage_to_feature_events(pytest.spindles_df, epoch_stages)
    channels = spindle_events['chan'].unique()
    stages = ['n2', 'n3']

    mins_df = sleep_stage_architecture(epoch_stages,
                                       return_type='dataframe')

    features = sleep_feature_variables_per_stage(spindle_events,
                                                 mins_in_stage_df=mins_df,
                                                 stages_to_consider=stages)
    assert features.shape[0] == len(stages)
    expected_cols = ['density', 'count'] + ['av_' + col for col in spindle_events.columns]+list(spindle_events.columns)
    assert all([col in expected_cols for col in features.columns])
    features_per_chan = sleep_feature_variables_per_stage(spindle_events,
                                                 av_across_channels=False,
                                                 mins_in_stage_df=mins_df,
                                                 stages_to_consider=stages)

    assert features_per_chan.shape[0] == len(stages)*len(channels)
    assert set(features_per_chan['chan'].unique()) == set(channels)
    assert set(features_per_chan['stage'].unique()) == set(stages)

    # test if density is 0 for channels that dont have spindles
    channels = np.append(channels, 'F9').tolist()
    features_per_chan = sleep_feature_variables_per_stage(spindle_events,
                                                          av_across_channels=False,
                                                          mins_in_stage_df=mins_df,
                                                          stages_to_consider=stages,
                                                          channels=channels)
    assert set(features_per_chan['chan'].unique()) == set(channels)
    assert all(0 == features_per_chan.loc[features_per_chan['chan'] == 'F9', 'count'])
