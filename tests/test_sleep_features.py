import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + '/../mednickdb_pysleep/')
from sleep_features import detect_spindles, \
    detect_slow_oscillation, assign_stage_to_feature_events, \
    sleep_feature_variables_per_stage, detect_rems, \
    load_and_slice_data_for_feature_extraction, extract_features
from mednickdb_pysleep import sleep_architecture
import time
import pytest
import pickle
import numpy as np
import pandas as pd
import yaml


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
                                                      start_offset=0,
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
                                                      start_offset=0,
                                                      end_offset=3000,
                                                      chans_to_consider=chans_to_consider)
    slow_oscillations = detect_slow_oscillation(data=data, start_offset=0)
    end_time = time.time()
    print('SO took', end_time - start_time)
    assert slow_oscillations is not None
    pytest.slow_oscillations_df = slow_oscillations


def test_feature_extraction():
    edf = os.path.join(file_dir, 'testfiles/example1_sleep_rec.edf')
    study_settings = yaml.safe_load(open(os.path.join(file_dir, 'testfiles/example1_study_settings.yaml'), 'rb'))
    chans_to_consider = list(study_settings['known_eeg_chans'].keys())
    epochstages_file = file_dir + '/testfiles/example1_epoch_stages.pkl'
    epochstages = pickle.load(open(epochstages_file, 'rb'))

    features_df = extract_features(edf_filepath= edf,
                                   epochstages=epochstages,
                                   offset_between_epochstages_and_edf=0,
                                   end_offset=3000,
                                   chans_for_spindles=chans_to_consider,
                                   chans_for_slow_osc=chans_to_consider,
                                   spindle_algo='Ferrarelli2007')

    features_df = features_df.drop(['stage','stage_idx'], axis=1)
    features_sep = pd.concat([pytest.spindles_df, pytest.slow_oscillations_df], axis=0, sort=False)

    assert np.all(features_sep.fillna(-1).values == features_df.fillna(-1).values)




# def test_rem_detection():
#     edf = os.path.join(file_dir, 'testfiles/example2_sleep_rec.edf')
#     epochstages_file = file_dir + '/testfiles/example2_epoch_stages.pkl'
#     epochstages = pickle.load(open(epochstages_file, 'rb'))
#     data = load_and_slice_data_for_feature_extraction(edf_filepath=edf,
#                                                       epochstages=epochstages,
#                                                       chans_to_consider=['Left Eye-A2','Right Eye-A1'],
#                                                       stages_to_consider=['rem']
#                                                       )
#     rem_locs_df = detect_rems(edf, data, 'Left Eye-A2', 'Right Eye-A1')
#     assert rem_locs_df is not None
#     assert rem_locs_df.shape[0] > 0
#     pytest.rem_locs_df = rem_locs_df


def test_density_and_mean_features_calculations():
    #TODO should check actual spindle averages (assuming deterministic spindle algo)
    epochstages_file = os.path.join(file_dir, 'testfiles/example1_epoch_stages.pkl')
    epoch_stages = pickle.load(open(epochstages_file, 'rb'))
    spindle_events = assign_stage_to_feature_events(pytest.spindles_df, epoch_stages)
    channels = spindle_events['chan'].unique()
    stages = ['n2', 'n3']

    mins_df = sleep_architecture.sleep_stage_architecture(epoch_stages,
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
    chan_stage_missing_spindles = 1
    assert features_per_chan.shape[0] == len(stages)*len(channels)-chan_stage_missing_spindles
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
