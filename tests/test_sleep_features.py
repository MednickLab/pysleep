import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + '/../mednickdb_pysleep')
from mednickdb_pysleep.sleep_features import *
import time
import pytest
import pickle


def test_spindle_detection():
    # TODO make a real test case...
    start_time = time.time()
    edf = file_dir + '/testfiles/example_sleep_rec.edf'
    spindles = detect_spindles(edf_filepath=edf, algo='Ferrarelli2007')
    end_time = time.time()
    print('Spindles took', end_time-start_time)
    assert spindles is not None
    spindles.to_csv(file_dir + '/testfiles/example_spindle_events.csv', index=False)


def test_so_detection():
    # TODO make a real test case...
    start_time = time.time()
    edf = file_dir + '/testfiles/example_sleep_rec.edf'
    slow_oscillations = detect_slow_oscillation(edf_filepath=edf)
    end_time = time.time()
    print('SO took', end_time - start_time)
    assert slow_oscillations is not None
    pytest.slow_oscillations = slow_oscillations


def test_density_and_mean_features_calculations():
    #TODO should check actual spindle averages (assuming deterministic spindle algo)
    epochstages_file = file_dir + '/testfiles/example_epoch_stages.pkl'
    epoch_stages = pickle.load(open(epochstages_file, 'rb'))
    spindles = pd.read_csv(file_dir + '/testfiles/example_spindle_events.csv')
    spindle_events = assign_stage_to_feature_events(spindles, epoch_stages)
    channels = spindle_events['chan'].unique()
    stages = ['stage2', 'sws']
    features = sleep_feature_variables_per_stage(spindle_events, epoch_stages, stages_to_consider=stages)
    assert features.shape[0] == len(stages)
    expected_cols = ['av_density'] + ['av_' + col for col in spindle_events.columns]+list(spindle_events.columns)
    assert all([col in expected_cols for col in features.columns])
    features_per_chan = sleep_feature_variables_per_stage(spindle_events, epoch_stages, av_across_channels=False, stages_to_consider=stages)
    assert features_per_chan.shape[0] == len(stages)*len(channels)
    assert set(features_per_chan['chan'].unique()) == set(channels)
