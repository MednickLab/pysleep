import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + '/../mednickdb_pysleep/')
from artifact_detection import employ_buckelmueller, employ_hjorth
import yaml
import pickle
import pytest
import numpy as np


def test_employ_buckelmueller():
    edf_filename = os.path.join(file_dir, 'testfiles/example1_sleep_rec.edf')
    study_settings = yaml.safe_load(open(os.path.join(file_dir, 'testfiles/example1_study_settings.yaml'), 'rb'))
    epochstages_file = os.path.join(file_dir, 'testfiles/example1_epoch_stages.pkl')
    epochstages = pickle.load(open(epochstages_file, 'rb'))
    bad_epochs = employ_buckelmueller(edf_filepath=edf_filename,
                                      epochstages=epochstages,
                                      chans_to_consider=list(study_settings['known_eeg_chans'].keys()),
                                      start_offset=0,
                                      end_offset=3000)

    assert len(bad_epochs) > 0
    pytest.bad_epochs = bad_epochs
    pytest.edf_filename=edf_filename
    pytest.study_settings = study_settings

def test_employ_hjorth():
    bad_epochs, events = employ_hjorth(edf_filepath=pytest.edf_filename,
                               chans_to_consider=list(pytest.study_settings['known_eeg_chans'].keys()),
                               start_offset=0,
                               end_offset=3000,
                               return_events=True)
    assert len(set(bad_epochs) - set(pytest.bad_epochs)) > 0

    #bad_epochs = np.array(bad_epochs+pytest.bad_epochs)

    #do some visual checks
    # import matplotlib.pyplot as plt
    # import mne
    # edf = mne.io.read_raw_edf(pytest.edf_filename, preload=True)
    # bad_events = events[bad_epochs, 0]
    # edf = edf.set_annotations(mne.Annotations(onset=(bad_events/edf.info['sfreq']).tolist(),
    #                                           duration=[30]*bad_events.shape[0],
    #                                           description=['bad']*bad_events.shape[0],
    #                                           orig_time=edf.info['meas_date']))
    # edf.plot()
    # plt.show()