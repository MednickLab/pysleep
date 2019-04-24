import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + '/../mednickdb_pysleep/')
from frequency_features import extract_band_power_per_stage, extract_band_power_per_epoch
from pysleep_defaults import sleep_stages
from pysleep_utils import pd_to_xarray_datacube
import pytest
import pickle
import yaml


def test_extract_band_power_per_epoch():
    edf_filename = os.path.join(file_dir, 'testfiles/example1_sleep_rec.edf')
    study_settings = yaml.safe_load(open(os.path.join(file_dir, 'testfiles/example1_study_settings.yaml'),'rb'))
    chans_to_consider = list(study_settings['known_eeg_chans'].keys())
    band_power_per_epoch, bands, chans = extract_band_power_per_epoch(edf_filepath=edf_filename,
                                                                      chans_to_consider=chans_to_consider)
    assert band_power_per_epoch.shape == (2, 100, 8)
    pytest.band_power_per_epoch = band_power_per_epoch
    pytest.bands = bands
    pytest.chans = chans


@pytest.mark.dependency(['test_extract_band_power_per_epoch'])
def test_extract_band_power_per_stage():
    epochstages_file = os.path.join(file_dir,'testfiles/example1_epoch_stages.pkl')
    epoch_stages = pickle.load(open(epochstages_file, 'rb'))
    try:
        extract_band_power_per_stage(pytest.band_power_per_epoch,
                                     epoch_stages,
                                     return_format='dataframe',
                                     stages_to_consider=sleep_stages,
                                     )
    except AssertionError:
        pass
    else:
        raise AssertionError("missing var assert not triggered")

    band_power_per_stage = extract_band_power_per_stage(pytest.band_power_per_epoch,
                                                        epoch_stages,
                                                        return_format='dataframe',
                                                        stages_to_consider=sleep_stages,
                                                        band_names=list(pytest.bands.keys()),
                                                        ch_names=list(pytest.chans))

    assert band_power_per_stage.shape[0] == 2*8*len(sleep_stages)

    dc = pd_to_xarray_datacube(band_power_per_stage, dim_cols=['stage', 'chan', 'band'], value_col='power')
    assert dc.shape == (len(sleep_stages), 2, 8)