import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + '/../mednickdb_pysleep/')
from frequency_features import assign_band_power_stage, extract_band_power, extract_band_power_per_epoch
from pysleep_defaults import sleep_stages
from pysleep_utils import pd_to_xarray_datacube
import pytest
import pickle
import yaml


def test_extract_band_power():
    edf_filename = os.path.join(file_dir, 'testfiles/example1_sleep_rec.edf')
    study_settings = yaml.safe_load(open(os.path.join(file_dir, 'testfiles/example1_study_settings.yaml'),'rb'))
    chans_to_consider = list(study_settings['known_eeg_chans'].keys())
    band_power_per_epoch = extract_band_power(edf_filepath=edf_filename,
                                              chans_to_consider=chans_to_consider,
                                              start_time=0,
                                              end_time=3000,
                                              epoch_len=3)
    assert band_power_per_epoch.shape[0] == 2*100*10*8
    pytest.band_power = band_power_per_epoch

@pytest.mark.dependency(['test_extract_band_power'])
def test_extract_band_power_per_epoch():
    pytest.band_power_per_epoch = extract_band_power_per_epoch(pytest.band_power)


@pytest.mark.dependency(['test_extract_band_power_per_epoch'])
def test_extract_band_power_per_stage():
    epochstages_file = os.path.join(file_dir,'testfiles/example1_epoch_stages.pkl')
    epoch_stages = pickle.load(open(epochstages_file, 'rb'))
    band_power_w_stage = assign_band_power_stage(pytest.band_power_per_epoch, epoch_stages)

    band_power_w_stage = band_power_w_stage.drop(['onset','duration'], axis=1)
    band_power_w_stage = band_power_w_stage.loc[band_power_w_stage['stage'].isin(sleep_stages), :]
    band_power_per_stage = band_power_w_stage.groupby(['chan', 'band', 'stage']).mean().reset_index()

    assert band_power_per_stage.shape[0] == 2*8*len(sleep_stages)

    dc = pd_to_xarray_datacube(band_power_per_stage, dim_cols=['stage', 'chan', 'band'], value_col='power')
    assert dc.shape == (len(sleep_stages), 2, 8)