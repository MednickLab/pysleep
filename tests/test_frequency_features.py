import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + '/../mednickdb_pysleep/')
from frequency_features import extract_band_power_per_stage, extract_band_power_per_epoch
from pysleep_defaults import sleep_stages
from pysleep_utils import pd_to_xarray_datacube
import pytest
import pickle


def test_extract_band_power_per_epoch():
    edf_filename = 'testfiles/example_sleep_rec.edf'
    band_power_per_epoch, bands, chans = extract_band_power_per_epoch(edf_filepath=edf_filename)
    assert band_power_per_epoch.shape == (2, 1229, 7)
    pytest.band_power_per_epoch = band_power_per_epoch
    pytest.bands = bands
    pytest.chans = chans


@pytest.mark.dependency(['test_extract_band_power_per_epoch'])
def test_extract_band_power_per_stage():
    epochstages_file = file_dir + '/testfiles/example_epoch_stages.pkl'
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
                                                        ch_names=pytest.chans,
                                                        )

    assert band_power_per_stage.shape[0] == 2*7*len(sleep_stages)

    dc = pd_to_xarray_datacube(band_power_per_stage, dim_cols=['stage', 'chan', 'band'], value_col='power')
    assert dc.shape == (4, 2, 7)