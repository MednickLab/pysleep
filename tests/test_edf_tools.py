import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + '/../mednickdb_pysleep')
from edf_tools import write_edf_from_mne_raw_array, rereference
import pandas as pd
import numpy as np
import mne
import wonambi.ioeeg.edf as wnbi_edf_reader


def test_write_edf_from_mne_raw_array():
    raw = mne.io.read_raw_brainvision(os.path.join(os.path.dirname(__file__), 'testfiles/eeg_file_test.vhdr'))
    out_path = os.path.join(os.path.dirname(__file__), "testfiles/edf2_from_edf1.edf")
    write_edf_from_mne_raw_array(raw, out_path, ref_type='testing')
    raw2 = mne.io.read_raw_edf(out_path)
    diff = raw.get_data('P3') - raw2.get_data('P3')
    #assert 'testing' == get_ref
    assert np.sum(np.abs(diff))/diff.shape[1] < 1e-6


def test_reference():
    edf = mne.io.read_raw_brainvision(os.path.join(os.path.dirname(__file__), 'testfiles/eeg_file_test.vhdr'))
    edf.load_data()

    le_edf, le_ref = rereference(edf.copy(), desired_ref='linked_ear')
    cm_edf, cm_ref = rereference(edf.copy(), desired_ref='contra_mastoid')
    cm_from_le_edf, cm_from_le_ref = rereference(le_edf.copy(), desired_ref='contra_mastoid', current_ref='linked_ear')
    le_from_cm_edf, le_from_cm_ref = rereference(cm_edf.copy(), desired_ref='linked_ear', current_ref='contra_mastoid')
    assert np.sum(np.abs(cm_from_le_edf.get_data('P3') - cm_edf.get_data('P3'))) < 1e-12
    assert np.sum(np.abs(le_from_cm_edf.get_data('P3') - le_edf.get_data('P3'))) < 1e-12
    assert le_ref == le_from_cm_ref == 'linked_ear'
    assert cm_ref == cm_from_le_ref == 'contra_mastoid'


