import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + '/../mednickdb_pysleep')
from scorefiles import extract_epochstages_from_scorefile, score_wake_as_waso_wbso_wase
from pysleep_utils import utc_epochnum_to_local_datetime
import wonambi
import datetime


def test_starttimes():

    pass

    # wo_edftype1 = wonambi.ioeeg.edf.Edf(os.path.join(file_dir, 'testfiles/edftype1_scorefile.edf'))
    # assert wo_edftype1.hdr['start_time'] == datetime.datetime.strptime('2014-07-08 23:05:09', "%Y-%m-%d %H:%M:%S")
    #
    # edftype1 = mne.io.read_raw_edf(os.path.join(file_dir, 'testfiles/edftype1_scorefile.edf'))
    # print(utc_epochnum_to_local_datetime(edftype1['meas_date'][0]))
    #
    # edf1 = mne.io.read_raw_edf(os.path.join(file_dir, 'testfiles/edf1.edf'))
    # print(utc_epochnum_to_local_datetime(edf1['meas_date'][0]))
    #
    # edftype2 = mne.io.read_raw_edf(os.path.join(file_dir, 'testfiles/edftype2_scorefile.edf'))
    # print(utc_epochnum_to_local_datetime(edftype2['meas_date'][0]))
    #
    # edftype3 = mne.io.read_raw_edf(os.path.join(file_dir, 'testfiles/edftype3_scorefile.edf'))
    # print(utc_epochnum_to_local_datetime(edftype3['meas_date'][0]))