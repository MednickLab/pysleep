import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + '/../mednickdb_pysleep')
from scorefiles import extract_epochstages_from_scorefile, score_wake_as_waso_wbso_wase
from pysleep_utils import get_stagemap_by_name


def test_extract_epochstages_from_scorefile():
    stagemap = get_stagemap_by_name('hume')

    testfile1 = os.path.join(file_dir, 'testfiles/example3_scorefile.mat')
    epochstages, starttime = extract_epochstages_from_scorefile(testfile1, stagemap)

    correct_epochstages = ['unknown', 'unknown', 'unknown', 'wake', 'wake', 'wake', 'wake', 'wake', 'wake', 'wake']
    assert epochstages == correct_epochstages

    epochstages = score_wake_as_waso_wbso_wase(epochstages)
    correct_epochstages = ['unknown', 'unknown', 'unknown', 'wbso', 'wbso', 'wbso', 'wbso', 'wbso', 'wbso', 'wbso']
    assert epochstages == correct_epochstages

    testfile2 = os.path.join(file_dir, 'testfiles/example4_scorefile.mat')
    epochstages, starttime = extract_epochstages_from_scorefile(testfile2, stagemap)
    correct_epochstages = ['wake', 'wake', 'n1', 'n1', 'n2', 'n2', 'n3', 'n3', 'n2', 'n2']
    assert epochstages == correct_epochstages

    epochstages = score_wake_as_waso_wbso_wase(epochstages)
    correct_epochstages = ['wbso', 'wbso', 'n1', 'n1', 'n2', 'n2', 'n3', 'n3', 'n2', 'n2']
    assert epochstages == correct_epochstages