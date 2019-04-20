import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + '/../mednickdb_pysleep')
from pysleep_utils import *


def test_epochstage_to_eegevents():
    epochstages = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1]
    eegevents = convert_epochstages_to_eegevents(epochstages)
    ans_cols = [[0, 1, 2, 3, 2, 1], [150, 180, 150, 150, 150, 120], [0, 150, 330, 480, 630, 780], [None]*6]
    for ret_col, ans in zip(eegevents.columns, ans_cols):
        assert all(eegevents[ret_col].values == ans), "Col "+ret_col+' does not match'


def test_fill_unknown_stages():
    epochstages = [0, 0, -1, 1, 2, -2, -3, 3, 2, 2, 2, 1, 1]
    filled_epoch_stages = fill_unknown_stages(epochstages.copy(), fill_direction='forward', stages_to_fill=(-3, -2, -1))
    forward_ans = [0, 0, 0, 1, 2, 2, 2, 3, 2, 2, 2, 1, 1]
    assert len(forward_ans) == len(filled_epoch_stages)
    assert all([f == a for f, a in zip(filled_epoch_stages, forward_ans)])

    filled_epoch_stages = fill_unknown_stages(epochstages, fill_direction='backward', stages_to_fill=(-3, -2, -1))
    back_ans = [0, 0, 1, 1, 2, 3, 3, 3, 2, 2, 2, 1, 1]
    assert len(back_ans) == len(filled_epoch_stages)
    assert all([f == a for f, a in zip(filled_epoch_stages, back_ans)])

    epochstages = [-1, 0, -1, 1, 2, -2, -3, 3, 2, 2, 2, 1, -1]
    filled_epoch_stages = fill_unknown_stages(epochstages.copy(), fill_direction='forward', stages_to_fill=(-3, -2, -1))
    forward_ans = [0, 0, 1, 2, 2, 2, 3, 2, 2, 2, 1]
    assert len(forward_ans) == len(filled_epoch_stages)
    assert all([f == a for f, a in zip(filled_epoch_stages, forward_ans)])

    filled_epoch_stages = fill_unknown_stages(epochstages, fill_direction='backward', stages_to_fill=(-3, -2, -1))
    back_ans = [0, 1, 1, 2, 3, 3, 3, 2, 2, 2, 1]
    assert len(back_ans) == len(filled_epoch_stages)
    assert all([f == a for f, a in zip(filled_epoch_stages, back_ans)])

    ret = fill_unknown_stages([0, 0, 1, 1, 2, 2, 3, 3, 2, 2], fill_direction='forward', stages_to_fill=(-3, -2, -1))
    good = [0, 0, 1, 1, 2, 2, 3, 3, 2, 2]
    assert all([f == a for f, a in zip(ret, good)])