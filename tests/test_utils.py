from mednickdb_pysleep.utils import convert_epochstages_to_eegevents


def test_epochstage_to_eegevents():
    epochstages = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 1, 1, 1, 1]
    eegevents = convert_epochstages_to_eegevents(epochstages)
    ans_cols = [[0, 1, 2, 3, 2, 1], [150, 180, 150, 150, 150, 120], [0, 150, 330, 480, 630, 780], [None]*6]
    for ret_col, ans in zip(eegevents.columns, ans_cols):
        assert all(eegevents[ret_col].values == ans), "Col "+ret_col+' does not match'