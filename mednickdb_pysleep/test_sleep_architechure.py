from mednickdb_pysleep.sleep_fragmentation import *
from mednickdb_pysleep.sleep_architecture import *


def test_sleep_arch():
    data = {'epochstage':[0, 0, 0, -1, -1,-1,1,1,1,2,2,2,3,3,3,4,4,4]}
    minutes_out, perc_out, total_mins = sleep_stage_architecture(data['epochstage'])
    correct_ans_minutes = {0: 1.5, 1: 1.5, 2: 1.5, 3: 1.5, 4: 1.5}
    assert(minutes_out == correct_ans_minutes)
    assert(total_mins == 7.5)
    assert(perc_out == {0: 20.0, 1: 20.0, 2: 20.0, 3: 20.0, 4: 20.0})
    sleep_ef_out = sleep_efficiency(correct_ans_minutes, total_mins, wake_stage=0)
    assert(sleep_ef_out == 0.8)
    tst = total_sleep_time(correct_ans_minutes, wake_stage=0)
    assert(tst==6)


def test_sleep_fragmentation():
    data = {'epochstage': [0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 0, 2, 3, 2, 0, 1, 2, 4]}
    frag_out = num_awakenings(data['epochstage'])
    assert (frag_out == 3)


def test_transition_counts():
    data = {'epochstage': [0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 0, 2, 3, 2, 0, 1, 2, 4]}
    zeroth, first, second = transition_counts(data['epochstage'])
    real_zeroth = [3, 2, 4, 1, 1]
    assert(np.all(zeroth == np.array(real_zeroth)))
    real_first = [[0, 2, 2, 0, 0],
                  [1, 0, 1, 0, 0],
                  [2, 0, 0, 1, 1],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0]]
    assert (np.all(first == np.array(real_first)))
    real_second = [[[2, 1, 1, 0, 0],
                  [0, 1, 1, 0, 0],
                  [1, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]],

                 [[1, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]],

                 [[0, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0]],

                 [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]],

                 [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]]
    assert (np.all(second == np.array(real_second)))


def test_durration_dists():
    data = {'epochstage': [0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 0, 2, 3, 2, 0, 1, 2, 4]}
    dist_out = duration_distributions(data['epochstage'], 30)
    real_dists = {0: [90, 90, 30, 30], 1: [90, 30], 2: [30, 30, 30, 30], 3: [30], 4: [30]}
    assert(dist_out == real_dists)
