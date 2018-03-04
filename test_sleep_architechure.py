import json
from sleep_architecture import *
from sleep_fragmentation import *


def test_sleep_arch():
    data = {'epochstage':[0, 0, 0, -1, -1,-1,1,1,1,2,2,2,3,3,3,4,4,4]}
    data_out = minutes_in_stage(data['epochstage'])
    correct_ans_minutes = {'minutes_in_0': 1.5, 'minutes_in_1': 1.5, 'minutes_in_2': 1.5, 'minutes_in_3': 1.5, 'minutes_in_4': 1.5, 'total_minutes': 7.5}
    assert(data_out == correct_ans_minutes)
    perc_out = percent_in_stage(correct_ans_minutes)
    assert(perc_out == {'percent__0': 20.0, 'percent__1': 20.0, 'percent__2': 20.0, 'percent__3': 20.0, 'percent__4': 20.0})
    sleep_ef_out = sleep_efficiency(correct_ans_minutes, wake_stage=0)
    assert(sleep_ef_out == 0.8)


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
