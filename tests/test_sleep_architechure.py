import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + '/../mednickdb_pysleep')
from sleep_dynamics import *
from sleep_architecture import *


def test_sleep_arch():
    data = {'epochstage':[0, 0, 0, -1, -1, -1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]}
    minutes_out, perc_out, total_mins = sleep_stage_architecture(data['epochstage'], stages_to_consider=(0,1,2,3,4))
    correct_ans_minutes = {0: 1.5, 1: 1.5, 2: 1.5, 3: 1.5, 4: 1.5}
    assert(minutes_out == correct_ans_minutes)
    assert(total_mins == 7.5)
    assert(perc_out == {0: 20.0, 1: 20.0, 2: 20.0, 3: 20.0, 4: 20.0})
    sleep_ef_out = sleep_efficiency(correct_ans_minutes, total_mins, wake_stages=[0])
    assert(sleep_ef_out == 0.8)
    tst = total_sleep_time(correct_ans_minutes, wake_stages=[0])
    assert(tst == 6)


def test_sleep_fragmentation():
    data = {'epochstage': [0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 0, 2, 3, 2, 0, 1, 2, 4]}
    frag_out = num_awakenings(data['epochstage'], waso_stage=0)
    assert (frag_out == 3)


def test_transition_counts():
    data = {'epochstage': [0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 0, 2, 3, 2, 0, 1, 2, 4]}
    zeroth, first, second = transition_counts(data['epochstage'], stages_to_consider=(0,1,2,3,4))
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

    zeroth_norm, first_norm, second_norm = transition_counts(data['epochstage'], normalize=True, stages_to_consider=(0,1,2,3,4))
    assert np.sum(zeroth_norm) == 1
    assert np.all(np.isnan(np.sum(first_norm, axis=1)) | (np.sum(first_norm, axis=1) == 1))
    assert np.all(np.isnan(np.sum(second_norm, axis=2)) | (np.sum(second_norm, axis=2) == 1))


def test_duration_dists():
    epoch_stage = [0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 0, 2, 3, 2, 0, 1, 2, 4]
    dist_out = bout_durations(epoch_stage, 30, stages_to_consider=(0,1,2,3,4))
    real_dists = {0: [1.5, 1.5, 0.5, 0.5], 1: [1.5, 0.5], 2: [0.5, 0.5, 0.5, 0.5], 3: [0.5], 4: [0.5]}
    assert(dist_out == real_dists)


def test_lights_on_off_and_sleep_latency():
    epoch_stage = ['unknown', 'wbso', 'wbso', 'wbso', 'n1', 'n1', 'n1', 'unknown', 'unknown', 'unknown',
                   'n2', 'waso', 'n2', 'n3', 'n2', 'n1', 'n1', 'n2', 'rem', 'unknown']
    lights_off, lights_on,  latency = lights_on_off_and_sleep_latency(epoch_stage)
    assert latency == 1.5
    assert lights_off == 0.5
    assert lights_on == 9.5

    epoch_stage = ['n1', 'n1', 'n1', 'unknown', 'unknown', 'unknown',
                   'n2', 'waso', 'n2', 'n3', 'n2', 'n1', 'n1', 'n2', 'rem']
    lights_off, lights_on, latency = lights_on_off_and_sleep_latency(epoch_stage)
    assert latency == 0
    assert lights_off == 0
    assert lights_on == 7.5

    epoch_stage = ['n1', 'n1', 'n1', 'n1', 'n1']
    lights_off, lights_on, latency = lights_on_off_and_sleep_latency(epoch_stage)
    assert latency == 0
    assert lights_off == 0
    assert lights_on == 2.5
