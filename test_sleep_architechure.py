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


def test_trans_probs():
    data = {'epochstage': [0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 0, 2, 3, 2, 0, 1, 2, 4]}
    trans_out = transition_probabilities(data['epochstage'])
    real_trans = {'trans_p_from_any_to_0': 0.27272727272727271,
                  'trans_p_from_any_to_1': 0.18181818181818182,
                  'trans_p_from_any_to_2': 0.36363636363636365,
                  'trans_p_from_any_to_3': 0.090909090909090912,
                  'trans_p_from_any_to_4': 0.090909090909090912}
    assert(trans_out == real_trans)