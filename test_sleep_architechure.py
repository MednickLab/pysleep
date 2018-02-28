import json
from sleep_architecture import *

def test_total_sleep_time():
    json_file = 'CAPStudy_subjectid216_visit1.json'
    json_data = open(json_file)
    data = {'epochstage':[0, 0, 0, -1, -1,-1,1,1,1,2,2,2,3,3,3,4,4,4]}
    data_out = minutes_in_stage(data['epochstage'])
    correct_ans = {'minutes_in_-1': 1.5, 'minutes_in_0': 1.5, 'minutes_in_1': 1.5, 'minutes_in_2': 1.5, 'minutes_in_3': 1.5, 'minutes_in_4': 1.5, 'total_sleep_minutes': 7.5}
    assert(data_out == correct_ans)