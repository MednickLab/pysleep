import json
import numpy as np
from pprint import pprint



# function.py is a bad name for a file. function is a reserved word, never use reserved words as var names, class names, or file names

def minutes_in_stage(epochstage, epoch_len=0.5, stages_to_include_in_total = (0,1,2,3,4)):
    # read in the epochstage and check the size
    unique, counts = np.unique(epochstage, return_counts=True)
    stages_to_include_in_total_str = ['minutes_in_'+str(s) for s in stages_to_include_in_total]
    mins_in_stage_out = {'minutes_in_'+str(u): c*epoch_len for u, c in zip(unique, counts)}
    mins_in_stage_out['total_sleep_minutes'] = np.sum([counts for stage, counts in mins_in_stage_out.items() if stage in stages_to_include_in_total_str])
    return mins_in_stage_out

# this function could calculate the proportion of the data
def proportion_in_stage(epochstage):
    temp = minutes_in_stage(epochstage)
    total = temp['total_sleep_time']
    return {'0': temp['0'] / total, '1': temp['1'] / total, '2': temp['2'] / total, '3': temp['3'] / total}


def sleep_efficiency(epochstage):
    temp = minutes_in_stage(epochstage)
    return (temp['1'] + temp['2'] + temp['3']) / temp['total_sleep_time']

