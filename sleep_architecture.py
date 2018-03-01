import numpy as np

def minutes_in_stage(epochstage, epoch_len=0.5, stages_to_consider=(0,1,2,3,4)):
    # read in the epochstage and check the size
    unique, counts = np.unique(epochstage, return_counts=True)
    mins_in_stage_out = {'minutes_in_'+str(u): c*epoch_len for u, c in zip(unique, counts) if u in stages_to_consider}
    mins_in_stage_out['total_minutes'] = np.sum([counts for stage, counts in mins_in_stage_out.items()])
    return mins_in_stage_out

# this function could calculate the proportion of the data
def percent_in_stage(mins_in_stage):
    """takes minutes in stage as calculated by the function as input"""
    perc_in_stage = {k.replace('minutes_in', 'percent_'): 100*v/mins_in_stage['total_minutes'] for k, v in mins_in_stage.items()}
    del perc_in_stage['total_minutes']
    return perc_in_stage

def sleep_efficiency(mins_in_stage, wake_stage):
    wake_stage_name = 'minutes_in_' + str(wake_stage)
    return (mins_in_stage['total_minutes'] - mins_in_stage[wake_stage_name])/mins_in_stage['total_minutes']

