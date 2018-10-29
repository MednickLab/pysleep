import numpy as np


def sleep_stage_architecture(epochstage, epoch_len=0.5, stages_to_consider=(0,1,2,3,4)):
    # read in the epochstage and check the size
    unique, counts = np.unique(epochstage, return_counts=True)
    mins_in_stage = {s:0 for s in stages_to_consider}
    for u, c in zip(unique, counts):
        if u in stages_to_consider:
            mins_in_stage[u] = c*epoch_len
    total_minutes = np.nansum([counts for stage, counts in mins_in_stage.items()])
    percent_in_stage = {k: 100 * v / total_minutes for k, v in mins_in_stage.items()}
    return mins_in_stage, percent_in_stage, total_minutes


def sleep_efficiency(mins_in_stage, total_minutes, wake_stage=0):
    return (total_minutes - mins_in_stage[wake_stage])/total_minutes


def total_sleep_time(mins_in_stage, wake_stage=0):
    return np.nansum([counts for stage, counts in mins_in_stage.items() if stage != wake_stage])


