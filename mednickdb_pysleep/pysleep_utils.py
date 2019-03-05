import numpy as np
import pandas as pd
from itertools import groupby


def convert_epochstages_to_eegevents(epochstages: dict, epoch_len: int=30, start_offset: float=0):
    """
    converts stages formated as epochstages type i.e. [1 1 1 1 1 2 2 2 2 3 3 3 3]
    to eegevents type, with a onset column, duration column, and description column, and channel column
    :param epochstages: A list or np.darray of stages for each epoch
    :param epoch_len: the length of each epoch in seconds
    :param start_offset: the offset of the first epoch from the edf that these stages came from, in seconds
    :return: converted stage info as a pd.DataFrame
    """
    stages = []
    durations = []
    for stage, group in groupby(epochstages):
        stages.append(stage)
        durations.append(len(list(group))*epoch_len)
    onsets = [start_offset]+list(start_offset+np.cumsum(durations))[:-1]
    eegevents = pd.DataFrame({'onset': onsets, 'description': stages, 'duration': durations})
    eegevents['channel'] = None
    eegevents['eventtype'] = 'stages'
    eegevents = eegevents.sort_values('onset')
    return eegevents


def fill_unknown_stages(epoch_stages, stages_to_fill=(-1, -2, -3), fill_direction='forward'):
    """
    fill the unknown stages with the previous or the next stage, remove unknown stages at start or end of record
    :param epoch_stages: A list or np.darray of stages for each epoch either with or without self transitions
    :param stages_to_fill: which stages are unknown or to be filled
    :param fill_direction: 'forward' or 'backward'. forward: overwrite the unknown stage with the previous stage, backward,
    overwrite with next stage.
    :return: epoch_stages list with unknown stages filled
    """
    #This algorithm got way out of hand, i tried to get fancy and have only one loop, but ended having to do a copy :(
    epoch_stages_ = epoch_stages.copy()
    start = 0
    end = len(epoch_stages_)
    step = 1
    start_good = 0
    found_good = False
    for idx in range(start, end, step):
        if not found_good:
            if epoch_stages_[idx] not in stages_to_fill:
                if fill_direction == 'backward':
                    break
                else:
                    found_good = True
            else:
                start_good = idx+1
        if fill_direction == 'forward':
            epoch_stages_[idx] = epoch_stages_[idx - step] if epoch_stages_[idx] in stages_to_fill else epoch_stages_[idx]

    start = len(epoch_stages_) - 1
    end = -1
    step = -1
    end_good = len(epoch_stages_)
    found_good = False
    for idx in range(start, end, step):
        if not found_good:
            if epoch_stages[idx] not in stages_to_fill:
                if fill_direction == 'forward':
                    break
                else:
                    found_good = True
            else:
                end_good = idx
        if fill_direction == 'backward' and idx != start:
            epoch_stages_[idx] = epoch_stages_[idx - step] if epoch_stages_[idx] in stages_to_fill else epoch_stages_[idx]

    return epoch_stages_[start_good:end_good]


def overlap(start1: float, end1: float, start2: float, end2: float) -> float:
    """how much does the range (start1, end1) overlap with (start2, end2)
    Looks strange, but algorithm is tight and tested.

    Args:
        start1: start of interval 1, in any unit
        end1: end of interval 1
        start2: start of interval 2
        end2: end of interval 2

    Returns:
        overlap of intervals in same units as supplied."""
    return max(max((end2 - start1), 0) - max((end2 - end1), 0) - max((start2 - start1), 0), 0)
