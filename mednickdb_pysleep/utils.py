import numpy as np
import pandas as pd
from itertools import groupby

def convert_epochstages_to_eegevents(epochstages, epoch_len=30):
    """
    converts stages formated as epochstages type i.e. [1 1 1 1 1 2 2 2 2 3 3 3 3]
    to eegevents type, with a onset column, duration column, and description column, and channel column
    :param epochstages: A list or np.darray of stages for each epoch
    :param epoch_len: the length of each epoch in seconds
    :return: converted stage info as a pd.DataFrame
    """
    stages = []
    durations = []
    for stage, group in groupby(epochstages):
        stages.append(stage)
        durations.append(len(list(group))*epoch_len)
    onsets = [0]+list(np.cumsum(durations))[:-1]
    eegevents = pd.DataFrame({'onset':onsets, 'description': stages, 'duration': durations})
    eegevents['channel'] = None
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
