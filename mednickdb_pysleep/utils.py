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