import numpy as np


def sleep_stage_architecture(epoch_stage, epoch_len=30, stages_to_consider=(0, 1, 2, 3, 4)):
    """
    Calculate the traditional measures of sleep (mins in stage, percent in stage, total minutes of sleep (total sleep time))
    :param epoch_stage: the pattern of sleep stages with self transitions, e.g. [0 0 1 1 1 2 2 2 1]
    :param epoch_len: length of an epoch in seconds
    :param stages_to_consider: which stages we should use for calcualtions, will default to sleep stages and wake (0,1,2,3,4)
    :return: a tuple of (minutes in stage, percent in stage, total minutes for stages to consider)
    """
    unique, counts = np.unique(epoch_stage, return_counts=True)
    mins_in_stage = {s: 0 for s in stages_to_consider}
    for u, c in zip(unique, counts):
        if u in stages_to_consider:
            mins_in_stage[u] = c*epoch_len/60
    total_minutes = np.nansum([counts for stage, counts in mins_in_stage.items()])
    percent_in_stage = {k: 100 * v / total_minutes for k, v in mins_in_stage.items()}
    return mins_in_stage, percent_in_stage, total_minutes


def sleep_efficiency(mins_in_stage, total_minutes, wake_stages=(0,)):
    """
    Standard measure of sleep efficiency: (total_minutes-wake minutes)/total minutes
    :param mins_in_stage: minutes in each stage, as a dict or list (if list, wake stage must be index)
    :param total_minutes: total minutes in stages to consider
    :param wake_stages: wake stages as "list of dict keys" or "list of list indexes" into mins in stage
    :return: sleep efficiency
    """
    return (total_minutes - np.sum([mins_in_stage[s] for s in wake_stages]))/total_minutes


def total_sleep_time(mins_in_stage, wake_stages=(0,)):
    """
    Total sleep time, the total time sleeping in stages that are not wake
    :param mins_in_stage: mins_in_stage: minutes in each stage, as a dict or list (if list, wake stage must be index)
    :param wake_stages: wake stages as "list of dict keys" or "list of list indexes" into mins in stage
    :return: total sleep time in minutes
    """
    return np.nansum([counts for stage, counts in mins_in_stage.items() if stage not in wake_stages])


def sleep_latency(epoch_stages, wake_stage=0, epoch_len=30):
    """
    Calculate the time of wake prior to sleep, i.e. sleep latency
    :param epoch_stages: the pattern of sleep stages with self transitions, e.g. [0 0 1 1 1 2 2 2 1]
    :param wake_stage: stage that represents wake (as opposed to WASO)
    :param epoch_len: length of an epoch in seconds (30s by default)
    :return: sleep latency in minutes
    """
    wake_only = np.where(np.array(epoch_stages) == wake_stage, 1, 0)
    first_trans_from_wake = list(np.diff(wake_only)).index(-1)+1
    return first_trans_from_wake*epoch_len/60


