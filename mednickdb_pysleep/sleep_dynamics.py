import numpy as np
import itertools
from typing import Dict, Union


def num_awakenings(epoch_stages, wake_stage=0):
    """
    Count the number of transitions to wake
    :param epoch_stages: The pattern of sleep stages, ignoring duration (e.g. [ 0 1 2 1]
    :param wake_stage: stage that represents wake, default 0
    :return: number of awakenings
    """
    wake_only = np.where(np.array(epoch_stages) == wake_stage, 1, 0)
    return np.sum(np.diff(wake_only) == 1)


def transition_counts(epoch_stages: list, count_self_trans: bool=False, normalize: bool=False): #first_order=False, second_order=False):
    """
    Get the number of transition from one stage to another
    :param epoch_stages: The pattern of sleep stages, will handle with and without duration e.g. [0 1 2 1]
    or [0 0 1 1 1 2 2 2 1] will produce the same ans (if count_self_trans=False)
    :param count_self_trans: If transitions from and too the same stages should be counted (i.e. [0 0]). If false, diagnals will be 0
    :param normalize: Whether to normalize to transition probabilities or leave as counts
    :return: a tuple of (zeroth order transitions, first order transitions, second order transitions):
        last dimension is stage transitioned to, other dimensions are the last stages.
        e.g. for first order, dims = [current stage, next stage]
        e.g. for 2nd order, dims=[previous stage, current stage, next stage]
    """
    stages = np.unique(epoch_stages)
    index_map = {stage:idx for idx, stage in enumerate(stages)}
    num_stages = len(stages)
    first = np.zeros((num_stages, num_stages))
    second = np.zeros((num_stages, num_stages, num_stages))

    for a, b, c in zip(epoch_stages[:-2], epoch_stages[1:-1], epoch_stages[2:]):
        first[a, b] = first[a, b]+1
        second[a, b, c] = second[a, b, c]+1

    first[epoch_stages[-2], epoch_stages[-1]] = first[epoch_stages[-2], epoch_stages[-1]] + 1
    if not count_self_trans:
        for stage in stages:
            first[stage, stage] = 0
    zeroth = np.sum(first, axis=0)
    if not normalize:
        return zeroth.astype(int), first.astype(int), second.astype(int)
    else:
        return zeroth.astype(int)/np.sum(zeroth), \
               first.astype(int)/np.expand_dims(np.sum(first, axis=1), 1), \
               second.astype(int)/np.expand_dims(np.sum(second, axis=2), 2)


def duration_distributions(epoch_stages: list, epoch_len: int=30) -> Dict[Union[int, str], list]:
    """
    Convert an epoch stages array (which includes self transitions) to a set of durations
    :param epoch_stages: epoch_stages: The pattern of sleep stages, with self-transitions e.g. [0 0 1 1 1 2 2 2 1]
    :param epoch_len: the length in seconds of each epoch
    :return: a dict, with one key per stage, and a list of durations for each bout of a stage
    """
    dur_dists = {stage:[] for stage in np.unique(epoch_stages)}
    for stage, run in itertools.groupby(epoch_stages):
        dur_dists[stage].append(sum(1 for _ in run)*epoch_len)
    return dur_dists

