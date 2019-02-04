import numpy as np
import itertools
from typing import Dict, Union, List


def num_awakenings(epoch_stages, waso_stage=0):
    """
    Count the number of transitions to wake
    :param epoch_stages: The pattern of sleep stages, ignoring duration (e.g. [ 0 1 2 1]
    :param waso_stage: stage that represents wake, default 0
    :return: number of awakenings
    """
    wake_only = np.where(np.array(epoch_stages) == waso_stage, 1, 0)
    return np.sum(np.diff(wake_only) == 1)


def transition_counts(epoch_stages: list, count_self_trans: bool=False, normalize: bool=False, stages_to_consider=(0,1,2,3,4)): #first_order=False, second_order=False):
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

    num_stages = len(stages_to_consider)
    first = np.zeros((num_stages, num_stages))
    second = np.zeros((num_stages, num_stages, num_stages))

    if len(epoch_stages) <= 3:
        return None, None, None

    for a, b, c in zip(epoch_stages[:-2], epoch_stages[1:-1], epoch_stages[2:]):
        first[a, b] += 1
        second[a, b, c] += 1
    first[b, c] += 1  # make sure to add the last one too :)

    first[epoch_stages[-2], epoch_stages[-1]] = first[epoch_stages[-2], epoch_stages[-1]] + 1
    if not count_self_trans:
        for stage in stages_to_consider:
            first[stage, stage] = 0
    zeroth = np.sum(first, axis=0)
    if not normalize:
        return zeroth.astype(int), first.astype(int), second.astype(int)
    else:
        return zeroth.astype(int)/np.sum(zeroth), \
               first.astype(int)/np.expand_dims(np.sum(first, axis=1), 1), \
               second.astype(int)/np.expand_dims(np.sum(second, axis=2), 2)


def bout_durations(epoch_stages: list, epoch_len: int=30, stages_to_consider=(0, 1, 2, 3, 4)) -> Dict[Union[str, int], List[float]]:
    """
    Convert an epoch stages array (which includes self transitions) to a set of durations
    :param epoch_stages: epoch_stages: The pattern of sleep stages, with self-transitions e.g. [0 0 1 1 1 2 2 2 1]
    :param epoch_len: the length in seconds of each epoch
    :param stages_to_consider: which stages to calculate bout durations for
    :return: a dict, with one key per stage, and a list of durations for each bout of a stage
    """
    dur_dists = {s: [] for s in np.unique(epoch_stages)}

    for stage, run in itertools.groupby(epoch_stages):
        if stage in stages_to_consider:
            dur_dists[stage].append(float(len([_ for _ in run]))*epoch_len/60)
    return dur_dists

