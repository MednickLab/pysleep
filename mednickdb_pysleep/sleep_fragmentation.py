import numpy as np
import itertools


def num_awakenings(epoch_stages, wake_stage=0):
    wake_only = np.where(np.array(epoch_stages) == wake_stage, 1, 0)
    return np.sum(np.diff(wake_only) == 1)


def transition_counts(epoch_stages, count_self_trans=False):#, first_order=False, second_order=False):
    stages = np.unique(epoch_stages)
    index_map = {stage:idx for idx, stage in enumerate(stages)}
    num_stages = len(stages)
    first = np.zeros((num_stages, num_stages))
    second = np.zeros((num_stages, num_stages, num_stages))

    for a, b, c in zip(epoch_stages[:-2], epoch_stages[1:-1], epoch_stages[2:]):
        first[a, b] = first[a, b]+1
        second[a,b,c] = second[a,b,c]+1

    first[epoch_stages[-2], epoch_stages[-1]] = first[epoch_stages[-2], epoch_stages[-1]] + 1
    if not count_self_trans:
        for stage in stages:
            first[stage, stage] = 0
    zeroth = np.sum(first, axis=0)
    return zeroth.astype(int), first.astype(int), second.astype(int)


def duration_distributions(epoch_stages, epoch_len=30):
    dur_dists = {stage:[] for stage in np.unique(epoch_stages)}
    for stage, run in itertools.groupby(epoch_stages):
        dur_dists[stage].append(sum(1 for _ in run)*epoch_len)
    return dur_dists

