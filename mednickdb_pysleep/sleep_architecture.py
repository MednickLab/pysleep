import numpy as np
from mednickdb_pysleep import pysleep_defaults, pysleep_utils
import pandas as pd

def sleep_stage_architecture(epoch_stage,
                             epoch_len=pysleep_defaults.epoch_len,
                             stages_to_consider=pysleep_defaults.stages_to_consider,
                             epochs_to_ignore=None,
                             per_quartile=False,
                             return_type='dict'):
    """
    Calculate the traditional measures of sleep (mins in stage, percent in stage, total minutes of sleep (total sleep time))
    :param epoch_stage: the pattern of sleep stages with self transitions, e.g. [0 0 1 1 1 2 2 2 1]
    :param epoch_len: length of an epoch in seconds
    :param stages_to_consider: which stages we should use for calcualtions, will default to sleep stages and wake (0,1,2,3,4) all other stages will be ignored and will not count towards metrics
    :return: a tuple of (minutes in stage, percent in stage, total minutes for stages to consider)
    """
    missing_stages = [stage for stage in stages_to_consider if stage not in epoch_stage]

    df = pd.DataFrame({'stage':epoch_stage}).reset_index().rename({'index':'stage_idx'}, axis=1)
    df['onset'] = df['stage_idx']*epoch_len
    df['duration'] = epoch_len

    df = df.loc[df['stage'].isin(stages_to_consider),:]

    def to_dict(df, colname):
        return {l: df.xs(l)[colname].to_dict() for l in df.index.levels[0]}

    if per_quartile:
        df, _ = pysleep_utils.assign_quartiles(df, epoch_stage)

    if epochs_to_ignore is not None:
        df = df.loc[~df['stage_idx'].isin(epochs_to_ignore),:]

    if per_quartile:
        df_out = df.groupby(['quartile', 'stage']).agg({'duration':np.nansum})
        df_out['minutes_in_stage'] = df_out['duration'] / 60
        df_total_mins = df_out.groupby(['quartile']).agg({'minutes_in_stage':np.nansum})
        df_total_mins.columns = ['total_minutes']
        df_out = df_out.join(df_total_mins)
        for q, quart_data in df_out.groupby('quartile'):
            for stage in stages_to_consider:
                if stage not in quart_data.index.get_level_values(1).tolist():
                    df_out.ix[(q, stage), 'total_minutes'] = df_total_mins.loc[q,'total_minutes']
                    df_out.ix[(q, stage), 'minutes_in_stage'] = 0

    else:
        df_out = df.groupby(['stage']).agg({'duration': np.nansum})
        df_out['minutes_in_stage'] = df_out['duration'] / 60
        df_out['total_minutes'] = df_out['minutes_in_stage'].agg(np.nansum)
        for stage in stages_to_consider:
            if stage not in df_out.index.tolist():
                df_out.loc[stage, 'total_minutes'] = df_out['total_minutes'][0]
                df_out.loc[stage, 'minutes_in_stage'] = 0


    df_out = df_out.drop('duration', axis=1)
    df_out['percent'] = 100*df_out['minutes_in_stage']/df_out['total_minutes']

    if return_type == 'dataframe':
        return df_out
    elif return_type == 'dict':
        if per_quartile:
            return to_dict(df_out,'minutes_in_stage'), \
                   to_dict(df_out, 'percent'), \
                   df_out.reset_index().drop('stage', axis=1).set_index('quartile')['total_minutes'].to_dict()
        else:
            return df_out['minutes_in_stage'].to_dict(), \
                   df_out['percent'].to_dict(), \
                   df_out['total_minutes'].reset_index(drop=True).drop_duplicates().values.tolist()[0]
    else:
        raise ValueError('return type is unknown')


def sleep_efficiency(mins_in_stage, total_minutes, wake_stages=pysleep_defaults.wake_stages_to_consider):
    """
    Standard measure of sleep efficiency: (total_minutes-wake minutes)/total minutes
    :param mins_in_stage: minutes in each stage, as a dict or list (if list, wake stage must be index)
    :param total_minutes: total minutes in stages to consider
    :param wake_stages: wake stages as "list of dict keys" or "list of list indexes" into mins in stage
    :return: sleep efficiency
    """
    return (total_minutes - np.sum([mins_in_stage[s] for s in wake_stages]))/total_minutes if total_minutes else np.NaN


def total_sleep_time(mins_in_stage, wake_stages=pysleep_defaults.wake_stages_to_consider):
    """
    Total sleep time, the total time sleeping in stages that are not wake
    :param mins_in_stage: mins_in_stage: minutes in each stage, as a dict or list (if list, wake stage must be index)
    :param wake_stages: wake stages as "list of dict keys" or "list of list indexes" into mins in stage
    :return: total sleep time in minutes
    """
    return np.nansum([counts for stage, counts in mins_in_stage.items() if stage not in wake_stages])


def lights_on_off_and_sleep_latency(epoch_stages,
                                    lights_off=None, #FIXME is this ever used? remove arg...
                                    lights_on=None,
                                    epoch_sync_offset_seconds=0,
                                    wbso_stage=pysleep_defaults.wbso_stage,
                                    wase_stage=pysleep_defaults.wase_stage,
                                    stages_to_consider=pysleep_defaults.stages_to_consider,
                                    epoch_len=30):
    """
        Calculates:
     - lights off (either passed through from input, or assumed the start of the first epoch of wbso or sleep/waso)
     - lights on (either passed though from input, or assumed the end of the last sleep or wase epoch)
     - sleep latency (difference between lights off and the first epoch of sleep, may be None if no sleep occurred)
     Note: wbso and wase must be coded different (and given as arguments if not default)
    :param lights_off: when lights where turned off in seconds since (edf record start+epoch_sync_offset)
    :param lights_on: when lights where turned on in seconds since (edf record start+epoch_sync_offset)
    :param epoch_sync_offset: offset (in seconds) between the start of the first scored epoch and the start of the edf record
    :param epoch_stages: the pattern of sleep stages with self transitions, e.g. [0 0 1 1 1 2 2 2 1]
    :param wbso_stage: stage that represents Wake Before Sleep Onset (as opposed to WASO)
    :param wase_stage: stage that represents Wake After Sleep End (as opposed to WASO)
    :param stages_to_consider: stages that we consider sleep, including waso
    :param epoch_len: length of an epoch in seconds (30s by default)
    :return: lights off in seconds, lights on in seconds, sleep latency in minutes, epoch_stages with only epochs between lights on and lights off
    """
    wbso_epochs = np.where(np.array(epoch_stages) == wbso_stage)[0]
    wase_epochs = np.where(np.array(epoch_stages) == wase_stage)[0]
    sleep_epochs = np.where([e in stages_to_consider for e in epoch_stages])[0]
    assert len(sleep_epochs) or len(wbso_epochs) or len(wase_epochs), "no known stages, this seems like an error"
    if len(sleep_epochs):  # no sleep, therefore no latency
        sleep_start = sleep_epochs[0]
    else:
        sleep_start = None

    if lights_off is None:
        if len(wbso_epochs):
            lights_off = wbso_epochs[0]
        else:
            lights_off = sleep_start

    if lights_on is None:
        if len(wase_epochs):
            lights_on = wase_epochs[-1]+1
        else:
            lights_on = 1 + sleep_epochs[-1] if (sleep_start is not None) else 1 + wbso_epochs[-1]

    sleep_latency = sleep_start - lights_off if (sleep_start is not None) else None

    epoch_stages_sliced = epoch_stages[lights_off:lights_on]

    lights_off_seconds = None if lights_off is None else lights_off * epoch_len + epoch_sync_offset_seconds
    lights_on_seconds = None if lights_on is None else lights_on * epoch_len + epoch_sync_offset_seconds
    sleep_latency_mins = None if sleep_latency is None else sleep_latency * epoch_len / 60

    return lights_off_seconds, lights_on_seconds, sleep_latency_mins, epoch_stages_sliced


