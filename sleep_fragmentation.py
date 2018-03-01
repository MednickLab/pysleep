import numpy as np


def num_awakenings(epoch_stages, wake_stage=0):
    wake_only = np.where(np.array(epoch_stages) == wake_stage, 1, 0)
    return np.sum(np.diff(wake_only) == 1)


def transition_probabilities(epoch_stages, zeroth_order=True):#, first_order=False, second_order=False):
    zeroth_order_dict = {}
    for stage in np.unique(epoch_stages):
        stage_only = np.where(np.array(epoch_stages) == stage, 1, 0)
        zeroth_order_dict['trans_p_from_any_to_' + str(stage)] = np.sum(np.diff(stage_only) == 1)
    all_trans = np.sum([c for s, c in zeroth_order_dict.items()])
    zeroth_order_dict = {k: v / all_trans for k, v in zeroth_order_dict.items()}

    if zeroth_order:
        return zeroth_order_dict
    # if first_order:
    #     return [zeroth_order_dict] + [first_order]
    # if second_order:
    #     return [zeroth_order_dict] + [first_order] + [second_order]
