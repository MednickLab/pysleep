# Default parameters for automatic parsing

# Sleep defaults:
epoch_len = 30

# %% Sleep stages used in mednickdb
sleep_stages = ['stage1', 'stage2', 'sws', 'rem']
wake_stages = ['wbso', 'waso', 'wase']
wbso_stage = 'wbso'  # wake before sleep onset
waso_stage = 'waso'  # wake after sleep onset
wase_stage = 'wase'  # wake after sleep end
nrem_stages = ['stage1', 'stage2', 'sws']
non_sleep_stages = ['unknown', 'lights', 'movement', 'artifact']
all_stages = non_sleep_stages + sleep_stages
unknown_stage = 'unknown'
wake_stages_to_consider = ['waso']
stages_to_consider = wake_stages_to_consider + sleep_stages


# %% transition probability
include_self_transitions = False
