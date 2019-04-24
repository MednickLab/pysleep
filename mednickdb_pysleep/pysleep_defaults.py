# Default parameters for automatic parsing

# General Sleep defaults:
epoch_len = 30

# %% Sleep stages used in mednickdb
stage1_stage = 'n1'
stage2_stage = 'n2'
sws_stage = 'n3'
rem_stage = 'rem'
sleep_stages = [stage1_stage, stage2_stage, sws_stage, rem_stage]
wbso_stage = 'wbso'  # wake before sleep onset
waso_stage = 'waso'  # wake after sleep onset
wase_stage = 'wase'  # wake after sleep end
wake_stages = [wbso_stage, waso_stage, wase_stage]
nrem_stages = [stage1_stage, stage2_stage, sws_stage]
unknown_stage = 'unknown'
non_sleep_or_wake_stages = [unknown_stage, 'lights', 'movement', 'artifact']
all_stages = non_sleep_or_wake_stages + sleep_stages
wake_stages_to_consider = [waso_stage]
stages_to_consider = wake_stages_to_consider + sleep_stages


# %% transition probability
include_self_transitions = False

# %% Turn or of off matlab detector functionality
load_matlab_detectors = False
