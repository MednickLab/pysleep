import os
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

#%% Sleep feature defaults:
default_freq_bands = {
    'SWA_band_hz': (0.5, 1),
    'delta_band_hz': (1, 4),
    'theta_band_hz': (4, 8),
    'alpha_band_hz': (8, 12),
    'sigma_band_hz': (11, 16),
    'slowsigma_band_hz': (11, 13),
    'fastsigma_band_hz': (13, 16),
    'beta_band_hz': (16, 20)
}
band_power_epoch_len = 30

# %% transition probability
include_self_transitions = False

# %% Turn or of off matlab detector functionality
load_matlab_detectors = os.name != 'nt'


# Quartile ->
