import numpy as np
import scipy as sp


def signal_envelope(sig):
    return sp.signal.hilbert(sig)


def mean_amplitude(sig):
    return np.nanmean(sig)


def rms_power(sig):
    return np.sqrt(np.mean(sig**2))


def max_amplitude(sig):
    return np.nanmax(sig)


def peak_freq(sig_spectrogram, freq_bins):
    return freq_bins[np.argmax(np.nanmean(sig_spectrogram, axis=0))]




#Rise time, etc



