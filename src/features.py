"""Whitening and feature extraction for detector-strain segments.

The features are designed to be physically interpretable:
  * per-band excess power and the time at which it peaks
    (a chirp sweeps upward in frequency, so low bands peak earlier)
  * properties of the loudest time-frequency pixel
  * a fitted "chirp track": how fast the dominant frequency rises,
    which depends on the chirp mass  (df/dt ~ Mc^(5/3) f^(11/3))
  * simple statistics of the whitened time series
"""
import numpy as np
from scipy.signal import spectrogram  # pyright: ignore[reportMissingImports]
from scipy.signal.windows import tukey  # pyright: ignore[reportMissingImports]
from scipy.stats import kurtosis  # pyright: ignore[reportMissingImports]

BANDS = [(20, 40), (40, 60), (60, 100), (100, 150),
         (150, 250), (250, 400), (400, 1000)]


def whiten(x, psd, fs, f_lower):
    """Divide the data by the noise amplitude spectrum, so noise becomes white."""
    n = len(x)
    X = np.fft.rfft(x * tukey(n, 0.1))
    f = np.fft.rfftfreq(n, 1.0 / fs)
    asd = np.sqrt(np.where(psd > 0, psd, np.inf))
    X = X / asd
    X[(f < f_lower) | (f > 1000)] = 0.0
    w = np.fft.irfft(X, n)
    # normalise using the central part (edges are tapered)
    core = w[n // 8: -n // 8]
    return w / np.std(core)


def extract_features(w, fs):
    feats = {}
    n = len(w)
    edge = n // 16                   # ignore the tapered edges
    core = w[edge:-edge]

    feats["max_abs_strain"] = np.max(np.abs(core))
    feats["kurtosis"] = kurtosis(core)

    f, t, S = spectrogram(w, fs=fs, nperseg=256, noverlap=224, window="hann")
    keep = (t > edge / fs) & (t < (n - edge) / fs)
    t, S = t[keep], S[:, keep]
    S = S / np.median(S, axis=1, keepdims=True)   # normalise each frequency row

    for lo, hi in BANDS:
        sel = (f >= lo) & (f < hi)
        p = S[sel].mean(axis=0)
        name = f"{lo}_{hi}Hz"
        feats[f"bandpow_max_{name}"] = p.max()
        feats[f"bandpow_t_{name}"] = t[np.argmax(p)]

    sel = (f >= 20) & (f <= 1000)
    fS, S2 = f[sel], S[sel]
    i, j = np.unravel_index(np.argmax(S2), S2.shape)
    feats["peak_pixel_power"] = S2[i, j]
    feats["peak_pixel_freq"] = fS[i]
    feats["peak_pixel_time"] = t[j]

    # loudest frequency in each time column -> "chirp track"
    col_max = S2.max(axis=0)
    col_f = fS[np.argmax(S2, axis=0)]
    feats["n_loud_columns"] = int(np.sum(col_max > 15.0))
    feats["excess_power"] = float(np.sum(np.clip(S2 - 1.0, 0, None)) / S2.size)

    # estimated merger time = column with most total excess power (smoothed)
    col_excess = np.convolve(np.clip(S2 - 1.0, 0, None).sum(axis=0),
                             np.ones(5) / 5, mode="same")
    k = int(np.argmax(col_excess))
    feats["t_peak_excess"] = t[k]
    feats["peak_col_excess"] = col_excess[k] / np.median(col_excess)
    # fit log(f) vs time over the second before the peak, weighted by loudness
    win = (t >= t[k] - 1.0) & (t <= t[k] + 0.02)
    wts = np.clip(col_max[win] - 12.0, 0, None)
    if np.count_nonzero(wts) >= 3:
        tt, ff = t[win] - t[k], np.log(col_f[win])
        slope, intercept = np.polyfit(tt, ff, 1, w=np.sqrt(wts))
        feats["track_logf_slope"] = slope
        feats["track_f_at_peak"] = np.exp(intercept)
        feats["track_loudness"] = float(np.mean(wts))
    else:
        feats["track_logf_slope"] = 0.0
        feats["track_f_at_peak"] = fS[i]
        feats["track_loudness"] = 0.0
    # time each band peaks BEFORE the estimated merger: a heavier binary sweeps
    # through the band faster (tau ~ Mc^(-5/3) f^(-8/3)), so this encodes Mc
    for lo, hi in BANDS[:4]:
        feats[f"dt_to_merger_{lo}_{hi}Hz"] = t[k] - feats[f"bandpow_t_{lo}_{hi}Hz"]
    # power-weighted mean frequency of loud pixels near the peak
    near = (t >= t[k] - 0.5) & (t <= t[k] + 0.05)
    P = np.clip(S2[:, near] - 12.0, 0, None)
    feats["freq_centroid"] = float((fS[:, None] * P).sum() / (P.sum() + 1e-9))
    return feats
