"""Plot one simulated segment: whitened strain and its spectrogram (figures/example.png)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from features import whiten
from simulate import F_LOWER, FS, N, PSD, make_signal, noise_from_psd

rng = np.random.default_rng(7)
noise = noise_from_psd(N, 1.0 / FS, PSD, seed=11).numpy()
signal = make_signal(30.0, 25.0, snr=15.0, t_merger=3.0, rng=rng)
w = whiten(noise + signal, PSD.numpy(), FS, F_LOWER)
# whitening is (almost exactly) linear, so the signal part is the difference
w_sig = w - whiten(noise, PSD.numpy(), FS, F_LOWER)

t = np.arange(N) / FS
nperseg, step = 256, 32
frames = np.lib.stride_tricks.sliding_window_view(w, nperseg)[::step]
frames = frames - frames.mean(axis=1, keepdims=True)
# Periodic Tukey window, matching spectrogram's default window.
x = np.arange(nperseg) / nperseg
window = np.ones(nperseg)
edge = x < 0.125
window[edge] = 0.5 * (1 + np.cos(np.pi * (x[edge] / 0.125 - 1)))
edge = x >= 0.875
window[edge] = 0.5 * (1 + np.cos(np.pi * (x[edge] / 0.125 - 7)))
f = np.fft.rfftfreq(nperseg, d=1.0 / FS)
ts = (np.arange(len(frames)) * step + nperseg / 2) / FS
S = np.abs(np.fft.rfft(frames * window, axis=1)).T ** 2
S /= FS * np.sum(window**2)
S /= np.median(S, axis=1, keepdims=True)

fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
ax[0].plot(t, w, lw=0.3, color="0.6", label="whitened data (noise + signal)")
ax[0].plot(t, w_sig, lw=0.8, color="C3", label="injected signal (whitened)")
ax[0].set(ylabel="whitened strain", title=r"BBH $30+25\,M_\odot$, SNR 15, in simulated aLIGO noise")
ax[0].legend(loc="upper left", fontsize=8)
m = ax[1].pcolormesh(ts, f, S, shading="auto", vmax=25, cmap="magma")
ax[1].set(ylim=(20, 500), xlabel="time [s]", ylabel="frequency [Hz]")
fig.colorbar(m, ax=ax, label="normalised power", shrink=0.5)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/example.png", dpi=120, bbox_inches="tight")
print("saved figures/example.png")
