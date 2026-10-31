"""Simulate LIGO-like detector data with and without binary black hole signals.

Each sample is a 4 s segment of Gaussian noise coloured by the Advanced LIGO
design sensitivity curve. Half of the samples additionally contain an
inspiral-merger-ringdown waveform (IMRPhenomD) scaled to a chosen optimal SNR.
Every segment is whitened and reduced to a row of physically motivated
features (see features.py), giving a tabular dataset for scikit-learn.

Usage:
    python src/simulate.py --n 6000 --out data/dataset.csv
"""
import argparse
import importlib
import os

import numpy as np
pd = importlib.import_module("pandas")

try:
    joblib = importlib.import_module("joblib")
    Parallel, delayed = joblib.Parallel, joblib.delayed
except ImportError:  # pragma: no cover - fallback for environments without joblib
    def delayed(func):
        def wrapper(*args, **kwargs):
            return (func, args, kwargs)
        return wrapper

    class Parallel:
        def __init__(self, n_jobs=-1, verbose=0):
            self.n_jobs = n_jobs
            self.verbose = verbose

        def __call__(self, tasks):
            results = []
            for task in tasks:
                func, args, kwargs = task
                results.append(func(*args, **kwargs))
            return results

sigma = importlib.import_module("pycbc.filter").sigma
noise_from_psd = importlib.import_module("pycbc.noise").noise_from_psd
aLIGOZeroDetHighPower = importlib.import_module("pycbc.psd").aLIGOZeroDetHighPower
get_td_waveform = importlib.import_module("pycbc.waveform").get_td_waveform

from features import extract_features, whiten

FS = 2048            # sampling rate [Hz]
DURATION = 4.0       # segment length [s]
F_LOWER = 20.0       # low-frequency cutoff [Hz]
N = int(FS * DURATION)
DELTA_F = 1.0 / DURATION
PSD = aLIGOZeroDetHighPower(N // 2 + 1, DELTA_F, F_LOWER)


def chirp_mass(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def make_signal(m1, m2, snr, t_merger, rng):
    """Waveform placed in a zero array of length N, scaled to optimal SNR `snr`."""
    hp, hc = get_td_waveform(approximant="IMRPhenomD", mass1=m1, mass2=m2,
                             delta_t=1.0 / FS, f_lower=F_LOWER)
    # random sky position / polarisation -> random mix of the two polarisations
    psi = rng.uniform(0, 2 * np.pi)
    h = hp * np.cos(psi) + hc * np.sin(psi)
    # pycbc puts the merger at t=0; index of the merger inside the array:
    i_merger = int(round(-float(h.start_time) * FS))
    h = h.numpy()
    out = np.zeros(N)
    j_merger = int(t_merger * FS)
    start = j_merger - i_merger
    lo, hi = max(start, 0), min(start + len(h), N)
    out[lo:hi] = h[lo - start:hi - start]
    # rescale to requested optimal SNR (computed on what actually fits in the window)
    TimeSeries = importlib.import_module("pycbc.types").TimeSeries
    ts = TimeSeries(out, delta_t=1.0 / FS)
    s = sigma(ts, psd=PSD, low_frequency_cutoff=F_LOWER)
    return out * snr / s


def one_sample(i, has_signal, seed):
    rng = np.random.default_rng(seed)
    noise = noise_from_psd(N, 1.0 / FS, PSD, seed=int(seed % 2**31)).numpy()
    row = {"id": i, "label": int(has_signal)}
    if has_signal:
        m1, m2 = sorted(rng.uniform(5, 60, size=2), reverse=True)
        snr = rng.uniform(4, 20)
        t_merger = rng.uniform(2.5, 3.5)
        data = noise + make_signal(m1, m2, snr, t_merger, rng)
        row.update(m1=m1, m2=m2, chirp_mass=chirp_mass(m1, m2),
                   snr=snr, t_merger=t_merger)
    else:
        data = noise
        row.update(m1=np.nan, m2=np.nan, chirp_mass=np.nan,
                   snr=0.0, t_merger=np.nan)
    white = whiten(data, PSD.numpy(), FS, F_LOWER)
    row.update(extract_features(white, FS))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000, help="total number of samples")
    ap.add_argument("--out", default="data/dataset.csv")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()

    seeds = np.random.SeedSequence(args.seed).generate_state(args.n)
    labels = np.arange(args.n) % 2 == 0          # balanced classes
    rows = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(one_sample)(i, labels[i], seeds[i]) for i in range(args.n))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} samples x {df.shape[1]} columns to {args.out}")


if __name__ == "__main__":
    main()
