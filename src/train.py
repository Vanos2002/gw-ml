"""Train and evaluate scikit-learn models on the simulated GW dataset.

Task 1 - detection:  is there a binary black hole signal in this segment?
Task 2 - estimation: for segments with a signal, what is the chirp mass?

Outputs figures to figures/ and numbers to results/metrics.json.

Usage:
    python src/train.py --data data/dataset.csv
"""
# pyright: reportMissingModuleSource=false
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (mean_absolute_error, r2_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

META = ["id", "label", "m1", "m2", "chirp_mass", "snr", "t_merger"]
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})


def detection(df, feats, out, seed):
    X, y, snr = df[feats].values, df["label"].values, df["snr"].values
    X_tr, X_te, y_tr, y_te, snr_tr, snr_te = train_test_split(
        X, y, snr, test_size=0.25, stratify=y, random_state=seed)

    models = {
        "Logistic regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000)),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, random_state=seed),
    }
    res = {}
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, m in models.items():
        cv_auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc")
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_te)[:, 1]
        fpr, tpr, thr = roc_curve(y_te, p)
        res[name] = {"cv_auc_mean": cv_auc.mean(), "cv_auc_std": cv_auc.std(),
                     "test_auc": roc_auc_score(y_te, p)}
        ax[0].plot(fpr, tpr, label=f"{name} (AUC={res[name]['test_auc']:.3f})")
        # efficiency vs SNR at a fixed 1% false-alarm probability
        t1 = np.quantile(p[y_te == 0], 0.99)
        bins = np.arange(4, 21, 2)
        sig = y_te == 1
        idx = np.digitize(snr_te[sig], bins)
        eff = [np.mean(p[sig][idx == b] > t1) for b in range(1, len(bins))]
        ax[1].plot(0.5 * (bins[1:] + bins[:-1]), eff, "o-", label=name)
        res[name]["efficiency_at_1pct_FAP"] = dict(
            zip([f"{a}-{b}" for a, b in zip(bins[:-1], bins[1:])], map(float, eff)))
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
              title="Detection ROC (test set)")
    ax[0].legend(loc="lower right")
    ax[1].set(xlabel="Optimal SNR of injected signal",
              ylabel="Detection efficiency", ylim=(0, 1.02),
              title="Efficiency at 1% false-alarm probability")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{out}/detection.png")

    # which features does the gradient boosting model rely on?
    gb = models["Gradient boosting"]
    imp = permutation_importance(gb, X_te, y_te, scoring="roc_auc",
                                 n_repeats=10, random_state=seed, n_jobs=-1)
    plot_importance(imp, feats, f"{out}/detection_importance.png",
                    "Permutation importance - detection (drop in AUC)")
    return res


def chirp_mass_regression(df, feats, out, seed):
    s = df[df.label == 1]
    X, y, snr = s[feats].values, np.log(s["chirp_mass"].values), s["snr"].values
    X_tr, X_te, y_tr, y_te, snr_tr, snr_te = train_test_split(
        X, y, snr, test_size=0.25, random_state=seed)
    m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05,
                                      random_state=seed)
    m.fit(X_tr, y_tr)
    pred = np.exp(m.predict(X_te))
    true = np.exp(y_te)
    rel = np.abs(pred - true) / true
    baseline = np.full_like(true, np.exp(np.median(y_tr)))   # always guess the median
    res = {"test_mae_Msun": mean_absolute_error(true, pred),
           "test_r2": r2_score(true, pred),
           "baseline_mae_Msun": mean_absolute_error(true, baseline),
           "median_rel_error": float(np.median(rel)),
           "median_rel_error_snr_gt_12": float(np.median(rel[snr_te > 12]))}

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sc = ax[0].scatter(true, pred, c=snr_te, s=8, cmap="viridis")
    lim = [true.min() * 0.9, true.max() * 1.1]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set(xlabel=r"True chirp mass [$M_\odot$]",
              ylabel=r"Predicted chirp mass [$M_\odot$]",
              title=f"Chirp-mass regression (R$^2$={res['test_r2']:.2f})")
    fig.colorbar(sc, ax=ax[0], label="SNR")
    bins = np.arange(4, 21, 2)
    idx = np.digitize(snr_te, bins)
    med = [np.median(rel[idx == b]) for b in range(1, len(bins))]
    ax[1].plot(0.5 * (bins[1:] + bins[:-1]), np.array(med) * 100, "o-")
    ax[1].set(xlabel="Optimal SNR", ylabel="Median relative error [%]",
              title="Accuracy improves with signal strength")
    fig.tight_layout()
    fig.savefig(f"{out}/chirp_mass.png")

    imp = permutation_importance(m, X_te, y_te, n_repeats=10,
                                 random_state=seed, n_jobs=-1)
    plot_importance(imp, feats, f"{out}/chirp_mass_importance.png",
                    "Permutation importance - chirp mass (drop in R$^2$)")
    return res


def plot_importance(imp, feats, path, title, top=12):
    order = np.argsort(imp.importances_mean)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(np.array(feats)[order], imp.importances_mean[order],
            xerr=imp.importances_std[order], color="#4a7ab5")
    ax.set(title=title, xlabel="Importance")
    fig.tight_layout()
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs("figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(args.data)
    feats = [c for c in df.columns if c not in META]
    print(f"{len(df)} samples, {len(feats)} features")

    metrics = {"detection": detection(df, feats, "figures", args.seed),
               "chirp_mass": chirp_mass_regression(df, feats, "figures", args.seed)}
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
