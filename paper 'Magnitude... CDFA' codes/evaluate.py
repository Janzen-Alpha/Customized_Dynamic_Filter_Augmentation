"""The rolling protocol: one fixed evaluation used by stocks and panels.

Protocol (identical for both families, only the data differs):

    training window W = 300 observations
    learner input length t = 15
    a window is augmented, then a distance-weighted kNN (k=5) predicts the
    next observation; predictions are clipped to [0.67, 1.5] x the previous one
    the window rolls forward by `stride` observations
    one evaluation unit = one calendar year (stocks) or one calendar month
    (panels); the unit's MAE is one cell of the paired comparison

Weights are computed once at the start of each unit and held fixed.  Nothing
is tuned per dataset.
"""
import numpy as np
import pandas as pd

import cdfa
from cdfa import W as PHI_W          # number of weight positions (15)
import datasets

TRAIN_WINDOW = 300
T = 15
CLIP_LO, CLIP_HI = 0.67, 1.5
SEED = 42
MIN_PRED = 10
STOCK_STRIDE = 3

METHODS = ['baseline', 'cdfa', 'preset_b', 'ccf', 'jitter', 'scaled_noise']
# Every table compares these against the baseline.
REPORT = ['cdfa', 'preset_b', 'ccf', 'jitter', 'scaled_noise']
METHOD_LABEL = {
    'baseline': 'Baseline (no augmentation)',
    'cdfa': 'cdfa (this work)',
    'preset_b': 'Preset curve B',
    'ccf': 'CCF (antecedent)',
    'jitter': 'Jitter',
    'scaled_noise': 'Scale-aware noise',
}


def weights_for(method, ns, k):
    # The weight vector for this unit; None for the noise baselines.
    if method == 'cdfa':
        return cdfa.cdfa_weights(ns, PHI_W, k)
    if method == 'preset_b':
        return cdfa.preset_b_weights(PHI_W)
    if method == 'ccf':
        return cdfa.ccf_weights(ns, PHI_W, k)
    return None


def evaluate_unit(values, ys, ye, method, stride, rng):
    # MAE of one method over one unit, or None when the unit is too short.
    from numpy.lib.stride_tricks import sliding_window_view

    prefix = values[:ye + 1]
    lo, hi = float(prefix.min()), float(prefix.max())
    ns = (prefix - lo) / (hi - lo) if hi - lo > 1e-12 else prefix * 0.0
    sw = sliding_window_view(ns, T)          # window ending at i is sw[i - T]
    phi = weights_for(method, ns, ys - 1)

    preds, actuals = [], []
    prev = None
    for idx in range(ys, ye + 1, stride):
        if idx - TRAIN_WINDOW < T:
            continue
        Xtr = np.ascontiguousarray(sw[idx - TRAIN_WINDOW: idx - T])
        if len(Xtr) < 10:
            continue
        ytr = ns[idx - TRAIN_WINDOW + T: idx]
        Xte = np.ascontiguousarray(sw[idx - T])[None, :]
        Xa = cdfa.augment(Xtr, method, phi, rng)
        p = cdfa_knn(Xa, ytr, Xte[0])
        if prev is not None and abs(prev) > 1e-12:
            if p / prev > CLIP_HI:
                p = prev * CLIP_HI
            elif p / prev < CLIP_LO:
                p = prev * CLIP_LO
        prev = p
        preds.append(p * (hi - lo) + lo)
        actuals.append(float(ns[idx]) * (hi - lo) + lo)

    if len(preds) < MIN_PRED:
        return None
    return float(np.mean(np.abs(np.array(preds) - np.array(actuals))))


def cdfa_knn(Xa, ytr, q):
    from models import knn_next
    return knn_next(Xa, ytr, q)


def evaluate_series(values, units, methods, stride, extra=None):
    # All (unit, method) cells of one series.  The RNG is reseeded per method
    # and then runs across the units.
    rows = []
    for method in methods:
        rng = np.random.RandomState(SEED)
        for label, ys, ye in units:
            if ys < TRAIN_WINDOW + T:
                continue
            mae = evaluate_unit(values, ys, ye, method, stride, rng)
            if mae is None:
                continue
            row = dict(extra or {})
            row.update(Unit=label, Method=method, MAE=mae)
            rows.append(row)
    return rows


# ==========================================================================
# parallel drivers
# ==========================================================================
def _stock_job(args):
    ticker, values, units = args
    return evaluate_series(values, units, METHODS, STOCK_STRIDE,
                           extra={'Ticker': ticker})


def _panel_job(args):
    name, channel, values, units, stride = args
    return evaluate_series(values, units, METHODS, stride,
                           extra={'Dataset': name, 'Channel': channel})


def _run_jobs(job, args_list, workers):
    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            out = pool.map(job, args_list, chunksize=1)
    else:
        out = [job(a) for a in args_list]
    return [r for sub in out for r in sub]


def run_stocks(workers=1):
    # One cell per (ticker, year, method).
    df = datasets.load_stocks()
    units_by_year = datasets.stock_units(df.index)
    jobs = [(tk, df[tk].dropna().to_numpy(float), units_by_year)
            for tk in df.columns]
    return pd.DataFrame(_run_jobs(_stock_job, jobs, workers))


def run_panels(names=None, workers=1):
    # One cell per (dataset, channel, unit, method).
    names = names or datasets.PANEL_ORDER
    jobs = []
    for name in names:
        cfg = datasets.PANELS[name]
        frame = datasets.load_panel(name)
        chans = datasets.panel_channels(name, frame)
        units = datasets.panel_units(name, frame.index)
        for ch in chans:
            s = frame[ch].dropna()
            jobs.append((name, ch, s.to_numpy(float), units, cfg['stride']))
    return pd.DataFrame(_run_jobs(_panel_job, jobs, workers))
