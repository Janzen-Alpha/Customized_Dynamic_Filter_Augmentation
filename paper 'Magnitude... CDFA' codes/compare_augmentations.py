"""Compare CDFA with published augmentation methods on the Stocks dataset.

Run from this folder:

    python compare_augmentations.py --workers 12

Outputs (three new files in outputs/; existing files are never touched):

    cells_augmentations.csv   one row per (ticker, year, method)
    table_augmentations.csv   paired statistics behind Table III
    log_augmentations.txt     console log of the run

Protocol: one cell is one calendar year of one stock, min-max scaled on its
own prefix; the library holds the last 300 windows; a distance-weighted kNN
(k=5) forecasts from 15-point windows; predictions are clipped to
[0.67, 1.5] times the previous value; the origin advances by 3 observations.
Every method rewrites each library window once, so the library size is the
same for all methods.

Statistics are paired on the 280 cells: improvement is the mean relative
change against no augmentation, the interval is a 5000-draw bootstrap, the
win rate is the fraction of improved cells, and p is a Wilcoxon signed-rank
test. "CDFA win" is the fraction of cells where CDFA beats that method.
"""

# Single-threaded BLAS inside each worker; set before numpy is imported.
import os
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'BLIS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import augmentations as aug
import cdfa
import datasets
from models import knn_next


# ---------------------------------------------------------------------------
# fixed experiment settings
# ---------------------------------------------------------------------------
T = 15                  # learner input length
TRAIN_WINDOW = 300      # windows in one library
STRIDE = 3              # how far the forecast origin moves
CLIP_LO, CLIP_HI = 0.67, 1.5
SEED = 42
MIN_PRED = 10
YEARS = range(2012, 2026)
BOOT = 5000

# Nine settings: six form Table III, three check the strength of a transform.
METHODS = ['none', 'cdfa',
           'ww30', 'ww15',            # window warping, two slice fractions
           'mw20', 'mw10', 'mw04',    # magnitude warping, three sigmas
           'fmask', 'fmix']           # FrAug, one rFFT band

WW_FRAC = {'ww30': 0.3, 'ww15': 0.15}
MAG_SIGMA = {'mw20': 0.2, 'mw10': 0.01, 'mw04': 0.004}
FREQ_RATE = 0.1         # one of the nine rFFT bins of a 16-point segment

TABLE_ORDER = ['none', 'cdfa', 'ww15', 'mw04', 'fmask', 'fmix']
EXTRA_ORDER = ['ww30', 'mw10', 'mw20']

LABEL = {
    'none':  'None (no augmentation)',
    'cdfa':  'CDFA (Proposed)',
    'ww15':  'Window warping [5]',
    'mw04':  'Magnitude warping [6]',
    'fmask': 'FrAug masking [7]',
    'fmix':  'FrAug mixing [7]',
    'ww30':  'Window warping, f=0.30',
    'mw10':  'Magnitude warping, sigma=0.01',
    'mw20':  'Magnitude warping, sigma=0.20 (published)',
}


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def apply_method(method, Xtr, ytr, rng, phi):
    """Rewrite every window of one library; return the new library."""
    if method == 'none':
        return Xtr
    if method == 'cdfa':
        return cdfa.augment(Xtr, 'cdfa', phi, rng)
    if method in WW_FRAC:
        return np.array([aug.window_warp(x, rng, frac=WW_FRAC[method])
                         for x in Xtr])
    if method in MAG_SIGMA:
        return np.array([aug.magnitude_warp(x, rng, sigma=MAG_SIGMA[method])
                         for x in Xtr])
    if method == 'fmask':
        return np.array([aug.freq_mask(x, y, rng, mu=FREQ_RATE)[0]
                         for x, y in zip(Xtr, ytr)])
    if method == 'fmix':
        partner = rng.permutation(len(Xtr))
        return np.array([aug.freq_mix(x, y, Xtr[j], ytr[j], rng,
                                      mu=FREQ_RATE)[0]
                         for (x, y), j in zip(zip(Xtr, ytr), partner)])
    raise ValueError(method)


def evaluate_unit(values, ys, ye, method, rng):
    """MAE of one method on one (ticker, year) cell, plus its dose."""
    prefix = values[:ye + 1]
    lo, hi = float(prefix.min()), float(prefix.max())
    ns = (prefix - lo) / (hi - lo) if hi - lo > 1e-12 else prefix * 0.0
    sw = sliding_window_view(ns, T)
    phi = cdfa.cdfa_weights(ns, T, ys - 1) if method == 'cdfa' else None

    preds, actuals, dose = [], [], []
    prev = None
    for idx in range(ys, ye + 1, STRIDE):
        if idx - TRAIN_WINDOW < T:
            continue
        Xtr = np.ascontiguousarray(sw[idx - TRAIN_WINDOW: idx - T])
        if len(Xtr) < 10:
            continue
        ytr = ns[idx - TRAIN_WINDOW + T: idx]
        q = np.ascontiguousarray(sw[idx - T])

        Xa = apply_method(method, Xtr, ytr, rng, phi)
        if not dose:
            sd = float(np.mean(np.std(Xtr, axis=1)))
            dose.append(float(np.mean(np.abs(Xa - Xtr))) / sd
                        if sd > 1e-12 else 0.0)

        p = knn_next(Xa, ytr, q)
        if prev is not None and abs(prev) > 1e-12:
            p = min(max(p, prev * CLIP_LO), prev * CLIP_HI)
        prev = p
        preds.append(p * (hi - lo) + lo)
        actuals.append(float(ns[idx]) * (hi - lo) + lo)

    if len(preds) < MIN_PRED:
        return None
    mae = float(np.mean(np.abs(np.array(preds) - np.array(actuals))))
    return mae, dose[0], len(preds)


def run_ticker(job):
    """All cells of one ticker; the RNG restarts for each method."""
    ticker, values, years = job
    rows = []
    for method in METHODS:
        rng = np.random.RandomState(SEED)
        for yr in YEARS:
            idx = np.where(years == yr)[0]
            if len(idx) < 50:
                continue
            ys, ye = int(idx[0]), int(idx[-1])
            if ys < TRAIN_WINDOW + T:
                continue
            out = evaluate_unit(values, ys, ye, method, rng)
            if out is None:
                continue
            mae, dose, n = out
            rows.append(dict(Ticker=ticker, Year=yr, Method=method,
                             MAE=mae, Dose=dose, N=n))
    return rows


# ---------------------------------------------------------------------------
# paired statistics
# ---------------------------------------------------------------------------
def rel(base, other):
    """Relative change in percent; positive means a lower error."""
    return (base - other) / base * 100.0


def boot_ci(x, rng):
    """Bootstrap 95% interval of the mean over cells."""
    idx = rng.integers(0, len(x), (BOOT, len(x)))
    return np.percentile(x[idx].mean(axis=1), [2.5, 97.5])


def summarise(cells):
    """Pair every method with None and with CDFA on the same cells."""
    wide = cells.pivot_table(index=['Ticker', 'Year'], columns='Method',
                             values='MAE')
    dose = cells.groupby('Method')['Dose'].mean()
    rng = np.random.default_rng(SEED)
    out = {}
    # Keep the pivot column order: the bootstrap RNG is shared by all methods.
    for m in wide.columns:
        d = rel(wide['none'].to_numpy(), wide[m].to_numpy())
        lo, hi = boot_ci(d, rng)
        p = stats.wilcoxon(wide[m], wide['none']).pvalue \
            if m != 'none' else np.nan
        out[m] = dict(mae=wide[m].mean(), imp=d.mean(), lo=lo, hi=hi,
                      wr=float((d > 0).mean() * 100), p=p, dose=dose[m],
                      n=len(wide))
        if m != 'cdfa':
            dc = rel(wide[m].to_numpy(), wide['cdfa'].to_numpy())
            out[m]['vs_cdfa'] = dc.mean()
            out[m]['cdfa_win'] = float((dc > 0).mean() * 100)
            out[m]['p_cdfa'] = stats.wilcoxon(wide['cdfa'], wide[m]).pvalue
    return out


def fmt_p(p):
    """Short p-value for the log."""
    if p != p:
        return '--'
    return '%.1e' % p if p < 1e-3 else '%.3f' % p


def print_table(s, order, title):
    """Print one block of the results in a plain text table."""
    print('\n' + title)
    print('-' * 92)
    print('%-42s %8s %-20s %8s %11s %9s' %
          ('Method', 'MAE', 'Imp. vs None (%)', 'WR (%)', 'CDFA win (%)', 'p'))
    print('-' * 92)
    for m in order:
        r = s[m]
        imp = '--' if m == 'none' else \
            '%+.2f [%+.2f, %+.2f]' % (r['imp'], r['lo'], r['hi'])
        wr = '--' if m == 'none' else '%.1f' % r['wr']
        win = '--' if m in ('none', 'cdfa') else '%.1f' % r['cdfa_win']
        print('%-42s %8.4f %-20s %8s %11s %9s' %
              (LABEL[m], r['mae'], imp, wr, win, fmt_p(r['p'])))
    print('-' * 92)


def build_table(s):
    """Turn the statistics into a tidy CSV table."""
    rows = []
    for m in TABLE_ORDER + EXTRA_ORDER:
        r = s[m]
        is_none = m == 'none'
        no_head2head = m in ('none', 'cdfa')
        rows.append(dict(
            Method=LABEL[m],
            Setting=m,
            In_Table_III=m in TABLE_ORDER,
            MAE=round(r['mae'], 6),
            Imp_vs_None_pct=None if is_none else round(r['imp'], 6),
            CI_lo=None if is_none else round(r['lo'], 6),
            CI_hi=None if is_none else round(r['hi'], 6),
            WR_pct=None if is_none else round(r['wr'], 6),
            CDFA_Win_pct=None if no_head2head else round(r['cdfa_win'], 6),
            CDFA_minus_method_pp=None if no_head2head
            else round(r['vs_cdfa'], 6),
            p_vs_None=None if is_none else r['p'],
            p_vs_CDFA=None if no_head2head else r['p_cdfa'],
            Dose=round(r['dose'], 6),
            N_cells=r['n'],
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
class Tee:
    """Copy everything printed to the console into a log file."""
    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=12,
                    help='parallel workers (default 12)')
    ap.add_argument('--out', default='outputs', help='output folder')
    ap.add_argument('--from-cells', action='store_true',
                    help='rebuild the table and log from an existing '
                         'cells_augmentations.csv (skip the evaluation)')
    a = ap.parse_args()

    out = Path(a.out) if Path(a.out).is_absolute() else HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    log = Tee(out / 'log_augmentations.txt')
    sys.stdout = log
    t0 = time.perf_counter()
    print('run = augmentations   workers = %d   time = %s' %
          (a.workers, time.strftime('%Y-%m-%d %H:%M:%S')))

    cells_path = out / 'cells_augmentations.csv'
    try:
        if a.from_cells:
            # Fast path: reuse the per-cell results and only redo the stats.
            cells = pd.read_csv(cells_path)
            print('%d stocks x %d years x %d settings  (%d rows, reused %s)' %
                  (cells.Ticker.nunique(), cells.Year.nunique(),
                   cells.Method.nunique(), len(cells), cells_path))
        else:
            # Read the wide panel once in the parent, not inside every worker.
            df = datasets.load_stocks()
            jobs = []
            for tk in df.columns:
                s = df[tk].dropna()
                jobs.append((tk, s.to_numpy(float), s.index.year.to_numpy()))
            print('%d stocks x %d years x %d settings' %
                  (len(jobs), len(YEARS), len(METHODS)))

            with mp.Pool(a.workers) as pool:
                chunks = pool.map(run_ticker, jobs)
            cells = pd.DataFrame([r for chunk in chunks for r in chunk])
            cells.to_csv(cells_path, index=False, encoding='utf-8-sig')
            print('saved: %s  (%d rows)' % (cells_path, len(cells)))

        s = summarise(cells)
        print_table(s, TABLE_ORDER,
                    'Table III  Comparison with existing augmentations '
                    '(Stocks dataset, %d cells)' % s['cdfa']['n'])
        print_table(s, EXTRA_ORDER, 'Strength checks (not in Table III)')
        print('dose (mean |change| / window sd): '
              'CDFA %.2f, window warping %.2f, magnitude warping %.2f, '
              'FrAug masking %.2f, FrAug mixing %.2f' %
              (s['cdfa']['dose'], s['ww15']['dose'], s['mw04']['dose'],
               s['fmask']['dose'], s['fmix']['dose']))

        table = build_table(s)
        table_path = out / 'table_augmentations.csv'
        table.to_csv(table_path, index=False, encoding='utf-8-sig')
        print('saved: %s' % table_path)
    finally:
        verb = 'rebuilt' if a.from_cells else 'done'
        print('\n%s in %.0f s' % (verb, time.perf_counter() - t0))
        sys.stdout = log.stdout
        log.file.close()


if __name__ == '__main__':
    main()
