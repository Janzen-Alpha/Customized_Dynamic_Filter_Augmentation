"""Turn the per-cell MAE table into the three summary tables.

One cell is one (ticker, year) or one (dataset, channel, unit).  Every number
is a paired comparison on those cells:

    improvement = mean over cells of (baseline - method) / baseline * 100
    95% CI      = bootstrap over the cells (5000 draws, seed 42)
    win         = fraction of cells with a positive improvement
    p           = paired Wilcoxon signed-rank
"""
import numpy as np
import pandas as pd
from scipy import stats

import datasets
from evaluate import METHOD_LABEL, METHODS, REPORT

MIN_BASE = 1e-6     # cells whose baseline MAE is below this are dropped


def paired(x, y):
    # x = method, y = baseline.  Positive means the method is better.
    ok = x.notna() & y.notna()
    x, y = x[ok], y[ok]
    keep = y > MIN_BASE
    x, y = x[keep], y[keep]
    rel = (y - x) / y * 100
    try:
        p = stats.wilcoxon(x, y).pvalue
    except Exception:
        p = np.nan
    rng = np.random.RandomState(42)
    bs = [np.mean(rel.values[rng.randint(0, len(rel), len(rel))])
          for _ in range(5000)]
    return dict(n=len(rel), rel=rel.mean(),
                ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5),
                win=(rel > 0).mean() * 100, p=p)


def _pivot(cells, index, methods):
    w = cells.pivot_table(index=index, columns='Method', values='MAE')
    return w, {m: w[m] for m in methods if m in w.columns}


# ==========================================================================
# Table 2 -- equity panel
# ==========================================================================
def table2(cells):
    w, cols = _pivot(cells, ['Ticker', 'Unit'], METHODS)
    rows = []
    for m in REPORT:
        if m not in cols:
            continue
        s = paired(cols[m], cols['baseline'])
        rows.append(dict(Method=METHOD_LABEL[m], MAE=cols[m].mean(),
                         Improvement=s['rel'], CI_lo=s['ci_lo'],
                         CI_hi=s['ci_hi'], Win=s['win'], p=s['p'], n=s['n']))
    return pd.DataFrame(rows)


# ==========================================================================
# Table 3 -- public panels
# ==========================================================================
def table3(cells):
    rows = []
    for ds in datasets.PANEL_ORDER:
        s = cells[cells.Dataset == ds]
        if not len(s):
            continue
        w, cols = _pivot(s, ['Channel', 'Unit'], METHODS)
        if 'baseline' not in cols:
            continue
        a = paired(cols['cdfa'], cols['baseline'])
        row = dict(Dataset=ds, CDFA=a['rel'], CI_lo=a['ci_lo'],
                   CI_hi=a['ci_hi'], Win=a['win'])
        for m, key in [('preset_b', 'PresetB'), ('ccf', 'CCF'),
                       ('jitter', 'Jitter'), ('scaled_noise', 'SNoise')]:
            row[key] = paired(cols[m], cols['baseline'])['rel'] if m in cols \
                else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ==========================================================================
# Table 4 -- head to head
# ==========================================================================
def _head2head(cells, index):
    # Advantage of CDFA over another method, in percentage points of that
    # method's MAE.  Cells where the comparator is essentially zero cannot
    # give a ratio and are dropped (same rule as the baseline comparisons).
    w, cols = _pivot(cells, index, METHODS)
    out = {}
    for comp in ('ccf', 'preset_b'):
        if comp in cols:
            keep = cols[comp] > MIN_BASE
            rel = ((cols[comp][keep] - cols['cdfa'][keep]) / cols[comp][keep]
                   * 100)
            out[comp] = (rel.mean(), (rel > 0).mean() * 100)
        else:
            out[comp] = (np.nan, np.nan)
    return out


def table4(panel_cells, stock_cells):
    rows = []
    for ds in datasets.PANEL_ORDER:
        s = panel_cells[panel_cells.Dataset == ds]
        if not len(s):
            continue
        h = _head2head(s, ['Channel', 'Unit'])
        rows.append(dict(Dataset=ds, vs_CCF=h['ccf'][0], CCF_win=h['ccf'][1],
                         vs_PresetB=h['preset_b'][0],
                         PresetB_win=h['preset_b'][1]))
    h = _head2head(stock_cells, ['Ticker', 'Unit'])
    rows.append(dict(Dataset='Equity panel', vs_CCF=h['ccf'][0],
                     CCF_win=h['ccf'][1], vs_PresetB=h['preset_b'][0],
                     PresetB_win=h['preset_b'][1]))
    return pd.DataFrame(rows)


# ==========================================================================
# printing / saving
# ==========================================================================
def print_table2(df):
    print('\nTable 2  Equity panel: 20 S&P 500 stocks, 2012-2025, '
          '%d (stock, year) cells' % int(df.n.iloc[0]))
    print('-' * 78)
    print('%-20s %9s %9s %-18s %9s %10s' %
          ('Method', 'MAE', 'Improv.', '95% CI', 'Win rate', 'p'))
    print('-' * 78)
    for _, r in df.iterrows():
        print('%-20s %9.4f %+9.2f  [%+.2f, %+.2f] %8.1f%% %10.1e'
              % (r.Method, r.MAE, r.Improvement, r.CI_lo, r.CI_hi, r.Win, r.p))
    print('-' * 78)


def print_table3(df):
    print('\nTable 3  Public panels: relative MAE change vs no augmentation (%)')
    print('-' * 82)
    print('%-14s %22s %9s %10s %10s' %
          ('Dataset', 'CDFA (this work)', 'win', 'Preset B', 'CCF'))
    print('-' * 82)
    for _, r in df.iterrows():
        print('%-14s %+6.2f [%+.2f, %+.2f] %8.1f%% %+10.2f %+10.2f'
              % (r.Dataset, r.CDFA, r.CI_lo, r.CI_hi, r.Win, r.PresetB, r.CCF))
    print('-' * 82)


def print_table4(df):
    print('\nTable 4  Paired head-to-head advantage of CDFA '
          '(percentage points)')
    print('-' * 62)
    print('%-16s %12s %9s %12s %9s' %
          ('Dataset', 'vs CCF', 'win', 'vs Preset B', 'win'))
    print('-' * 62)
    for _, r in df.iterrows():
        print('%-16s %+12.2f %8.1f%% %+12.2f %8.1f%%'
              % (r.Dataset, r.vs_CCF, r.CCF_win, r.vs_PresetB, r.PresetB_win))
    print('-' * 62)


def save(df, path):
    df.to_csv(path, index=False, encoding='utf-8-sig')
    print('saved:', path)
