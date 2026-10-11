"""Data loading for the three dataset families.

    darts   the eight Darts datasets          -> Datasets/Darts/
    stocks  the S&P 500 adjusted-close panel  -> Datasets/Stocks/
    panels  the public forecasting panels     -> Datasets/ETTs/

Paths are resolved from the location of this file, so the project folder can
be moved anywhere as long as its internal layout is kept.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

# codes/ sits next to Datasets/, outputs/ and README.md
BASE = Path(__file__).resolve().parent.parent
DART_DIR = BASE / 'Datasets' / 'Darts'
STOCK_DIR = BASE / 'Datasets' / 'Stocks'
ETT_DIR = BASE / 'Datasets' / 'ETTs'

SPLIT = 0.8   # Darts train/test split


# ==========================================================================
# Darts
# ==========================================================================
DART_NAMES = ["AirPassengersDataset", "AusBeerDataset", "GasRateCO2Dataset",
              "MonthlyMilkDataset", "SunspotsDataset", "WineDataset",
              "WoolyDataset", "HeartRateDataset"]


def load_darts(name):
    # Read one Darts series as a 1-D array.  GasRateCO2 keeps its second
    # column; Sunspots is thinned 4x and HeartRate 2x.
    df = pd.read_csv(os.path.join(DART_DIR, name + ".csv"))
    col = 2 if name == "GasRateCO2Dataset" else 1
    s = df.iloc[:, col].astype(float).reset_index(drop=True)
    if name == "SunspotsDataset":
        s = s.iloc[::4].reset_index(drop=True)
    if name == "HeartRateDataset":
        s = s.iloc[::2].reset_index(drop=True)
    return s.to_numpy()


# ==========================================================================
# Stocks
# ==========================================================================
SP500_FILE = 'SP500_adjClose_2010-2025.csv'

# The fixed 20-name pool of the equity panel.
STOCK_POOL = ['AAPL', 'MSFT', 'NVDA', 'ORCL', 'GOOGL', 'DIS', 'AMZN', 'HD',
              'MCD', 'WMT', 'PG', 'PEP', 'KO', 'JPM', 'V', 'MA', 'JNJ',
              'UNH', 'XOM', 'CAT']

TEST_START, TEST_END = 2012, 2025   # evaluation years of the equity panel


def load_stocks():
    # Wide adjusted-close frame, indexed by date, restricted to the pool.
    df = pd.read_csv(os.path.join(STOCK_DIR, SP500_FILE))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    return df[[t for t in STOCK_POOL if t in df.columns]]


def stock_units(dates):
    # One calendar year per unit; a year with fewer than 50 rows is dropped.
    units = []
    for yr in range(TEST_START, TEST_END + 1):
        idx = np.where(dates.year == yr)[0]
        if len(idx) >= 50:
            units.append((int(yr), int(idx[0]), int(idx[-1])))
    return units


# ==========================================================================
# Public panels
# ==========================================================================
# kind='panel' : a CSV with a date column and a header
# kind='raw'   : a headerless value matrix, a synthetic index is imposed
PANELS = {
    'ETTh2': dict(kind='panel', file='ETTh2.csv', target='OT', sample=None,
                  unit='M', min_rows=200, stride=3),
    'ETTm2': dict(kind='panel', file='ETTm2.csv', target='OT', sample=None,
                  unit='M', min_rows=200, stride=12),
    'weather': dict(kind='panel', file='weather.csv', target='OT', sample=None,
                    unit='M', min_rows=200, stride=18),
    'Air-Quality': dict(kind='raw', file='Air-Quality.csv', epoch='2004-03-10',
                        freq='1h', target=None, sample=60,
                        unit='M', min_rows=200, stride=3),
    'Fred-MD': dict(kind='raw', file='Fred-MD.csv', epoch='1959-01-01',
                    freq='MS', target=None, sample=60,
                    unit='Y', min_rows=12, stride=1),
    # --- further panels; uncomment once the file is added to Datasets/ETTs/ ---
    # 'electricity': dict(kind='panel', file='electricity.csv', target='OT',
    #                     sample=60, unit='M', min_rows=200, stride=3),
    # 'solar': dict(kind='raw', file='solar_AL.txt', epoch='2006-01-01 00:00',
    #               freq='10min', target=None, sample=60,
    #               unit='M', min_rows=200, stride=18),
    # 'M4': dict(kind='m4', file='M4_Monthly-train.csv', target=None,
    #            sample=200, min_len=330, unit='Y', min_rows=12, stride=1),
}

# Order the panels are reported in.  The full order is
# Fred-MD, weather, ETTm2, Air-Quality, M4, ETTh2, solar, electricity.
PANEL_ORDER = ['Fred-MD', 'weather', 'ETTm2', 'Air-Quality', 'ETTh2']


def _clean_table(frame):
    # Date column in, float body out; the Jena sentinel becomes NaN and is
    # interpolated; duplicated timestamps are dropped.
    body = frame.iloc[:, 1:].astype(float)
    body = body.mask(body < -8000)
    body = body.interpolate(axis=0, limit_direction='both').bfill().ffill()
    t = pd.to_datetime(frame.iloc[:, 0])
    keep = ~t.duplicated(keep='first')
    out = body.loc[keep].copy()
    out.index = pd.DatetimeIndex(t.loc[keep], name='date')
    return out


def _raw_matrix(path, epoch, freq):
    # Headerless matrix: rows are time, columns are series.
    arr = pd.read_csv(path, header=None).values.astype(float)
    idx = pd.date_range(epoch, periods=len(arr), freq=freq)
    cols = ['S%04d' % (j + 1) for j in range(arr.shape[1])]
    return pd.DataFrame(arr, index=pd.DatetimeIndex(idx, name='date'),
                        columns=cols)


def load_panel(name):
    cfg = PANELS[name]
    path = os.path.join(ETT_DIR, cfg['file'])
    if cfg['kind'] == 'panel':
        return _clean_table(pd.read_csv(path))
    return _raw_matrix(path, cfg['epoch'], cfg['freq'])


def sample_channels(frame, n, target=None):
    # Order the channels by mean level, take evenly spaced quantiles, always
    # keep the benchmark target column.  Keeps the whole magnitude range.
    cols = list(frame.columns)
    if n is None or n >= len(cols):
        return cols
    means = frame[cols].mean().sort_values()
    idx = np.linspace(0, len(means) - 1, n).round().astype(int)
    picked = list(dict.fromkeys(means.index[i] for i in idx))
    if target is not None and target in cols and target not in picked:
        picked[-1] = target
    return [c for c in cols if c in set(picked)]


def panel_channels(name, frame):
    cfg = PANELS[name]
    return sample_channels(frame, cfg['sample'], cfg['target'])


def panel_units(name, index):
    # Evaluation units: a calendar month, or a year for the low-frequency
    # panels.  A unit with fewer than min_rows rows is dropped.
    cfg = PANELS[name]
    period, min_rows = cfg['unit'], cfg['min_rows']
    keys = index.year if period == 'Y' else index.year * 100 + index.month
    out = []
    for k in sorted(set(int(v) for v in keys)):
        pos = np.where(keys == k)[0]
        if len(pos) < min_rows:
            continue
        label = ('%d' % k) if period == 'Y' else ('%d-%02d' % (k // 100, k % 100))
        out.append((label, int(pos[0]), int(pos[-1])))
    return out
