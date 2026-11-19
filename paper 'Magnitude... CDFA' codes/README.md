> **Language:** English · [中文](README_CN.md)

# Reproduction Package

This folder contains the source code, the numerical results and the run logs
that accompany the submitted manuscript **"Magnitude-Controlled Increment
Redistribution for Time-Series by Adapting Advanced Filtering Techniques"**.

CDFA rewrites a training window by redistributing its own first differences
under a data-dependent weight profile and one global magnitude:

```
d' = d + 1/2 * alpha * Delta(phi .* d)
```

Two entry points cover the experiments: `main.py` runs the three dataset
families, and `compare_augmentations.py` runs the comparison with published
augmentation methods on the equity panel. Every table in the paper can be
regenerated from the scripts below; the CSV and log files under `outputs/` are
the exact results used for the reported numbers.

## Layout

```
paper 'Magnitude... CDFA' codes/
|-- README.md
|-- main.py                   entry point; dispatches on the dataset family
|-- cdfa.py                   the method: weight profiles and redistribution rule
|-- models.py                 learners: one-step kNN, direct multi-step kNN, NumPy LSTM
|-- datasets.py               loading and unit slicing for the three families
|-- evaluate.py               rolling evaluation protocol and parallel drivers
|-- tables.py                 paired statistics and the three summary tables
|-- darts_pipeline.py         Darts pipeline (8:2 split, direct forecast)
|-- augmentations.py          published comparators: window/magnitude warping, FrAug
|-- compare_augmentations.py  equity-panel comparison with published augmentations
`-- outputs/                  tables, per-cell results and run logs
```

The datasets live one level up, in `Datasets/` at the repository root.
`datasets.py` resolves those paths from its own location, so the project can
be moved as long as the relative layout is kept.

## Usage

Run from the repository root. `main.py` resolves `--out` against the
repository root, so `--out` is used to keep every table in this folder's
`outputs/`. Without it, the default target is `outputs/` at the repository
root.

```bash
python "paper 'Magnitude... CDFA' codes/main.py" darts  --out "paper 'Magnitude... CDFA' codes/outputs"
python "paper 'Magnitude... CDFA' codes/main.py" stocks --out "paper 'Magnitude... CDFA' codes/outputs"
python "paper 'Magnitude... CDFA' codes/main.py" panels --out "paper 'Magnitude... CDFA' codes/outputs"
python "paper 'Magnitude... CDFA' codes/main.py" all    --out "paper 'Magnitude... CDFA' codes/outputs"

python "paper 'Magnitude... CDFA' codes/compare_augmentations.py" --workers 12
```

`main.py` accepts `darts`, `stocks`, `panels` or `all`, plus `--workers N`
(parallel evaluation, default 1) and `--out DIR`. Its console output is copied
into `outputs/log_<family>.txt`.

`compare_augmentations.py` writes to `outputs/` inside this folder by default
and accepts `--workers N` (default 12), `--out DIR` and `--from-cells`. The
full run takes about eight minutes with 12 workers; `--from-cells` skips the
evaluation and rebuilds the table and log from `cells_augmentations.csv` in a
few seconds.

## The three dataset families

**darts** — eight univariate series from the Darts collection, split 80/20.
The look-back length and the number of neighbours are chosen on the training
tail, and the forecast is direct multi-step. Four models are evaluated: kNN,
CDFA+kNN, LSTM and CDFA+LSTM, where the LSTM is a small single-layer network
written in NumPy. Errors are reported in raw units, together with their
arithmetic and geometric means scaled by the naive forecast.

**stocks** — the adjusted-close panel of the S&P 500, restricted to a fixed
pool of 20 names over 2012–2025. One evaluation unit is one calendar year, so
the panel holds 280 (stock, year) cells.

**panels** — the public long-horizon forecasting datasets. One evaluation unit
is one calendar month, or one calendar year for the low-frequency panels.
Channels are sampled systematically by mean level, always keeping the
benchmark target column.

The stock and panel families share one rolling protocol: a training window of
300 observations, a learner input length of 15, a distance-weighted kNN with 5
neighbours, predictions clipped to `[0.67, 1.5]` times the previous
observation, and a rolling step of a few observations. The weight vector is
computed once at the start of each unit and held fixed. Nothing is tuned per
dataset.

The panels currently shipped are `Fred-MD`, `weather`, `ETTm2`,
`Air-Quality` and `ETTh2`. Further panels (`electricity`, `solar`, `M4`) are
described in `datasets.py` but commented out; add the file under
`Datasets/ETTs/`, uncomment the entry, and add its name to `PANEL_ORDER`.

## Comparison with published augmentations

`compare_augmentations.py` evaluates nine augmentation settings on the same
280 equity-panel cells: no augmentation, CDFA, window warping (two slice
fractions), magnitude warping (three strengths), and FrAug frequency masking
and mixing. Every setting rewrites each library window once, so the library
size is identical across settings. `augmentations.py` holds the four published
transforms.

Six settings form the comparison table in the paper; the remaining three are
strength checks. The CSV also reports a *dose* column — the mean absolute
change of a window divided by its own standard deviation — so the strength of
each transform can be read directly. On this panel CDFA has a dose of 0.11 and
the published comparators range from 0.10 to 0.36.

## Outputs

| File | Content |
|---|---|
| `table2_equity_panel.csv` | per method: MAE, improvement with 95% CI, win rate and Wilcoxon p on the equity panel |
| `table3_public_panels.csv` | per dataset: CDFA improvement with CI and win rate, next to the two comparison filters |
| `table4_head_to_head.csv` | per dataset: CDFA advantage in percentage points over each comparison filter, with the fraction of cells won |
| `knn_lstm_cdfa_mae_table.csv/.xlsx` | the Darts table: raw-unit MAE and scaled means |
| `table_augmentations.csv` | the comparison with published augmentations: MAE, improvement with 95% CI, win rate, CDFA win rate, p-values and dose; six table rows plus three strength checks |
| `cells_stocks.csv`, `cells_panels.csv` | the per-unit MAE behind Tables 2–4 |
| `cells_augmentations.csv` | 2520 rows: per-cell MAE, dose and number of origins for all nine augmentation settings |
| `log_*.txt` | console output of each run, including `log_augmentations.txt` |

Statistics are paired at the level of one evaluation cell. The improvement of
a method is the mean over cells of `(baseline - method) / baseline`, the
interval is a 5000-draw bootstrap over the cells (seed 42), the win rate is
the fraction of cells with a positive improvement, and the p-value is a paired
Wilcoxon signed-rank test. Cells whose comparator is numerically zero cannot
give a ratio and are excluded.

## Environment

Python 3.13 or later with NumPy, pandas, SciPy and scikit-learn. The Darts
table also writes an xlsx file and uses openpyxl when available. No
pretrained weights or external services are required.
