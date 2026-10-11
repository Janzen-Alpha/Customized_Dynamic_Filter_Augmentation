# CDFA - Customized Dynamic Filter Augmentation

Source code, data and results for CDFA, a data-augmentation method for
time-series forecasting. The method rewrites a training window by
redistributing its own first differences under a data-dependent weight profile
and one global magnitude; the closed form is
`d' = d + 1/2 * alpha * Delta(phi .* d)`.

The method is evaluated on three groups of datasets through a single entry
point.

## Layout

```
.
|-- README.md
|-- codes/                    all source code
|   |-- main.py               entry point; dispatches on the dataset family
|   |-- cdfa.py               the method: weight profiles and the redistribution rule
|   |-- models.py             learners: one-step kNN, direct multi-step kNN, NumPy LSTM
|   |-- datasets.py           loading and unit slicing for the three families
|   |-- evaluate.py           the rolling evaluation protocol and its parallel drivers
|   |-- tables.py             paired statistics and the three summary tables
|   `-- darts_pipeline.py     the Darts pipeline (8:2 split, direct forecast)
|-- Datasets/
|   |-- Darts/                eight Darts series
|   |-- Stocks/               S&P 500 adjusted-close panel
|   `-- ETTs/                 public forecasting panels
`-- outputs/                  tables, per-cell results and run logs
```

## Usage

```bash
cd codes
python main.py darts              # 8 Darts datasets   -> knn_lstm_cdfa_mae_table.csv
python main.py stocks             # 20 S&P 500 stocks  -> table2_equity_panel.csv
python main.py panels             # public panels      -> table3, table4
python main.py all                # all three groups
```

Options: `--workers N` (parallel evaluation, default 1) and `--out DIR`
(output folder, default `outputs/`). Everything is written under `outputs/`,
and the console output of each run is duplicated into `outputs/log_<family>.txt`.

## The three families

**darts** - eight univariate series from the Darts collection, split 80/20. The
look-back length and the number of neighbours are chosen on the training tail,
and the forecast is direct multi-step. Four models are evaluated: kNN, CDFA+kNN,
LSTM, CDFA+LSTM, where the LSTM is a small single-layer network written in
NumPy. Errors are reported in raw units, together with their arithmetic and
geometric means scaled by the naive forecast.

**stocks** - the adjusted-close panel of the S&P 500, restricted to a fixed
pool of 20 names over 2012-2025. One evaluation unit is one calendar year, so
the panel holds 280 (stock, year) cells.

**panels** - the public long-horizon forecasting datasets. One evaluation unit
is one calendar month, or one calendar year for the low-frequency panels.
Channels are sampled systematically by mean level, always keeping the benchmark
target column.

The stock and panel families share one rolling protocol: a training window of
300 observations, a learner input length of 15, a distance-weighted kNN with 5
neighbours, predictions clipped to `[0.67, 1.5]` times the previous observation,
and a rolling step of a few observations. The weight vector is computed once at
the start of each unit and held fixed. Nothing is tuned per dataset.

The panels currently shipped are `Fred-MD`, `weather`, `ETTm2`, `Air-Quality`
and `ETTh2`. Further panels (`electricity`, `solar`, `M4`) are
already described in `codes/datasets.py` but commented out; add the file under
`Datasets/ETTs/`, uncomment the entry, and add its name to `PANEL_ORDER`.

## Outputs

| File | Content |
|---|---|
| `table2_equity_panel.csv` | per method: MAE, improvement with 95% CI, win rate and Wilcoxon p on the equity panel |
| `table3_public_panels.csv` | per dataset: CDFA improvement with CI and win rate, next to the two comparison filters |
| `table4_head_to_head.csv` | per dataset: CDFA advantage in percentage points over each comparison filter, with the fraction of cells won |
| `knn_lstm_cdfa_mae_table.csv/.xlsx` | the Darts table: raw-unit MAE and scaled means |
| `cells_*.csv` | the per-unit MAE behind every table |
| `log_*.txt` | console output of each run |

Statistics are paired at the level of one evaluation cell. The improvement of a
method is the mean over cells of `(baseline - method) / baseline`, the interval
is a bootstrap over the cells, the win rate is the fraction of cells with a
positive improvement, and the p-value is a paired Wilcoxon test. Cells whose
comparator is numerically zero cannot give a ratio and are excluded.

## Environment

Python 3.13 with NumPy, pandas, SciPy, scikit-learn and matplotlib. The Darts
table also writes an xlsx file and uses openpyxl when available.
