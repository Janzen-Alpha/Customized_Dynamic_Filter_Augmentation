

> **Language:** English · [中文](README_CN.md)

# CDFA — Customized Dynamic Filter Augmentation

## What CDFA is

CDFA is a data-augmentation method for time-series forecasting. It rewrites a  
training window by redistributing the window's own first differences under a  
data-dependent weight profile and one global magnitude. The closed form is

```
d' = d + 1/2 * alpha * Delta(phi .* d)
```

where `d` denotes the first differences of the window, `phi` is a weight  
profile built from local volatility and rescaled to unit mean, and `alpha` is  
the single global magnitude (`0.3` in all experiments). Because the  
redistribution acts on increments rather than levels, the level of the window  
and its alignment with the value that follows it are both preserved. Every  
window is rewritten once, so the size of the training library stays fixed.

## What it is for

- Perturbing training windows in a controlled way when labelled history is  
  limited; one scalar sets the strength of the perturbation and no new  
  observations are needed.
- A learner-agnostic preprocessing step. The experiments use a  
  distance-weighted kNN and a small single-layer LSTM written in NumPy, on  
  both one-step rolling forecasts and direct multi-step forecasts.
- A controlled comparison against published augmentation families on the  
  equity panel, including window warping, magnitude warping and  
  frequency-domain masking and mixing.

The evaluation covers eight univariate Darts series, a fixed pool of 20 S\&P  
500 names over 2012–2025 (280 stock-year cells), and public long-horizon  
forecasting panels. Results are reported as MAE together with paired  
improvement, a bootstrap confidence interval, win rate and a Wilcoxon test.

## Repository layout

```
.
|-- README.md
|-- Datasets/                          external datasets (read-only inputs)
|   |-- Darts/                          eight univariate series
|   |-- Stocks/                         S&P 500 adjusted-close panel
|   `-- ETTs/                           public forecasting panels
`-- paper 'Magnitude... CDFA' codes/    reproduction package: code + results
    |-- README.md
    |-- main.py, cdfa.py, ...           source code
    |-- augmentations.py, compare_augmentations.py
    `-- outputs/                        tables, per-cell results, run logs
```

Other folders in the repository are not needed to reproduce the results.

## The reproduction package

`paper 'Magnitude... CDFA' codes/` holds the source code, the numerical  
results and the run logs that accompany the submitted manuscript  
**"Magnitude-Controlled Increment Redistribution for Time-Series by Adapting  
Advanced Filtering Techniques"**. Every table in that paper can be  
regenerated from these scripts; the CSV files under `outputs/` are the  
per-cell results and summary statistics used for the reported numbers. The  
README inside that folder lists the files, the commands and the evaluation  
protocol.

## Data

`Datasets/` is the only data source. `datasets.py` resolves these paths  
relative to the repository root, so the project can be moved as long as this  
layout is kept. The data files are inputs only; nothing is written into  
`Datasets/`.

## Environment

Python 3.13 or later with NumPy, pandas, SciPy and scikit-learn. openpyxl is  
used when available to write the xlsx version of the Darts table. No  
pretrained weights, API keys or external services are required.
