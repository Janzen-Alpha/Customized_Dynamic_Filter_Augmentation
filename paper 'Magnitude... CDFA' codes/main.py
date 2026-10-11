"""Entry point for the three dataset families.

    python main.py darts     ~ 8 Darts datasets   -> knn_lstm_cdfa_mae_table.csv
    python main.py stocks    ~ 20 S&P 500 stocks  -> Table 2
    python main.py panels    ~ public panels      -> Table 3 and Table 4
    python main.py all       ~ all of the above

Optional: --workers N (default 1) for the panel/stock runs, --out DIR.
The printed tables are also written to a log file in outputs/.
"""
import os

# Single-threaded BLAS inside each worker; must be set before numpy is imported.
for _v in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
           'BLIS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import argparse
import sys
import time
from pathlib import Path

CODES = Path(__file__).resolve().parent      # this folder, for the flat imports
ROOT = CODES.parent                          # project folder: Datasets/, outputs/
sys.path.insert(0, str(CODES))


class Tee:
    # Copy everything printed to the console into a log file as well.
    def __init__(self, path):
        self.file = open(path, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.file.write(s)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def run_darts(out):
    import darts_pipeline
    darts_pipeline.run(out)


def run_stocks(out, workers):
    import evaluate
    import tables
    print('running the equity panel ...')
    cells = evaluate.run_stocks(workers)
    cells.to_csv(os.path.join(out, 'cells_stocks.csv'), index=False,
                 encoding='utf-8-sig')
    t2 = tables.table2(cells)
    tables.print_table2(t2)
    tables.save(t2, os.path.join(out, 'table2_equity_panel.csv'))
    return cells


def run_panels(out, workers, stock_cells=None):
    import evaluate
    import tables
    print('running the public panels ...')
    cells = evaluate.run_panels(workers=workers)
    cells.to_csv(os.path.join(out, 'cells_panels.csv'), index=False,
                 encoding='utf-8-sig')
    t3 = tables.table3(cells)
    tables.print_table3(t3)
    tables.save(t3, os.path.join(out, 'table3_public_panels.csv'))

    # Table 4 also has an equity-panel row, so the stock cells are needed.
    if stock_cells is None:
        stock_cells = run_stocks(out, workers)
    t4 = tables.table4(cells, stock_cells)
    tables.print_table4(t4)
    tables.save(t4, os.path.join(out, 'table4_head_to_head.csv'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('family', choices=['darts', 'stocks', 'panels', 'all'])
    ap.add_argument('--workers', type=int, default=1,
                    help='parallel workers for the stock/panel runs')
    ap.add_argument('--out', default='outputs', help='output folder')
    a = ap.parse_args()

    out = os.path.join(ROOT, a.out)
    os.makedirs(out, exist_ok=True)
    log = Tee(os.path.join(out, 'log_%s.txt' % a.family))
    sys.stdout = log
    t0 = time.time()
    print('family = %s   workers = %d   time = %s'
          % (a.family, a.workers, time.strftime('%Y-%m-%d %H:%M:%S')))

    try:
        if a.family == 'darts':
            run_darts(out)
        elif a.family == 'stocks':
            run_stocks(out, a.workers)
        elif a.family == 'panels':
            run_panels(out, a.workers)
        else:
            run_darts(out)
            stock = run_stocks(out, a.workers)
            run_panels(out, a.workers, stock_cells=stock)
    finally:
        print('\ndone in %.0f s' % (time.time() - t0))
        sys.stdout = log.stdout
        log.file.close()


if __name__ == '__main__':
    main()
