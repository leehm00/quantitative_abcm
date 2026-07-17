from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from abcm.evaluation import evaluate_factor_frame, write_evaluation_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate exported ABCM factor CSV files.")
    parser.add_argument("--factors-csv", required=True, help="CSV containing TRADE_DT, S_INFO_WINDCODE, factor columns, y1_raw.")
    parser.add_argument("--output-dir", default=None, help="Directory for metrics CSV files. Defaults to the factor file directory.")
    parser.add_argument("--rolling-window", type=int, default=243)
    parser.add_argument("--autocorr-lag", type=int, default=5)
    args = parser.parse_args()

    df = pd.read_csv(args.factors_csv, dtype={"TRADE_DT": str, "S_INFO_WINDCODE": str})
    result = evaluate_factor_frame(df, rolling_window=args.rolling_window, autocorr_lag=args.autocorr_lag)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.factors_csv).resolve().parent
    write_evaluation_outputs(result, output_dir)
    print(result.beta_metrics.to_string(index=False))
    print(f"wrote_metrics_dir={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
