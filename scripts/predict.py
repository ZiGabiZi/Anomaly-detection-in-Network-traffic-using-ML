"""Score network flows with the random forest saved by scripts/train.py.

The input CSV needs the 42 UNSW-NB15 feature columns (other columns are ignored). The output has
one row per flow: id (if the input has one), score, alert and, with --category, the most likely
attack category for flows with an alert. Score and alert come from the binary model; the category
comes from a separate multi-class model and doesn't change the alert. It is indicative only: that
model has a macro-F1 of about 0.50 on test, and Analysis, Backdoor and DoS mostly come out as
Exploits.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from netad.models import MODELS_DIR
from netad.models.forest import load_forest, score_flows, unknown_categories

SECURITY = ("Models are loaded with joblib, which runs code stored in the file (pickle). "
            "Only load models you trained yourself with scripts/train.py.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, epilog=SECURITY,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path, help="CSV with one flow per row")
    parser.add_argument("--output", required=True, type=Path, help="CSV to write")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help=f"where train.py saved the models (default: {MODELS_DIR})")
    parser.add_argument("--category", action="store_true",
                        help="also predict the attack category (needs train.py --multiclass)")
    args = parser.parse_args(argv)

    try:
        # utf-8-sig: the UNSW-NB15 CSVs start with a BOM
        flows = pd.read_csv(args.input, encoding="utf-8-sig")
        model, meta = load_forest("binary", args.models_dir)
        category_model = load_forest("multiclass", args.models_dir)[0] if args.category else None
        out = score_flows(flows, model, meta["threshold"], category_model)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    for col, n in unknown_categories(model, flows).items():
        print(f"warning: {n} flows with {col} values not seen in training", file=sys.stderr)

    out.to_csv(args.output, index=False)
    n_alerts = int(out["alert"].to_numpy().sum())
    print(f"{len(out)} flows, {n_alerts} alerts ({n_alerts / max(len(out), 1):.1%}) "
          f"at threshold {meta['threshold']:.4f} -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
