"""Train the random forest detector on the UNSW-NB15 training file and save it to models/.

Uses the hyperparameters saved in reports/results.json by notebooks/02_supervised.ipynb, with the
same split and seed, so it rebuilds the model reported there. The alert threshold is set for 1% FPR
on the validation part.

Only the random forest is trained here: it is the model selected on validation. The other models
(gradient boosting, Isolation Forest, autoencoder) are compared in the notebooks; see the README.
"""

import argparse
import sys
from pathlib import Path

from netad.data import load_unsw_frame
from netad.models import MODELS_DIR
from netad.models.forest import save_forest, train_forest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--multiclass", action="store_true",
                        help="also train the attack category model (a few minutes more)")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help=f"where to save the models (default: {MODELS_DIR})")
    args = parser.parse_args(argv)

    train_df = load_unsw_frame("train")
    for task in ["binary", "multiclass"] if args.multiclass else ["binary"]:
        model, info = train_forest(train_df, task)
        path = save_forest(model, info, args.models_dir)
        if task == "binary":
            print(f"binary: threshold {info['threshold']:.4f}, validation recall {info['val_recall']:.3f} "
                  f"at FPR {info['val_fpr']:.3f} -> {path}")
        else:
            print(f"multiclass: validation macro-F1 {info['val_macro_f1']:.3f} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
