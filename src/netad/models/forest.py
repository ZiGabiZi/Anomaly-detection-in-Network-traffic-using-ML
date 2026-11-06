"""Random forest detectors behind scripts/train.py and scripts/predict.py."""

import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline

from netad.data import BINARY_TARGET, FEATURES, ID_COL, MULTI_TARGET, check_columns, train_val_split
from netad.evaluation import RESULTS_PATH, choose_threshold, evaluate_binary, evaluate_multiclass
from netad.models import MODELS_DIR
from netad.preprocessing import build_preprocessor

NAMES = {"binary": "rf_binary", "multiclass": "rf_multiclass"}
MAX_FPR = 0.01


def saved_params(task: str, path: Path | str = RESULTS_PATH) -> dict:
    """Random forest hyperparameters found by 02_supervised for `task`."""
    results = json.loads(Path(path).read_text(encoding="utf-8"))
    entry = next(r for r in results if (r["model"], r["task"]) == ("RandomForest", task))
    allowed = RandomForestClassifier().get_params()
    # binary entries also hold threshold settings
    return {k: v for k, v in entry["params"].items() if k in allowed}


def train_forest(train_df: pd.DataFrame, task: str, seed: int = 42,
                 results_path: Path | str = RESULTS_PATH) -> tuple[Pipeline, dict]:
    """Same split, hyperparameters and seed as 02_supervised, so the same model.

    The binary model also gets its alert threshold: highest recall at 1% FPR on validation.
    """
    if task not in NAMES:
        raise ValueError(f"task must be one of {list(NAMES)}, got {task!r}")
    target = BINARY_TARGET if task == "binary" else MULTI_TARGET
    X_tr, X_val, y_tr, y_val = train_val_split(train_df, target, seed=seed)
    params = saved_params(task, results_path)
    model = make_pipeline(
        build_preprocessor(), RandomForestClassifier(random_state=seed, n_jobs=-1, **params),
    ).fit(X_tr, y_tr)

    info = {"task": task, "params": params, "seed": seed, "train_rows": len(X_tr), "val_rows": len(X_val)}
    if task == "binary":
        scores = model.predict_proba(X_val)[:, 1]
        threshold = choose_threshold(y_val, scores, "fpr", max_fpr=MAX_FPR)
        val = evaluate_binary(y_val, scores, threshold)
        info |= {"threshold": threshold, "max_fpr": MAX_FPR, "val_recall": val["recall"], "val_fpr": val["fpr"]}
    else:
        info["val_macro_f1"] = evaluate_multiclass(y_val, model.predict(X_val))["macro_f1"]
    return model, info


def save_forest(model: Pipeline, info: dict, directory: Path | str = MODELS_DIR) -> Path:
    """Pipeline as .joblib, `info` plus creation time and library versions as .json."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = NAMES[info["task"]]
    joblib.dump(model, directory / f"{name}.joblib")
    meta = {
        **info,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (directory / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return directory / f"{name}.joblib"


def load_forest(task: str, directory: Path | str = MODELS_DIR) -> tuple[Pipeline, dict]:
    """Inverse of save_forest.

    joblib files are pickles: loading one runs code stored in it. Only load files you created
    yourself with scripts/train.py.
    """
    if task not in NAMES:
        raise ValueError(f"task must be one of {list(NAMES)}, got {task!r}")
    path = Path(directory) / f"{NAMES[task]}.joblib"
    if not path.is_file():
        flag = " --multiclass" if task == "multiclass" else ""
        raise FileNotFoundError(f"{path} not found, run scripts/train.py{flag} first")

    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["versions"]["scikit-learn"] != sklearn.__version__:
        warnings.warn(f"{path.name} was saved with scikit-learn {meta['versions']['scikit-learn']}, "
                      f"this is {sklearn.__version__}; retrain if predictions look wrong")
    return joblib.load(path), meta


def score_flows(df: pd.DataFrame, model: Pipeline, threshold: float,
                category_model: Pipeline | None = None) -> pd.DataFrame:
    """One row per flow: attack score, alert (score >= threshold), and the predicted category
    if `category_model` is given. The id column is kept when present."""
    check_columns(df)
    X = df[FEATURES]
    scores = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({"score": scores, "alert": scores >= threshold}, index=df.index)
    if ID_COL in df.columns:
        out.insert(0, ID_COL, df[ID_COL])
    if category_model is not None:
        out["category"] = category_model.predict(X)
    return out
