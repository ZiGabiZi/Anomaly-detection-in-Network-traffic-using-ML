"""Loading the UNSW-NB15 train/test CSVs from data/raw/."""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# src/netad/data.py -> project root is two levels above the package
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
DATASET_URL = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"

FILES = {
    "train": "UNSW_NB15_training-set.csv",
    "test": "UNSW_NB15_testing-set.csv",
}

ID_COL = "id"
BINARY_TARGET = "label"
MULTI_TARGET = "attack_cat"

# Column lists are explicit on purpose: with pandas 3 text columns have dtype
# "str", so dtype-based selection (select_dtypes) is not reliable.
CATEGORICAL = ["proto", "service", "state"]
NUMERIC = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
    "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
    "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
    "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
    "ct_srv_dst", "is_sm_ips_ports",
]

FEATURES = CATEGORICAL + NUMERIC
TARGETS = [BINARY_TARGET, MULTI_TARGET]
EXPECTED_COLUMNS = [ID_COL, *FEATURES, *TARGETS]


def load_unsw_frame(split: str, data_dir: Path | str | None = None) -> pd.DataFrame:
    """Returns the full CSV for `split` ("train" or "test"), all columns included."""
    if split not in FILES:
        raise ValueError(f"split must be one of {list(FILES)}, got {split!r}")

    path = Path(data_dir or DATA_DIR) / FILES[split]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Download {FILES[split]} from {DATASET_URL} "
            f"and place it in {path.parent}."
        )

    # The CSVs start with a UTF-8 BOM; without utf-8-sig the first column is "\ufeffid".
    df = pd.read_csv(path, encoding="utf-8-sig")

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")
    return df


def load_unsw(split: str, data_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y_binary, y_multi) for `split`; X holds only the 42 features."""
    df = load_unsw_frame(split, data_dir)
    return df[FEATURES], df[BINARY_TARGET], df[MULTI_TARGET]


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows identical on features and both labels (`id` is ignored, it is always unique)."""
    return df.drop_duplicates(subset=FEATURES + TARGETS)


def resolve_conflicts(df: pd.DataFrame, target: str = MULTI_TARGET, policy: str = "majority") -> pd.DataFrame:
    """Keep a single `target` value per feature vector. Train only, call before deduplicate.

    "drop": remove every vector that has more than one value.
    "majority": keep the most frequent value, drop the vector on ties.
    """
    if policy not in ("drop", "majority"):
        raise ValueError(f"policy must be 'drop' or 'majority', got {policy!r}")

    counts = df.groupby(FEATURES + [target]).size()
    if policy == "drop":
        ok = counts.groupby(level=FEATURES).transform("size") == 1
    else:
        is_top = counts == counts.groupby(level=FEATURES).transform("max")
        ok = is_top & (is_top.groupby(level=FEATURES).transform("sum") == 1)

    keep = pd.MultiIndex.from_frame(df[FEATURES + [target]]).isin(ok[ok].index)
    return df[keep]


def overlap_with_train(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.Series:
    """Boolean mask over `test_df`: True where the feature vector also appears in `train_df`."""
    train_vectors = train_df[FEATURES].drop_duplicates()
    merged = test_df[FEATURES].merge(train_vectors, on=FEATURES, how="left", indicator=True)
    return pd.Series(merged["_merge"].eq("both").to_numpy(), index=test_df.index, name="in_train")


def distinct_rows(df: pd.DataFrame) -> pd.Series:
    """Boolean mask over `df`: True for the first row of each feature vector + attack_cat, False for repeats."""
    return (~df.duplicated(FEATURES + [MULTI_TARGET])).rename("distinct")


def train_val_split(
    train_df: pd.DataFrame, target: str, val_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Clean train for `target`, then a stratified train/validation split.

    Returns X_train, X_val, y_train, y_val.
    """
    df = resolve_conflicts(train_df, target)
    # one target value per vector now, so identical features = duplicate for this task
    df = df.drop_duplicates(subset=FEATURES)
    X_train, X_val, y_train, y_val = train_test_split(
        df[FEATURES], df[target],
        test_size=val_size, stratify=df[MULTI_TARGET], random_state=seed,
    )
    return X_train, X_val, y_train, y_val
