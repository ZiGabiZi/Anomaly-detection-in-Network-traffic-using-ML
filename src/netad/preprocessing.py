"""Preprocessing."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from netad.data import CATEGORICAL, NUMERIC


def build_preprocessor(min_frequency: int = 100) -> ColumnTransformer:
    """log1p + scaling on numeric columns, one-hot on proto/service/state.

    Categories seen fewer than `min_frequency` times in train go to a shared "infrequent"
    column, unseen ones in test (e.g. state ACC, CLO) go there too. Other columns are dropped.
    """
    numeric = make_pipeline(
        FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
        StandardScaler(),
    )
    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=min_frequency,
        sparse_output=False,
    )
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])
