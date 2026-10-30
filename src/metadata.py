"""
metadata.py
-----------
Patient-metadata preprocessing for the multimodal skin-lesion models.

The HAM10000 metadata CSV provides, per lesion image:
    image_id, dx, dx_type, age, sex, localization

For the *image + metadata* fusion models we turn three of those clinical
fields into a fixed-length numeric vector:

    age           -> median-imputed, then standardized (z-score)
    sex           -> one-hot  {female, male, unknown}
    localization  -> one-hot over the anatomical sites seen at fit time
                     (+ an explicit 'unknown' bucket for missing / unseen)

Design goals
============
* **No sklearn dependency** — the transform is a handful of lookups, so we
  implement it in plain numpy. That keeps the Streamlit app lightweight and
  makes the exact feature layout transparent for the write-up.
* **Fit on train only** — age statistics and the categorical vocabularies are
  learned from the training split and then frozen, so val/test/inference see
  no leakage.
* **Robust at inference** — the app lets a user leave a field blank or pick a
  site the training set never contained; anything unknown maps to the
  'unknown' bucket instead of crashing.
* **Serializable** — `save()/load()` round-trips through a small JSON file
  (`preprocessing.json`) that ships next to the trained models so the app
  reconstructs the *exact* same feature ordering.

Typical usage
=============
    proc = MetadataProcessor().fit(train_df)
    X_meta_train = proc.transform(train_df)          # (N, feature_dim)
    proc.save("models/preprocessing.json")
    ...
    proc = MetadataProcessor.load("models/preprocessing.json")
    x = proc.transform_one(age=55, sex="male", localization="back")  # (feature_dim,)
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

# HAM10000 uses the literal string "unknown" for missing categoricals; we reuse
# it as our catch-all bucket so real "unknown" rows and truly-missing values
# share one column instead of fragmenting the encoding.
UNKNOWN = "unknown"


class MetadataProcessor:
    """Fit/transform patient metadata into a fixed-length numeric vector.

    Attributes are intentionally simple (floats / lists of str) so the whole
    fitted state serializes to JSON without any custom encoders.
    """

    def __init__(self) -> None:
        self.age_median: float = 0.0
        self.age_mean: float = 0.0
        self.age_std: float = 1.0
        self.sex_categories: list[str] = []
        self.loc_categories: list[str] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "MetadataProcessor":
        """Learn age statistics and categorical vocabularies from a dataframe.

        Expects columns ``age``, ``sex`` and ``localization`` (HAM10000
        naming). Missing columns are tolerated and treated as all-unknown,
        which keeps the class usable on metadata-poor datasets too.
        """
        age = self._clean_age_series(df)
        # Median for imputation, mean/std (of the imputed series) for scaling.
        self.age_median = float(np.nanmedian(age)) if np.isfinite(age).any() else 0.0
        age_imputed = np.where(np.isfinite(age), age, self.age_median)
        self.age_mean = float(age_imputed.mean())
        std = float(age_imputed.std())
        # Guard against a degenerate (constant-age) column producing div-by-zero.
        self.age_std = std if std > 1e-6 else 1.0

        self.sex_categories = self._vocab(df, "sex")
        self.loc_categories = self._vocab(df, "localization")
        self._fitted = True
        return self

    # ------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Vectorize a whole dataframe -> ``(len(df), feature_dim)`` float32."""
        self._check_fitted()
        age = self._clean_age_series(df)
        age_imputed = np.where(np.isfinite(age), age, self.age_median)
        age_scaled = ((age_imputed - self.age_mean) / self.age_std).astype("float32")

        sex_oh = self._one_hot_series(df, "sex", self.sex_categories)
        loc_oh = self._one_hot_series(df, "localization", self.loc_categories)

        return np.concatenate(
            [age_scaled.reshape(-1, 1), sex_oh, loc_oh], axis=1
        ).astype("float32")

    def transform_one(
        self,
        age: float | None = None,
        sex: str | None = None,
        localization: str | None = None,
    ) -> np.ndarray:
        """Vectorize a single patient -> ``(feature_dim,)`` float32.

        Convenience path for the Streamlit app, where inputs arrive as loose
        Python values rather than a dataframe row. ``None`` / blank values fall
        back to the fitted median (age) or the 'unknown' bucket (categoricals).
        """
        row = {"age": age, "sex": sex, "localization": localization}
        return self.transform(pd.DataFrame([row]))[0]

    # ------------------------------------------------------------- (de)serialize
    def save(self, path: str) -> None:
        """Persist fitted state to JSON (ships beside the trained models)."""
        self._check_fitted()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "age_median": self.age_median,
                    "age_mean": self.age_mean,
                    "age_std": self.age_std,
                    "sex_categories": self.sex_categories,
                    "loc_categories": self.loc_categories,
                    "feature_names": self.feature_names,
                    "version": 1,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "MetadataProcessor":
        with open(path, encoding="utf-8") as f:
            state: dict[str, Any] = json.load(f)
        proc = cls()
        proc.age_median = float(state["age_median"])
        proc.age_mean = float(state["age_mean"])
        proc.age_std = float(state["age_std"])
        proc.sex_categories = list(state["sex_categories"])
        proc.loc_categories = list(state["loc_categories"])
        proc._fitted = True
        return proc

    # --------------------------------------------------------------- properties
    @property
    def feature_dim(self) -> int:
        """Length of the emitted vector: 1 (age) + |sex| + |localization|."""
        self._check_fitted()
        return 1 + len(self.sex_categories) + len(self.loc_categories)

    @property
    def feature_names(self) -> list[str]:
        """Human-readable name per output column (handy for figures/tables)."""
        self._check_fitted()
        names = ["age_z"]
        names += [f"sex={c}" for c in self.sex_categories]
        names += [f"loc={c}" for c in self.loc_categories]
        return names

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _clean_age_series(df: pd.DataFrame) -> np.ndarray:
        """Return age as a float array with non-finite for missing values."""
        if "age" not in df.columns:
            return np.full(len(df), np.nan, dtype="float64")
        return pd.to_numeric(df["age"], errors="coerce").to_numpy(dtype="float64")

    @staticmethod
    def _norm_cat(value: Any) -> str:
        """Normalize a raw categorical cell to a clean lowercase token."""
        if value is None:
            return UNKNOWN
        # pandas NaN is a float; catch it before str().
        if isinstance(value, float) and np.isnan(value):
            return UNKNOWN
        text = str(value).strip().lower()
        return text if text and text != "nan" else UNKNOWN

    def _vocab(self, df: pd.DataFrame, col: str) -> list[str]:
        """Sorted category vocabulary for ``col``, always including 'unknown'."""
        values = set()
        if col in df.columns:
            values = {self._norm_cat(v) for v in df[col].tolist()}
        values.add(UNKNOWN)
        # Sorted for a deterministic, reproducible column order.
        return sorted(values)

    def _one_hot_series(
        self, df: pd.DataFrame, col: str, categories: list[str]
    ) -> np.ndarray:
        """One-hot encode ``df[col]`` against a fixed vocabulary.

        Values outside ``categories`` (unseen sites, blanks) collapse to the
        'unknown' column so every row sums to exactly 1.
        """
        index = {c: i for i, c in enumerate(categories)}
        out = np.zeros((len(df), len(categories)), dtype="float32")
        raw = df[col].tolist() if col in df.columns else [None] * len(df)
        for r, value in enumerate(raw):
            token = self._norm_cat(value)
            out[r, index.get(token, index[UNKNOWN])] = 1.0
        return out

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "MetadataProcessor is not fitted. Call .fit(train_df) or "
                "MetadataProcessor.load(path) first."
            )
