"""Validare temporală și pe grupuri pentru date panel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, cross_validate


@dataclass(frozen=True)
class ValidationDesign:
    """Descrierea fold-urilor utilizate în backend."""

    temporal_cv: list[tuple[np.ndarray, np.ndarray]]
    temporal_folds: list[dict[str, str | int]]
    group_cv: GroupKFold
    group_folds: int


def build_temporal_cv(years: pd.Series) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, str | int]]]:
    """Construiește fold-uri expanding-window bazate pe ani."""

    years_array = np.asarray(years)
    proposed = [(2012, 2013, 2014), (2014, 2015, 2016), (2016, 2017, 2018)]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    descriptions: list[dict[str, str | int]] = []
    for train_until, val_start, val_end in proposed:
        train_idx = np.where(years_array <= train_until)[0]
        val_idx = np.where((years_array >= val_start) & (years_array <= val_end))[0]
        if len(train_idx) > 20 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))
            descriptions.append(
                {
                    "Fold": len(folds),
                    "Train": f"ani <= {train_until}",
                    "Validation": f"{val_start}–{val_end}",
                    "N train": int(len(train_idx)),
                    "N validation": int(len(val_idx)),
                }
            )
    if not folds:
        unique_years = sorted(pd.Series(years_array).dropna().unique())
        for index in range(8, len(unique_years) - 1, 2):
            train_years = unique_years[:index]
            val_years = unique_years[index : index + 2]
            train_idx = np.where(np.isin(years_array, train_years))[0]
            val_idx = np.where(np.isin(years_array, val_years))[0]
            if len(train_idx) > 20 and len(val_idx) > 0:
                folds.append((train_idx, val_idx))
                descriptions.append(
                    {
                        "Fold": len(folds),
                        "Train": f"{min(train_years)}–{max(train_years)}",
                        "Validation": f"{min(val_years)}–{max(val_years)}",
                        "N train": int(len(train_idx)),
                        "N validation": int(len(val_idx)),
                    }
                )
    return folds, descriptions


def build_validation_design(years: pd.Series, groups: pd.Series) -> ValidationDesign:
    """Creează validarea temporală și GroupKFold pe țări."""

    temporal_cv, temporal_folds = build_temporal_cv(years)
    group_count = int(pd.Series(groups).nunique())
    group_folds = max(2, min(5, group_count))
    return ValidationDesign(
        temporal_cv=temporal_cv,
        temporal_folds=temporal_folds,
        group_cv=GroupKFold(n_splits=group_folds),
        group_folds=group_folds,
    )


def evaluate_temporal_cv(estimator, x_train, y_train, cv, scoring: str) -> float:
    """Evaluează modelul prin fold-urile temporale definite în backend."""

    if not cv:
        return np.nan
    with parallel_backend("threading"):
        scores = cross_validate(clone(estimator), x_train, y_train, cv=cv, scoring=scoring, error_score=np.nan)
    values = scores["test_score"]
    if scoring.startswith("neg_"):
        values = -values
    return float(np.nanmean(values))


def evaluate_group_cv(estimator, x_train, y_train, groups, cv, scoring: str) -> float:
    """Evaluează generalizarea către țări nevăzute în training."""

    with parallel_backend("threading"):
        scores = cross_validate(
            clone(estimator),
            x_train,
            y_train,
            groups=groups,
            cv=cv,
            scoring=scoring,
            error_score=np.nan,
        )
    values = scores["test_score"]
    if scoring.startswith("neg_"):
        values = -values
    return float(np.nanmean(values))
