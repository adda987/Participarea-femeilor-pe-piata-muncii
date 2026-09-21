"""Construirea țintei de clasificare fără scurgere de informație."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CLASS_LABELS = ["Participare scăzută", "Participare medie", "Participare ridicată"]
CLASS_TO_CODE = {label: index for index, label in enumerate(CLASS_LABELS)}
CODE_TO_CLASS = {index: label for label, index in CLASS_TO_CODE.items()}


@dataclass(frozen=True)
class ClassificationTarget:
    """Pragurile și seriile codificate pentru clasificare."""

    q33: float
    q67: float
    y_train_labels: pd.Series
    y_test_labels: pd.Series
    y_train_codes: pd.Series
    y_test_codes: pd.Series
    distribution_train: pd.DataFrame
    distribution_test: pd.DataFrame
    warning: str | None


def assign_class(values: pd.Series, q33: float, q67: float) -> pd.Series:
    """Aplică pragurile fixe calculate pe train."""

    labels = np.select(
        [values < q33, values > q67],
        ["Participare scăzută", "Participare ridicată"],
        default="Participare medie",
    )
    return pd.Series(labels, index=values.index, name="participation_class")


def distribution_table(labels: pd.Series, sample: str) -> pd.DataFrame:
    """Returnează număr și procent pe clasă."""

    counts = labels.value_counts().reindex(CLASS_LABELS, fill_value=0)
    total = counts.sum()
    return pd.DataFrame(
        {
            "Eșantion": sample,
            "Clasă": counts.index,
            "Observații": counts.values,
            "Procent": np.where(total > 0, counts.values / total * 100, np.nan),
        }
    )


def build_classification_target(y_train: pd.Series, y_test: pd.Series) -> ClassificationTarget:
    """Calculează tertilele exclusiv pe training și aplică pragurile pe test."""

    q33 = float(y_train.quantile(1 / 3))
    q67 = float(y_train.quantile(2 / 3))
    y_train_labels = assign_class(y_train, q33, q67)
    y_test_labels = assign_class(y_test, q33, q67)
    y_train_codes = y_train_labels.map(CLASS_TO_CODE).astype(int)
    y_test_codes = y_test_labels.map(CLASS_TO_CODE).astype(int)
    distribution_train = distribution_table(y_train_labels, "Train")
    distribution_test = distribution_table(y_test_labels, "Test")
    small_classes = pd.concat([distribution_train, distribution_test])
    too_small = small_classes.loc[small_classes["Observații"] < 5]
    warning = None
    if not too_small.empty:
        warning = "Cel puțin o clasă are mai puțin de 5 observații într-un eșantion; evaluarea clasificării trebuie interpretată prudent."
    return ClassificationTarget(
        q33=q33,
        q67=q67,
        y_train_labels=y_train_labels,
        y_test_labels=y_test_labels,
        y_train_codes=y_train_codes,
        y_test_codes=y_test_codes,
        distribution_train=distribution_train,
        distribution_test=distribution_test,
        warning=warning,
    )

