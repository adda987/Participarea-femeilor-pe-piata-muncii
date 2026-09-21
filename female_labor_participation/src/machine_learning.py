"""Structuri pentru fluxul de Machine Learning."""

from __future__ import annotations


def ml_workflow_steps() -> list[str]:
    """Returnează fluxul planificat pentru modelarea predictivă."""

    return [
        "formularea problemei predictive",
        "selectarea caracteristicilor",
        "preprocesare",
        "tratarea valorilor lipsă",
        "standardizare",
        "codificare",
        "împărțire temporală pentru antrenare și testare",
        "validare în timp",
        "validare pe țări",
        "prevenirea scurgerii de informație între antrenare și testare",
    ]


def ml_model_catalog() -> list[dict[str, str]]:
    """Returnează modelele predictive planificate, fără estimare."""

    return [
        {"Model": "Regresie liniară de bază", "Stadiu": "planificat"},
        {"Model": "Regresie cu penalizare Ridge", "Stadiu": "planificat"},
        {"Model": "Regresie cu penalizare Lasso", "Stadiu": "planificat"},
        {"Model": "Rețea elastică", "Stadiu": "planificat"},
        {"Model": "Pădure aleatoare", "Stadiu": "planificat"},
        {"Model": "Amplificare prin gradient", "Stadiu": "planificat"},
        {"Model": "XGBoost", "Stadiu": "de analizat numai dacă este justificat"},
    ]


def ml_metrics() -> list[str]:
    """Returnează metricile predictive planificate."""

    return [
        "MAE",
        "MSE",
        "RMSE",
        "R-squared",
        "MAPE sau sMAPE, dacă este adecvat",
        "performanță pe antrenare",
        "performanță pe test",
        "diferența dintre antrenare și test",
    ]


def interpretability_methods() -> list[str]:
    """Returnează metodele planificate de interpretabilitate."""

    return [
        "coeficienți",
        "importanța caracteristicilor",
        "importanța prin permutare",
        "dependențe parțiale",
        "SHAP",
        "comparația cu rezultatele econometrice",
    ]
